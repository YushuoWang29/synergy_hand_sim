# src/models/origami_design.py

import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


@dataclass
class Point2D:
    x: float
    y: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def distance(self, other: 'Point2D') -> float:
        return float(np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2))

    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x - other.x, self.y - other.y)

    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)

    def __truediv__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x / scalar, self.y / scalar)

    def __repr__(self):
        return f"Point2D({self.x:.3f}, {self.y:.3f})"

    def __eq__(self, other: 'Point2D') -> bool:
        return np.isclose(self.x, other.x) and np.isclose(self.y, other.y)

    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))


class FoldType(Enum):
    MOUNTAIN = "mountain"
    VALLEY = "valley"
    OUTLINE = "outline"


@dataclass
class FoldLine:
    id: int
    start: Point2D
    end: Point2D
    fold_type: FoldType
    stiffness: float = 1.0

    @property
    def length(self) -> float:
        return self.start.distance(self.end)

    @property
    def direction(self) -> np.ndarray:
        v = np.array([self.end.x - self.start.x, self.end.y - self.start.y])
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            raise ValueError(f"FoldLine {self.id}: start and end are too close")
        return v / norm

    @property
    def normal(self) -> np.ndarray:
        d = self.direction
        return np.array([-d[1], d[0]])

    @property
    def midpoint(self) -> Point2D:
        return Point2D(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2
        )

    @property
    def is_fold(self) -> bool:
        return self.fold_type in (FoldType.MOUNTAIN, FoldType.VALLEY)

    def point_along(self, t: float) -> Point2D:
        return Point2D(
            self.start.x + t * (self.end.x - self.start.x),
            self.start.y + t * (self.end.y - self.start.y)
        )


@dataclass
class OrigamiFace:
    id: int
    vertices: List[Point2D]
    edge_ids: List[int]

    def __post_init__(self):
        if len(self.vertices) != len(self.edge_ids):
            raise ValueError(
                f"Face {self.id}: vertices({len(self.vertices)}) "
                f"!= edge_ids({len(self.edge_ids)})"
            )

    @property
    def centroid(self) -> Point2D:
        if not self.vertices:
            return Point2D(0, 0)
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return Point2D(np.mean(xs), np.mean(ys))

    @property
    def area(self) -> float:
        if len(self.vertices) < 3:
            return 0.0
        total = 0.0
        n = len(self.vertices)
        for i in range(n):
            x1, y1 = self.vertices[i].x, self.vertices[i].y
            x2, y2 = self.vertices[(i + 1) % n].x, self.vertices[(i + 1) % n].y
            total += x1 * y2 - x2 * y1
        return abs(total) / 2

    def contains_point(self, point: Point2D) -> bool:
        n = len(self.vertices)
        if n < 3:
            return False
        inside = False
        for i in range(n):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % n]
            if ((v1.y > point.y) != (v2.y > point.y)):
                x_intersect = v1.x + (point.y - v1.y) * (v2.x - v1.x) / (v2.y - v1.y)
                if point.x < x_intersect:
                    inside = not inside
        return inside


@dataclass
class JointConnection:
    id: int
    fold_line_id: int
    face_a_id: int
    face_b_id: int
    fold_type: FoldType

    @property
    def offset_direction(self) -> int:
        if self.fold_type == FoldType.VALLEY:
            return 1
        elif self.fold_type == FoldType.MOUNTAIN:
            return -1
        else:
            return 0


@dataclass
class Pulley:
    id: int
    position: Point2D
    radius: float = 3.5
    friction_coefficient: float = 0.3
    attached_fold_line_id: Optional[int] = None


@dataclass
class Hole:
    """
    孔：用于孔类腱绳传动的路径节点。
    
    ID 约定：孔使用 <= -100 的负整数（-1,-2 已被驱动器占用）
    
    物理模型：腱绳穿过孔跨过折痕，通过驱动臂产生扭矩。
    驱动臂随折叠角度变化：arm(q=0) = plate_offset (高度h)
    
    参数说明：
        id : int (<= -100)
        position : 孔在2D板面上的位置
        attached_fold_line_id : 关联折痕ID（孔位于折痕的哪一侧）
        face_id : 孔所在的面片ID（用于确定孔在折痕的哪一侧）
        plate_offset : 板b上表面到折痕平面的高度 h（≈ 材料厚度）
        friction_coefficient : 孔对腱绳的摩擦系数
        radius : 孔的半径（用于摩擦模型和碰撞检测）
    """
    id: int
    position: Point2D
    attached_fold_line_id: Optional[int] = None
    face_id: Optional[int] = None
    plate_offset: float = 3.0
    friction_coefficient: float = 0.3
    radius: float = 1.5


# ID 类型识别常量
HOLE_ID_OFFSET = -100  # 孔使用 <= -100 的 ID
ACTUATOR_A_ID = -1
ACTUATOR_B_ID = -2


def is_hole_id(element_id: int) -> bool:
    """判断 ID 是否为孔"""
    return element_id <= HOLE_ID_OFFSET


def is_actuator_id(element_id: int) -> bool:
    """判断 ID 是否为驱动器"""
    return element_id in (ACTUATOR_A_ID, ACTUATOR_B_ID)


def is_pulley_id(element_id: int) -> bool:
    """判断 ID 是否为滑轮"""
    return element_id > 0


def element_type_name(element_id: int) -> str:
    """返回元素类型名称"""
    if is_pulley_id(element_id):
        return "pulley"
    elif is_actuator_id(element_id):
        return "actuator"
    elif is_hole_id(element_id):
        return "hole"
    return "unknown"


@dataclass
class Tendon:
    id: int
    pulley_sequence: List[int]
    
    @property
    def has_holes(self) -> bool:
        """检查腱绳路径是否包含孔元素"""
        return any(is_hole_id(eid) for eid in self.pulley_sequence)
    
    def get_element_info(self) -> List[Tuple[str, int]]:
        """
        返回路径中每个元素的类型和ID
        
        Returns:
            [(type_name, id), ...] 类型为 "pulley"/"hole"/"actuator"
        """
        return [(element_type_name(eid), eid) for eid in self.pulley_sequence]


@dataclass
class Actuator:
    id: int
    name: str
    displacement: float = 0.0


class OrigamiHandDesign:
    """
    完整的折纸手设计
    """

    def __init__(self, name: str = "origami_hand", material_thickness: float = 3.0):
        self.name = name
        self.material_thickness = material_thickness

        self.fold_lines: dict[int, FoldLine] = {}
        self.faces: dict[int, OrigamiFace] = {}
        self.joints: list[JointConnection] = []
        self.root_face_id: Optional[int] = None
        self.face_parent: dict[int, int] = {}

        self.pulleys: dict[int, Pulley] = {}
        self.holes: dict[int, Hole] = {}
        self.tendons: dict[int, Tendon] = {}
        self.actuators: dict[int, Actuator] = {}
        self.actuator_positions: List[dict] = []

        self.actuators[0] = Actuator(0, "Motor_A")
        self.actuators[1] = Actuator(1, "Motor_B")

    # ========= 添加方法 =========

    def add_fold_line(self, line: FoldLine):
        if line.id in self.fold_lines:
            raise ValueError(f"Fold line {line.id} already exists")
        self.fold_lines[line.id] = line

    def add_face(self, face: OrigamiFace):
        if face.id in self.faces:
            raise ValueError(f"Face {face.id} already exists")
        self.faces[face.id] = face

    def add_joint(self, joint: JointConnection):
        if joint.face_a_id not in self.faces:
            raise ValueError(f"Joint {joint.id}: face_a {joint.face_a_id} not found")
        if joint.face_b_id not in self.faces:
            raise ValueError(f"Joint {joint.id}: face_b {joint.face_b_id} not found")
        if joint.fold_line_id not in self.fold_lines:
            raise ValueError(f"Joint {joint.id}: fold_line {joint.fold_line_id} not found")
        self.joints.append(joint)

    def add_pulley(self, pulley: Pulley):
        if pulley.id in self.pulleys:
            raise ValueError(f"Pulley {pulley.id} already exists")
        self.pulleys[pulley.id] = pulley

    def add_hole(self, hole: Hole):
        if hole.id in self.holes:
            raise ValueError(f"Hole {hole.id} already exists")
        if not is_hole_id(hole.id):
            raise ValueError(f"Hole ID must be <= {HOLE_ID_OFFSET}, got {hole.id}")
        self.holes[hole.id] = hole

    def add_tendon(self, tendon: Tendon):
        if tendon.id in self.tendons:
            raise ValueError(f"Tendon {tendon.id} already exists")
        self.tendons[tendon.id] = tendon

    def set_root_face(self, face_id: int):
        if face_id not in self.faces:
            raise ValueError(f"Face {face_id} not found")
        self.root_face_id = face_id

    # ========= 查询方法 =========

    def get_folds(self) -> list[FoldLine]:
        return [l for l in self.fold_lines.values() if l.is_fold]

    def get_outlines(self) -> list[FoldLine]:
        return [l for l in self.fold_lines.values() if l.fold_type == FoldType.OUTLINE]

    def get_face_by_id(self, face_id: int) -> OrigamiFace:
        return self.faces[face_id]

    def get_joints_for_face(self, face_id: int) -> list[JointConnection]:
        return [j for j in self.joints
                if j.face_a_id == face_id or j.face_b_id == face_id]

    def get_parent_joint(self, face_id: int) -> Optional[JointConnection]:
        if face_id not in self.face_parent:
            return None
        parent_id = self.face_parent[face_id]
        for j in self.joints:
            if {j.face_a_id, j.face_b_id} == {face_id, parent_id}:
                return j
        return None

    def get_actuator(self, actuator_id: int) -> Actuator:
        if actuator_id not in self.actuators:
            raise ValueError(f"Actuator {actuator_id} not found")
        return self.actuators[actuator_id]

    def get_joint_stiffness_list(self) -> list[float]:
        stiffmess_values = []
        for joint in self.joints:
            fold = self.fold_lines[joint.fold_line_id]
            if fold:
                stiffmess_values.append(fold.stiffness)
            else:
                stiffmess_values.append(1.0)
        return stiffmess_values

    # ========= 构建拓扑 =========

    def build_topology(self):
        """
        根据已有的 fold_lines 自动识别面片并生成关节连接。
        完成后设置 root_face_id, face_parent, 并填充 faces 和 joints 列表。
        """
        from .origami_parser import (
            split_at_intersections,
            build_graph_from_segments,
            find_minimal_cycles,
            remove_outer_face,
            rebuild_design_topology
        )

        segments = []
        for fl in self.fold_lines.values():
            segments.append({
                'start': np.array([fl.start.x, fl.start.y]),
                'end': np.array([fl.end.x, fl.end.y]),
                'fold_type': fl.fold_type,
                'length': fl.length
            })

        split_segs = split_at_intersections(segments, tolerance=2.0)
        nodes, edges = build_graph_from_segments(split_segs)
        faces = find_minimal_cycles(nodes, edges)
        faces = remove_outer_face(faces, nodes)

        # 重建 fold_lines: 添加图中新生成的边（已被分割后的子线段）
        self.fold_lines.clear()
        for edge in edges:
            u, v = edge['u'], edge['v']
            p1 = Point2D(float(nodes[u][0]), float(nodes[u][1]))
            p2 = Point2D(float(nodes[v][0]), float(nodes[v][1]))
            fl = FoldLine(id=edge['id'], start=p1, end=p2,
                          fold_type=edge['fold_type'], stiffness=1.0)
            self.add_fold_line(fl)

        self.faces.clear()
        self.joints.clear()
        rebuild_design_topology(self, nodes, edges, faces)

        if self.root_face_id is not None:
            self.build_face_tree()

    # ========= 构建面片树 =========

    def build_face_tree(self):
        if self.root_face_id is None:
            raise ValueError("Must set root_face_id before building tree")

        self.face_parent = {}
        visited = {self.root_face_id}
        queue = [self.root_face_id]

        adjacency = {fid: set() for fid in self.faces}
        for joint in self.joints:
            adjacency[joint.face_a_id].add(joint.face_b_id)
            adjacency[joint.face_b_id].add(joint.face_a_id)

        while queue:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.face_parent[neighbor] = current
                    queue.append(neighbor)

        unreachable = set(self.faces.keys()) - visited
        if unreachable:
            print(f"Warning: {len(unreachable)} faces not connected to root:")
            for fid in unreachable:
                print(f"  Face {fid}")

    # ========= 验证 =========

    def validate(self) -> bool:
        errors = []

        if self.root_face_id is None:
            errors.append("root_face_id is not set")
        elif self.root_face_id not in self.faces:
            errors.append(f"root_face_id {self.root_face_id} not in faces")

        for line_id, line in self.fold_lines.items():
            if not line.is_fold:
                continue
            connected_faces = []
            for joint in self.joints:
                if joint.fold_line_id == line_id:
                    connected_faces.extend([joint.face_a_id, joint.face_b_id])
            if len(set(connected_faces)) != 2:
                errors.append(
                    f"Fold line {line_id} ({line.fold_type.value}) "
                    f"connected to {len(set(connected_faces))} faces"
                )

        for face_id, face in self.faces.items():
            for edge_id in face.edge_ids:
                if edge_id not in self.fold_lines:
                    errors.append(f"Face {face_id} references non-existent fold_line {edge_id}")

        if errors:
            print("Validation errors:")
            for e in errors:
                print(f"  - {e}")
            return False

        print("Validation passed ✓")
        return True

    # ========= 输出 =========

    def summary(self):
        print(f"OrigamiHand: {self.name}")
        print(f"  Fold lines: {len(self.fold_lines)} "
              f"({len(self.get_folds())} folds, {len(self.get_outlines())} outlines)")
        print(f"  Faces: {len(self.faces)}")
        print(f"  Joints: {len(self.joints)}")
        print(f"  Root face: {self.root_face_id}")
        print(f"  Face tree depth: {self._max_tree_depth()}")

    def _max_tree_depth(self) -> int:
        if not self.face_parent:
            return 0

        def depth(face_id):
            if face_id == self.root_face_id:
                return 0
            return 1 + depth(self.face_parent[face_id])

        return max(depth(fid) for fid in self.faces)

    def to_dict(self) -> dict:
        design_dict = {
            "name": self.name,
            "material_thickness": self.material_thickness,
            "fold_lines": [
                {
                    "id": l.id,
                    "start": {"x": l.start.x, "y": l.start.y},
                    "end": {"x": l.end.x, "y": l.end.y},
                    "fold_type": l.fold_type.value,
                    "stiffness": l.stiffness
                } for l in self.fold_lines.values()
            ],
            "faces": [
                {
                    "id": f.id,
                    "vertices": [{"x": v.x, "y": v.y} for v in f.vertices],
                    "edge_ids": f.edge_ids
                } for f in self.faces.values()
            ],
            "joints": [
                {
                    "id": j.id,
                    "fold_line_id": j.fold_line_id,
                    "face_a_id": j.face_a_id,
                    "face_b_id": j.face_b_id,
                    "fold_type": j.fold_type.value
                } for j in self.joints
            ],
            "root_face_id": self.root_face_id,
            "face_parent": {str(k): v for k, v in self.face_parent.items()},
            "pulleys": [
                {
                    "id": p.id,
                    "position": {"x": p.position.x, "y": p.position.y},
                    "radius": p.radius,
                    "friction_coefficient": p.friction_coefficient,
                    "attached_fold_line_id": p.attached_fold_line_id
                } for p in self.pulleys.values()
            ],
            "holes": [
                {
                    "id": h.id,
                    "position": {"x": h.position.x, "y": h.position.y},
                    "attached_fold_line_id": h.attached_fold_line_id,
                    "face_id": h.face_id,
                    "plate_offset": h.plate_offset,
                    "friction_coefficient": h.friction_coefficient,
                    "radius": h.radius
                } for h in self.holes.values()
            ],
            "tendons": [
                {
                    "id": t.id,
                    "pulley_sequence": t.pulley_sequence,
                } for t in self.tendons.values()
            ],
            "actuators": [
                {
                    "id": a.id,
                    "name": a.name,
                    "displacement": a.displacement
                } for a in self.actuators.values()
            ],
            "actuator_positions": self.actuator_positions
        }
        return design_dict

    @classmethod
    def from_dict(cls, d: dict) -> 'OrigamiHandDesign':
        design = cls(
            name=d.get("name", "imported"),
            material_thickness=d.get("material_thickness", 3.0)
        )

        for fl in d.get("fold_lines", []):
            start = Point2D(fl["start"]["x"], fl["start"]["y"])
            end = Point2D(fl["end"]["x"], fl["end"]["y"])
            fold_type = FoldType(fl["fold_type"])
            stiffness = fl.get("stiffness", 1.0)
            design.add_fold_line(FoldLine(fl["id"], start, end, fold_type, stiffness))

        for face in d.get("faces", []):
            vertices = [Point2D(v["x"], v["y"]) for v in face["vertices"]]
            design.add_face(OrigamiFace(face["id"], vertices, face["edge_ids"]))

        for jnt in d.get("joints", []):
            design.add_joint(JointConnection(
                jnt["id"], jnt["fold_line_id"],
                jnt["face_a_id"], jnt["face_b_id"],
                FoldType(jnt["fold_type"])
            ))

        design.root_face_id = d.get("root_face_id")
        design.face_parent = {int(k): v for k, v in d.get("face_parent", {}).items()}

        for p in d.get("pulleys", []):
            pos = Point2D(p["position"]["x"], p["position"]["y"])
            design.add_pulley(Pulley(
                p["id"], pos, p.get("radius", 3.5),
                p.get("friction_coefficient", 0.3),
                p.get("attached_fold_line_id")
            ))

        for h in d.get("holes", []):
            pos = Point2D(h["position"]["x"], h["position"]["y"])
            design.add_hole(Hole(
                id=h["id"],
                position=pos,
                attached_fold_line_id=h.get("attached_fold_line_id"),
                face_id=h.get("face_id"),
                plate_offset=h.get("plate_offset", 3.0),
                friction_coefficient=h.get("friction_coefficient", 0.3),
                radius=h.get("radius", 1.5)
            ))

        for t in d.get("tendons", []):
            design.add_tendon(Tendon(t["id"], t["pulley_sequence"]))

        for a in d.get("actuators", []):
            design.actuators[a["id"]] = Actuator(
                a["id"], a["name"], a.get("displacement", 0.0)
            )

        design.actuator_positions = d.get("actuator_positions", [])
        return design

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'OrigamiHandDesign':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
