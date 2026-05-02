# src/models/origami_design.py

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

@dataclass
class Point2D:
    """二维点，所有几何计算的基础"""
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
        # 四舍五入到6位小数用于哈希
        return hash((round(self.x, 6), round(self.y, 6)))
    

class FoldType(Enum):
    MOUNTAIN = "mountain"  # 峰折：关节在板的下表面
    VALLEY = "valley"      # 谷折：关节在板的上表面
    OUTLINE = "outline"    # 轮廓线：不是关节，只是边界


@dataclass
class FoldLine:
    """一条折痕或轮廓线"""
    id: int
    start: Point2D
    end: Point2D
    fold_type: FoldType
    stiffness: float = 1.0  # 可选：折痕的刚度(Nm/rad)

    @property
    def length(self) -> float:
        return self.start.distance(self.end)
    
    @property
    def direction(self) -> np.ndarray:
        """单位方向向量"""
        v = np.array([self.end.x - self.start.x, self.end.y - self.start.y])
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            raise ValueError(f"FoldLine {self.id}: start and end are too close")
        return v / norm
    
    @property
    def normal(self) -> np.ndarray:
        """在2D平面内的单位法向量（向右旋转90度）"""
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
        """这条线是可折叠的（峰或谷），还是纯轮廓"""
        return self.fold_type in (FoldType.MOUNTAIN, FoldType.VALLEY)
    
    def point_along(self, t: float) -> Point2D:
        """参数化取点：t=0是start，t=1是end"""
        return Point2D(
            self.start.x + t * (self.end.x - self.start.x),
            self.start.y + t * (self.end.y - self.start.y)
        )
    

@dataclass
class OrigamiFace:
    """
    一个面片：由轮廓线和折痕围成的封闭区域
    
    vertices 必须按顺序排列（顺时针或逆时针），
    相邻顶点之间的边对应一条折痕或轮廓线
    """
    id: int
    vertices: List[Point2D]  # 按顺序排列
    edge_ids: List[int]      # 每条边对应的FoldLine id，长度=len(vertices)
    
    def __post_init__(self):
        if len(self.vertices) != len(self.edge_ids):
            raise ValueError(
                f"Face {self.id}: vertices数量({len(self.vertices)}) "
                f"必须等于edge_ids数量({len(self.edge_ids)})"
            )
    
    @property
    def centroid(self) -> Point2D:
        """面片中心"""
        if not self.vertices:
            return Point2D(0, 0)
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return Point2D(np.mean(xs), np.mean(ys))
    
    @property
    def area(self) -> float:
        """用鞋带公式计算多边形面积"""
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
        """射线法判断点是否在面片内"""
        n = len(self.vertices)
        if n < 3:
            return False
        
        inside = False
        for i in range(n):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % n]
            
            # 检查射线是否穿过这条边
            if ((v1.y > point.y) != (v2.y > point.y)):
                x_intersect = v1.x + (point.y - v1.y) * (v2.x - v1.x) / (v2.y - v1.y)
                if point.x < x_intersect:
                    inside = not inside
        
        return inside
    
@dataclass
class JointConnection:
    """
    一个关节连接
    
    每条折痕在被折叠后会成为一个旋转关节。
    
    fold_line_id: 对应的折痕
    face_a_id: 参考面（通常是更靠近手掌的面）
    face_b_id: 活动面（通常是更远离手掌的面）
    
    关节轴方向就是折痕线的方向。
    关节轴的位置：
    - 峰折(MOUNTAIN): 在板的下表面，即face_a所在平面的下方
    - 谷折(VALLEY):   在板的上表面，即face_a所在平面的上方
    
    """
    id: int
    fold_line_id: int
    face_a_id: int
    face_b_id: int
    fold_type: FoldType
    
    @property
    def offset_direction(self) -> int:
        """
        关节偏移方向
        +1: 向上偏移（谷折，关节在板上表面）
        -1: 向下偏移（峰折，关节在板下表面）
        """
        if self.fold_type == FoldType.VALLEY:
            return 1
        elif self.fold_type == FoldType.MOUNTAIN:
            return -1
        else:
            return 0
        

class OrigamiHandDesign:
    """
    完整的折纸手设计
    
    包含：
    - 所有折痕线（fold_lines）
    - 所有面片（faces）
    - 所有关节连接（joints）
    - 面片之间的父子关系（face_tree）
    - 根部面片（手掌）
    """
    
    def __init__(self, name: str = "origami_hand"):
        self.name = name
        
        self.fold_lines: dict[int, FoldLine] = {}
        self.faces: dict[int, OrigamiFace] = {}
        self.joints: list[JointConnection] = []
        
        self.root_face_id: Optional[int] = None
        self.face_parent: dict[int, int] = {}  # face_id -> parent_face_id
    
    # ========= 添加方法 =========
    
    def add_fold_line(self, line: FoldLine):
        """添加一条折痕线"""
        if line.id in self.fold_lines:
            raise ValueError(f"Fold line {line.id} already exists")
        self.fold_lines[line.id] = line
    
    def add_face(self, face: OrigamiFace):
        """添加一个面片"""
        if face.id in self.faces:
            raise ValueError(f"Face {face.id} already exists")
        self.faces[face.id] = face
    
    def add_joint(self, joint: JointConnection):
        """添加一个关节连接"""
        # 检查引用的面片和折痕是否存在
        if joint.face_a_id not in self.faces:
            raise ValueError(f"Joint {joint.id}: face_a {joint.face_a_id} not found")
        if joint.face_b_id not in self.faces:
            raise ValueError(f"Joint {joint.id}: face_b {joint.face_b_id} not found")
        if joint.fold_line_id not in self.fold_lines:
            raise ValueError(f"Joint {joint.id}: fold_line {joint.fold_line_id} not found")
        
        self.joints.append(joint)
    
    def set_root_face(self, face_id: int):
        """设置根部面片（手掌）"""
        if face_id not in self.faces:
            raise ValueError(f"Face {face_id} not found")
        self.root_face_id = face_id
    
    # ========= 查询方法 =========
    
    def get_folds(self) -> list[FoldLine]:
        """获取所有可折叠线（峰折+谷折）"""
        return [l for l in self.fold_lines.values() if l.is_fold]
    
    def get_outlines(self) -> list[FoldLine]:
        """获取所有轮廓线"""
        return [l for l in self.fold_lines.values() if l.fold_type == FoldType.OUTLINE]
    
    def get_face_by_id(self, face_id: int) -> OrigamiFace:
        return self.faces[face_id]
    
    def get_joints_for_face(self, face_id: int) -> list[JointConnection]:
        """获取与某个面片相关的所有关节"""
        return [j for j in self.joints 
                if j.face_a_id == face_id or j.face_b_id == face_id]
    
    def get_parent_joint(self, face_id: int) -> Optional[JointConnection]:
        """获取连接某个面片到其父面片的关节"""
        if face_id not in self.face_parent:
            return None
        parent_id = self.face_parent[face_id]
        for j in self.joints:
            if {j.face_a_id, j.face_b_id} == {face_id, parent_id}:
                return j
        return None
    
    def get_joint_stiffness_list(self) -> list[float]:
        """
        获取所有关节的刚度列表，顺序与 self.joints 一致
        如果某个关节对应的折痕没有设置刚度，则默认为 1.0
        用于构建对角刚度矩阵E
        """
        stiffmess_values = []
        for joint in self.joints:
            fold = self.fold_lines[joint.fold_line_id]
            if fold:
                stiffmess_values.append(fold.stiffness)
            else:
                stiffmess_values.append(1.0)
        return stiffmess_values
    # ========= 分析：构建面片树 =========
    
    def build_face_tree(self):
        """
        根据关节连接关系，构建面片的树状结构
        
        从root_face出发，BFS遍历所有关节，
        确定每个面片的父面片。
        """
        if self.root_face_id is None:
            raise ValueError("Must set root_face_id before building tree")
        
        self.face_parent = {}
        visited = {self.root_face_id}
        queue = [self.root_face_id]
        
        # 建立邻接表
        adjacency = {fid: set() for fid in self.faces}
        for joint in self.joints:
            adjacency[joint.face_a_id].add(joint.face_b_id)
            adjacency[joint.face_b_id].add(joint.face_a_id)
        
        # BFS
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.face_parent[neighbor] = current
                    queue.append(neighbor)
        
        # 检查是否所有面片都可达
        unreachable = set(self.faces.keys()) - visited
        if unreachable:
            print(f"Warning: {len(unreachable)} faces are not connected to root:")
            for fid in unreachable:
                print(f"  Face {fid}")
    
    # ========= 验证 =========
    
    def validate(self) -> bool:
        """验证设计的完整性"""
        errors = []
        
        # 1. 检查root_face是否设置
        if self.root_face_id is None:
            errors.append("root_face_id is not set")
        elif self.root_face_id not in self.faces:
            errors.append(f"root_face_id {self.root_face_id} not in faces")
        
        # 2. 检查每条折痕是否连接两个面
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
                    f"is connected to {len(set(connected_faces))} faces"
                )
        
        # 3. 检查面片的边是否都对应已有的fold_line
        for face_id, face in self.faces.items():
            for edge_id in face.edge_ids:
                if edge_id not in self.fold_lines:
                    errors.append(
                        f"Face {face_id} references non-existent "
                        f"fold_line {edge_id}"
                    )
        
        if errors:
            print("Validation errors:")
            for e in errors:
                print(f"  - {e}")
            return False
        
        print("Validation passed ✓")
        return True
    
    # ========= 输出 =========
    
    def summary(self):
        """打印设计摘要"""
        print(f"OrigamiHand: {self.name}")
        print(f"  Fold lines: {len(self.fold_lines)} "
              f"({len(self.get_folds())} folds, {len(self.get_outlines())} outlines)")
        print(f"  Faces: {len(self.faces)}")
        print(f"  Joints: {len(self.joints)}")
        print(f"  Root face: {self.root_face_id}")
        print(f"  Face tree depth: {self._max_tree_depth()}")
    
    def _max_tree_depth(self) -> int:
        """计算面片树的最大深度"""
        if not self.face_parent:
            return 0
        
        def depth(face_id):
            if face_id == self.root_face_id:
                return 0
            return 1 + depth(self.face_parent[face_id])
        
        return max(depth(fid) for fid in self.faces)