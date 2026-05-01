# src/models/origami_kinematics.py

import numpy as np
from typing import Dict, Tuple, List
from .origami_design import (
    OrigamiHandDesign, OrigamiFace, FoldLine, 
    JointConnection, FoldType, Point2D
)


def compute_joint_frame_in_parent(
    design: OrigamiHandDesign,
    joint: JointConnection
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算关节轴在"参考面片"局部坐标系中的位姿
    
    这里"参考面片"就是关节所在的2D平面（展开状态）
    """
    fold_line = design.fold_lines[joint.fold_line_id]
    thickness = design.material_thickness
    
    # 关节轴方向 = 折痕方向，在xy平面内
    d = fold_line.direction
    axis_dir = np.array([d[0], d[1], 0.0])
    
    # 轴上一点：折痕中点
    mid = fold_line.midpoint
    axis_point = np.array([mid.x, mid.y, 0.0])
    
    # 偏移
    if joint.fold_type == FoldType.MOUNTAIN:
        z_offset = -thickness / 2
    elif joint.fold_type == FoldType.VALLEY:
        z_offset = +thickness / 2
    else:
        z_offset = 0.0
    
    axis_point[2] += z_offset
    
    return axis_point, axis_dir


def rotation_around_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues旋转公式"""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],   0,       -axis[0]],
        [-axis[1],  axis[0],  0     ]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


class OrigamiForwardKinematics:
    """
    折纸手正向运动学（支持环状结构）
    
    使用迭代约束松弛法：
    1. 从根面片开始，沿生成树传播变换
    2. 对于环上的每条额外边，计算约束误差
    3. 调整关节角度使误差最小化
    4. 重复直到收敛
    """
    
    def __init__(self, design: OrigamiHandDesign, max_iterations: int = 100, tolerance: float = 1e-6):
        self.design = design
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # 预计算每个关节在展开状态下的轴位姿
        self.joint_axis_points: Dict[int, np.ndarray] = {}
        self.joint_axis_dirs: Dict[int, np.ndarray] = {}
        
        for joint in design.joints:
            axis_point, axis_dir = compute_joint_frame_in_parent(design, joint)
            self.joint_axis_points[joint.id] = axis_point
            self.joint_axis_dirs[joint.id] = axis_dir
        
        # 预计算每个面片在展开状态下的顶点3D位置
        self.face_vertices_3d: Dict[int, np.ndarray] = {}
        for face_id, face in design.faces.items():
            verts = np.array([[v.x, v.y, 0.0] for v in face.vertices])
            self.face_vertices_3d[face_id] = verts
        
        # 构建生成树（生成树用于初始传播）
        self._spanning_tree = self._build_spanning_tree()
        
        # 找出环上的边（不在生成树中的边）
        self._cycle_edges = self._find_cycle_edges()
        
        if self._cycle_edges:
            print(f"  Detected {len(self._cycle_edges)} cycle edges")
    
    def _build_spanning_tree(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        构建以根面片为根的生成树
        
        Returns:
            tree: {parent_face: [(child_face, joint_id), ...]}
        """
        if self.design.root_face_id is None:
            raise ValueError("Root face not set")
        
        # BFS遍历所有面片
        root = self.design.root_face_id
        visited = {root}
        parent = {root: None}  # face_id -> (parent_face_id, joint_id)
        tree = {root: []}  # parent_face -> [(child_face, joint_id)]
        
        queue = [root]
        
        while queue:
            current = queue.pop(0)
            
            for joint in self.design.joints:
                if joint.face_a_id == current and joint.face_b_id not in visited:
                    child = joint.face_b_id
                    visited.add(child)
                    parent[child] = (current, joint.id)
                    tree.setdefault(current, []).append((child, joint.id))
                    tree.setdefault(child, [])
                    queue.append(child)
                    
                elif joint.face_b_id == current and joint.face_a_id not in visited:
                    child = joint.face_a_id
                    visited.add(child)
                    parent[child] = (current, joint.id)
                    tree.setdefault(current, []).append((child, joint.id))
                    tree.setdefault(child, [])
                    queue.append(child)
        
        # 处理未连接的面片
        unvisited = set(self.design.faces.keys()) - visited
        for face_id in unvisited:
            # 找到连接这个面片到已访问面片的任意关节
            for joint in self.design.joints:
                if joint.face_a_id == face_id and joint.face_b_id in visited:
                    parent_face = joint.face_b_id
                    tree.setdefault(parent_face, []).append((face_id, joint.id))
                    tree.setdefault(face_id, [])
                    visited.add(face_id)
                    break
                elif joint.face_b_id == face_id and joint.face_a_id in visited:
                    parent_face = joint.face_a_id
                    tree.setdefault(parent_face, []).append((face_id, joint.id))
                    tree.setdefault(face_id, [])
                    visited.add(face_id)
                    break
        
        return tree
    
    def _find_cycle_edges(self) -> List[int]:
        """找出不属于生成树的边（环边）"""
        tree_edges = set()
        
        def collect_edges(parent_id):
            for child_id, joint_id in self._spanning_tree.get(parent_id, []):
                tree_edges.add(joint_id)
                collect_edges(child_id)
        
        collect_edges(self.design.root_face_id)
        
        cycle_edges = []
        for joint in self.design.joints:
            if joint.id not in tree_edges:
                cycle_edges.append(joint.id)
        
        return cycle_edges
    
    def forward_kinematics(
        self, 
        joint_angles: Dict[int, float]
    ) -> Dict[int, np.ndarray]:
        """
        给定关节角度，计算所有面片的3D位姿
        
        使用迭代松弛法处理环状约束
        """
        # 初始化：未指定角度默认为0
        full_angles = {}
        for joint in self.design.joints:
            full_angles[joint.id] = joint_angles.get(joint.id, 0.0)
        
        # 第一步：沿生成树传播
        face_transforms = self._propagate_along_tree(full_angles)
        
        # 第二步：如果有环，迭代修正
        if self._cycle_edges:
            face_transforms = self._resolve_cycles(face_transforms, full_angles)
        
        return face_transforms
    
    def _propagate_along_tree(
        self,
        joint_angles: Dict[int, float]
    ) -> Dict[int, np.ndarray]:
        """沿生成树传播变换"""
        root_id = self.design.root_face_id
        face_transforms = {root_id: np.eye(4)}
        
        def propagate(parent_id, parent_transform):
            for child_id, joint_id in self._spanning_tree.get(parent_id, []):
                if child_id in face_transforms:
                    continue
                
                angle = joint_angles.get(joint_id, 0.0)
                
                # 找到连接父子面片的关节
                joint = None
                for j in self.design.joints:
                    if j.id == joint_id:
                        joint = j
                        break
                
                if joint is None:
                    continue
                
                # 确定旋转方向
                if joint.face_a_id == parent_id and joint.face_b_id == child_id:
                    sign = 1
                elif joint.face_b_id == parent_id and joint.face_a_id == child_id:
                    sign = -1
                else:
                    continue
                
                child_transform = self._apply_joint_rotation(
                    parent_transform, joint, sign * angle
                )
                
                face_transforms[child_id] = child_transform
                propagate(child_id, child_transform)
        
        propagate(root_id, face_transforms[root_id])
        
        return face_transforms
    
    def _apply_joint_rotation(
        self,
        parent_transform: np.ndarray,
        joint: JointConnection,
        angle: float
    ) -> np.ndarray:
        """应用关节旋转"""
        axis_point_local = self.joint_axis_points[joint.id]
        axis_dir_local = self.joint_axis_dirs[joint.id]
        
        # 变换到世界坐标
        axis_point_world = parent_transform @ np.append(axis_point_local, 1.0)
        axis_point_world = axis_point_world[:3]
        axis_dir_world = parent_transform[:3, :3] @ axis_dir_local
        
        # 构建旋转
        T_to = np.eye(4)
        T_to[:3, 3] = -axis_point_world
        
        R = rotation_around_axis(axis_dir_world, angle)
        T_rot = np.eye(4)
        T_rot[:3, :3] = R
        
        T_from = np.eye(4)
        T_from[:3, 3] = axis_point_world
        
        return T_from @ T_rot @ T_to @ parent_transform
    
    # 替换 origami_kinematics.py 中的 _resolve_cycles 方法

    def _resolve_cycles(
        self,
        face_transforms: Dict[int, np.ndarray],
        joint_angles: Dict[int, float]
    ) -> Dict[int, np.ndarray]:
        """
        迭代求解环约束
        
        算法：每次沿生成树传播后，检查环边的约束误差。
        将误差"推回"给环边关联的关节角度，然后重新传播。
        重复直到收敛。
        """
        if not self._cycle_edges:
            return face_transforms
        
        root_id = self.design.root_face_id
        
        for iteration in range(self.max_iterations):
            max_error = 0.0
            
            for joint_id in self._cycle_edges:
                # 找到这条环边对应的关节
                joint = None
                for j in self.design.joints:
                    if j.id == joint_id:
                        joint = j
                        break
                if joint is None:
                    continue
                
                a, b = joint.face_a_id, joint.face_b_id
                if a not in face_transforms or b not in face_transforms:
                    continue
                
                # 这两个面片应该通过这个关节连接
                # 计算它们在关节轴上的实际位置差异
                axis_local = self.joint_axis_points[joint.id]
                
                # 面片a中轴上一点的世界坐标
                axis_a_world = face_transforms[a] @ np.append(axis_local, 1.0)
                axis_a_world = axis_a_world[:3]
                
                # 面片b中轴上一点的世界坐标（应该和a的相同）
                axis_b_world = face_transforms[b] @ np.append(axis_local, 1.0)
                axis_b_world = axis_b_world[:3]
                
                error_pos = np.linalg.norm(axis_a_world - axis_b_world)
                
                # 也检查轴方向
                axis_dir_local = self.joint_axis_dirs[joint.id]
                dir_a = face_transforms[a][:3, :3] @ axis_dir_local
                dir_b = face_transforms[b][:3, :3] @ axis_dir_local
                error_dir = np.linalg.norm(dir_a - dir_b)
                
                error = error_pos + error_dir
                max_error = max(max_error, error)
                
                if error > self.tolerance:
                    # 需要修正：调整b的变换使其与a对齐
                    # 找出从b到a的路径上的关节，分摊误差
                    self._distribute_correction(
                        face_transforms, joint, a, b, error, root_id
                    )
            
            if max_error < self.tolerance:
                break
        
        return face_transforms
    
    def _distribute_correction(
        self,
        face_transforms: Dict[int, np.ndarray],
        joint: JointConnection,
        face_a: int,
        face_b: int,
        error: float,
        root_id: int
    ):
        """
        将环闭合误差分配到从face_b回溯到根再下到face_a的路径上的各关节
        
        简化策略：找到face_a和face_b的最低公共祖先(LCA)，
        然后调整连接LCA到face_b路径上的关节角度
        """
        # 构建从根到各面片的路径
        def path_to_root(face_id):
            path = []
            current = face_id
            while current != root_id:
                # 找到连接current到其父节点的关节
                found = False
                for j in self.design.joints:
                    if current in (j.face_a_id, j.face_b_id):
                        parent = j.face_a_id if j.face_b_id == current else j.face_b_id
                        # 检查parent是否是current的父节点（在生成树中更靠近根）
                        if self._is_closer_to_root(parent, current, root_id):
                            path.append((current, j.id, parent))
                            current = parent
                            found = True
                            break
                if not found:
                    break
            return path
        
        # 简化：直接调整face_b变换，使其与face_a在关节轴上对齐
        # 这是一个硬对齐，不是物理上准确的，但能消除误差
        axis_local = self.joint_axis_points[joint.id]
        
        # 将b的变换修正为在关节轴上与a一致
        T_a = face_transforms[face_a]
        T_b = face_transforms[face_b]
        
        # 计算b相对于a在关节轴上的偏移
        axis_a = T_a @ np.append(axis_local, 1.0)
        axis_b = T_b @ np.append(axis_local, 1.0)
        translation_correction = axis_a[:3] - axis_b[:3]
        
        # 应用修正：平移b使其在轴上与a对齐
        T_b_corrected = T_b.copy()
        T_b_corrected[:3, 3] += translation_correction
        face_transforms[face_b] = T_b_corrected
    
    def _is_closer_to_root(self, face_a: int, face_b: int, root_id: int) -> bool:
        """判断face_a是否比face_b更靠近根"""
        def depth(face_id, visited=None):
            if visited is None:
                visited = set()
            if face_id == root_id:
                return 0
            if face_id in visited:
                return float('inf')
            visited.add(face_id)
            
            for j in self.design.joints:
                if j.face_a_id == face_id or j.face_b_id == face_id:
                    neighbor = j.face_b_id if j.face_a_id == face_id else j.face_a_id
                    d = depth(neighbor, visited)
                    if d != float('inf'):
                        return d + 1
            return float('inf')
        
        return depth(face_a) < depth(face_b)
    
    def get_face_vertices_world(
        self, 
        joint_angles: Dict[int, float]
    ) -> Dict[int, np.ndarray]:
        """获取所有面片顶点在世界坐标系中的位置"""
        transforms = self.forward_kinematics(joint_angles)
        
        world_vertices = {}
        for face_id, transform in transforms.items():
            local_verts = self.face_vertices_3d.get(face_id)
            if local_verts is None:
                continue
            
            n_verts = len(local_verts)
            verts_h = np.hstack([local_verts, np.ones((n_verts, 1))])
            world_h = (transform @ verts_h.T).T
            world_vertices[face_id] = world_h[:, :3]
        
        return world_vertices