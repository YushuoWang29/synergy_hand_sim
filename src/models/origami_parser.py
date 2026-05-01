# src/models/origami_parser.py

import ezdxf
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from .origami_design import (
    Point2D, FoldLine, FoldType, OrigamiFace, 
    JointConnection, OrigamiHandDesign
)
from itertools import combinations


class OrigamiParser:
    """
    从DXF文件解析折纸手设计
    
    核心算法：
    1. 读取所有线段
    2. 在所有线段交点处分割 → 确保图为严格平面图
    3. 建立图，用"最小环路"算法找面片
    4. 根据面的包含关系确定root face
    5. 组装OrigamiHandDesign
    """
    
    COLOR_RED = 1      # 峰折
    COLOR_BLUE = 5     # 谷折
    COLOR_BLACK = 7    # 轮廓
    
    def __init__(self, point_tolerance: float = 1.0):
        """
        Args:
            point_tolerance: 两点/线距离小于此值视为重合 (CAD单位，一般是mm)
        """
        self.tolerance = point_tolerance
        self._next_line_id = 0
    
    def parse(self, dxf_path: str) -> OrigamiHandDesign:
        """主入口：解析DXF，返回完整设计"""
        print(f"Reading DXF: {dxf_path}")
        doc = ezdxf.readfile(dxf_path)
        modelspace = doc.modelspace()
        
        # 步骤1: 提取线段，并标注颜色
        raw_segments = self._extract_segments(modelspace)
        print(f"  Extracted {len(raw_segments)} segments")
        
        if not raw_segments:
            raise ValueError("No line segments found in DXF")
        
        # 步骤2: 在所有交点处分割线段
        split_segments = self._split_at_intersections(raw_segments)
        print(f"  Split into {len(split_segments)} non-intersecting segments")
        
        # 步骤3: 合并端点 → 建立图
        nodes, edges = self._build_graph(split_segments)
        print(f"  Graph: {len(nodes)} nodes, {len(edges)} edges")
        
        # 步骤4: 找面片（最小环路）
        faces = self._find_minimal_cycles(nodes, edges)
        print(f"  Found {len(faces)} faces")
        
        # 步骤5: 去掉最外面的面（整个图形的外轮廓）
        faces = self._remove_outer_face(faces, nodes)
        print(f"  After removing outer face: {len(faces)} faces")
        
        # 步骤6: 确定root face和关节连接
        design = self._create_design(nodes, edges, faces)
        print(f"  Created design with {len(design.faces)} faces, "
              f"{len(design.joints)} joints")
        
        return design
    
    # ================================================================
    # 步骤1: 提取线段
    # ================================================================
    
    def _extract_segments(self, modelspace) -> List[dict]:
        segments = []

        # src/models/origami_parser.py

# 在 _extract_segments 方法中，替换entity处理部分：

    def _extract_segments(self, modelspace) -> List[dict]:
        segments = []
        
        for entity in modelspace:
            start, end, color, fold_type = None, None, 7, FoldType.OUTLINE
            
            if entity.dxftype() == 'LINE':
                start = np.array([float(entity.dxf.start.x), 
                                float(entity.dxf.start.y)])
                end = np.array([float(entity.dxf.end.x), 
                              float(entity.dxf.end.y)])
                color = entity.dxf.color
                
            elif entity.dxftype() == 'LWPOLYLINE':
                # 处理多段线：展开为独立的线段
                points = list(entity.get_points())
                # 如果多段线只有2个点，当作普通线段处理
                # 如果多于2个点，分解为连续的线段
                color = entity.dxf.color
                
                for i in range(len(points) - 1):
                    start = np.array([float(points[i][0]), float(points[i][1])])
                    end = np.array([float(points[i+1][0]), float(points[i+1][1])])
                    
                    length = np.linalg.norm(end - start)
                    if length < self.tolerance:
                        continue
                    
                    fold_type = self._dxf_color_to_fold_type(color)
                    segments.append({
                        'start': start,
                        'end': end,
                        'fold_type': fold_type,
                        'length': length
                    })
                continue  # 已处理完多段线的所有段
            
            else:
                continue
            
            if start is None or end is None:
                continue
            
            length = np.linalg.norm(end - start)
            if length < self.tolerance:
                continue
            
            fold_type = self._dxf_color_to_fold_type(color)
            segments.append({
                'start': start,
                'end': end,
                'fold_type': fold_type,
                'length': length
            })
        
        return segments

    def _dxf_color_to_fold_type(self, color: int) -> FoldType:
        """DXF颜色索引转折痕类型（扩展版）"""
        # 红色系：峰折
        if color in (1, 10, 11, 12, 13):
            return FoldType.MOUNTAIN
        # 蓝色/青色系：谷折
        if color in (4, 5, 140, 141, 170, 171):
            return FoldType.VALLEY
        # 黑色/深色系：轮廓
        if color in (0, 7, 18, 250, 251, 252, 253, 254, 255):
            return FoldType.OUTLINE
        # 默认：轮廓
        return FoldType.OUTLINE
    
    # ================================================================
    # 步骤2: 在交点处分割线段
    # ================================================================
    
    def _split_at_intersections(self, segments: List[dict]) -> List[dict]:
        """
        核心步骤：将线段在所有交点处分割
        
        这确保最终图中每条边都是"原子"的，
        不存在中间穿过其他顶点的情况。
        """
        # 先收集所有线段的唯一点
        # 初始点集：所有线段端点
        all_points = []  # list of np.array(2,)
        for seg in segments:
            all_points.append(seg['start'])
            all_points.append(seg['end'])
        
        # 合并相近点
        unique_points = self._merge_points(all_points)
        
        # 对于每对相交线段，添加交点
        for i, seg_a in enumerate(segments):
            for j, seg_b in enumerate(segments):
                if i >= j:
                    continue
                
                intersection = self._line_intersection(
                    seg_a['start'], seg_a['end'],
                    seg_b['start'], seg_b['end']
                )
                
                if intersection is not None:
                    # 检查是否和已有重点重合
                    is_new = True
                    for pt in unique_points:
                        if np.linalg.norm(intersection - pt) < self.tolerance:
                            is_new = False
                            break
                    if is_new:
                        unique_points.append(intersection)
        
        # 现在用这些唯一点来分割所有线段
        split_segments = []
        
        for seg in segments:
            # 找出这条线段上的所有唯一点
            points_on_line = []
            for pt in unique_points:
                if self._point_on_segment(pt, seg['start'], seg['end']):
                    points_on_line.append(pt)
            
            # 按在线段上的位置排序
            line_dir = seg['end'] - seg['start']
            line_dir_norm = line_dir / np.linalg.norm(line_dir)
            
            points_on_line.sort(key=lambda p: np.dot(p - seg['start'], line_dir_norm))
            
            # 分割为子线段
            for k in range(len(points_on_line) - 1):
                sub_start = points_on_line[k]
                sub_end = points_on_line[k + 1]
                sub_length = np.linalg.norm(sub_end - sub_start)
                
                if sub_length > self.tolerance:
                    split_segments.append({
                        'start': sub_start,
                        'end': sub_end,
                        'fold_type': seg['fold_type'],
                        'length': sub_length
                    })
        
        return split_segments
    
    def _merge_points(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """合并距离小于tolerance的点"""
        if not points:
            return []
        
        # 简单贪婪聚类
        clusters = []
        for p in points:
            found = False
            for cluster in clusters:
                if np.linalg.norm(p - cluster[0]) < self.tolerance:
                    cluster.append(p)
                    found = True
                    break
            if not found:
                clusters.append([p])
        
        # 每个聚类用一个代表点
        return [cluster[0] for cluster in clusters]
    
    def _line_intersection(self, p1, p2, p3, p4) -> Optional[np.ndarray]:
        """计算两条线段的交点（如果在内部的话）"""
        # 用参数方程求解
        d1 = p2 - p1
        d2 = p4 - p3
        
        # 检查是否平行
        cross = np.cross(d1, d2)
        if abs(cross) < 1e-10:
            return None
        
        # 计算参数t
        t = np.cross(p3 - p1, d2) / cross
        u = np.cross(p3 - p1, d1) / cross
        
        # 交点必须在两条线段内部（允许端点的微小误差）
        eps = self.tolerance / max(np.linalg.norm(d1), np.linalg.norm(d2), 1.0)
        
        if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
            t = np.clip(t, 0, 1)
            return p1 + t * d1
        
        return None
    
    def _point_on_segment(self, point, seg_start, seg_end) -> bool:
        """判断点是否在线段上"""
        d = seg_end - seg_start
        length = np.linalg.norm(d)
        if length < self.tolerance:
            return np.linalg.norm(point - seg_start) < self.tolerance
        
        t = np.dot(point - seg_start, d) / (length * length)
        
        if t < -self.tolerance / length or t > 1 + self.tolerance / length:
            return False
        
        projection = seg_start + t * d
        return np.linalg.norm(point - projection) < self.tolerance
    
    # ================================================================
    # 步骤3: 建立图
    # ================================================================
    
    def _build_graph(self, segments: List[dict]) -> Tuple[List[np.ndarray], List[dict]]:
        """
        建立无向图
        
        Returns:
            nodes: 所有顶点坐标的列表
            edges: [{'u': node_idx, 'v': node_idx, 'fold_type': ...}]
        """
        # 收集所有端点
        all_points = []
        for seg in segments:
            all_points.append(seg['start'])
            all_points.append(seg['end'])
        
        # 合并去重
        unique_points = self._merge_points(all_points)
        
        # 建立坐标→索引的映射
        coord_to_idx = {}
        for i, pt in enumerate(unique_points):
            # 用坐标元组做key（取整到小数后3位避免浮点误差）
            key = (round(pt[0], 3), round(pt[1], 3))
            coord_to_idx[key] = i
        
        nodes = unique_points
        
        # 建立边
        edges = []
        seen_edges = set()
        
        for seg in segments:
            key_start = (round(seg['start'][0], 3), round(seg['start'][1], 3))
            key_end = (round(seg['end'][0], 3), round(seg['end'][1], 3))
            
            u = coord_to_idx[key_start]
            v = coord_to_idx[key_end]
            
            if u == v:
                continue
            
            # 去重（同一条边可能来自多个共线线段）
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                # 如果已存在但之前是轮廓，新的是折痕，升级
                # （简化处理：如果有折痕，整条边算折痕）
                for existing_edge in edges:
                    if (min(existing_edge['u'], existing_edge['v']), 
                        max(existing_edge['u'], existing_edge['v'])) == edge_key:
                        if existing_edge['fold_type'] == FoldType.OUTLINE:
                            existing_edge['fold_type'] = seg['fold_type']
                        break
                continue
            
            seen_edges.add(edge_key)
            edges.append({
                'u': u,
                'v': v,
                'fold_type': seg['fold_type'],
                'id': self._next_line_id
            })
            self._next_line_id += 1
        
        return nodes, edges
    
    # ================================================================
    # 步骤4: 找最小面片
    # ================================================================
    
    def _find_minimal_cycles(self, nodes, edges) -> List[dict]:
        """
        使用平面图的面遍历算法找所有最小面
        
        算法：对每条有向边(u→v)，在v处找"左转最急"的下一条边，
        直到回到起点，形成一个面。
        """
        # 为每个节点建立排序邻接表（按极角）
        adj = {i: [] for i in range(len(nodes))}
        
        for edge in edges:
            u, v = edge['u'], edge['v']
            adj[u].append((v, edge['id'], edge['fold_type']))
            adj[v].append((u, edge['id'], edge['fold_type']))
        
        # 对每个节点的邻接边按角度排序
        for node_id, neighbors in adj.items():
            p0 = nodes[node_id]
            neighbors.sort(key=lambda x: np.arctan2(
                nodes[x[0]][1] - p0[1],
                nodes[x[0]][0] - p0[0]
            ))
        
        # 每条有向边标记为未使用
        used = set()  # {(u, v)}
        faces = []
        
        for u in adj:
            for (v, edge_id, fold_type) in adj[u]:
                if (u, v) in used:
                    continue
                
                # 追踪面
                path_nodes = [u]
                path_edges = [edge_id]
                
                current = u
                nxt = v
                
                too_long = False
                
                while True:
                    used.add((current, nxt))
                    
                    if nxt == u:
                        # 回到起点，找到面
                        if len(path_nodes) >= 3:
                            faces.append({
                                'node_indices': path_nodes[:],
                                'edge_ids': path_edges[:]
                            })
                        break
                    
                    if nxt in path_nodes:
                        # 非起点重复
                        break
                    
                    if len(path_nodes) > len(nodes) + 1:
                        too_long = True
                        break
                    
                    path_nodes.append(nxt)
                    
                    # 在nxt处找下一条边
                    incoming = current
                    nbr_list = adj[nxt]
                    
                    # 找incoming在邻接表中的位置
                    incoming_idx = None
                    for idx, (neighbor, _, _) in enumerate(nbr_list):
                        if neighbor == incoming:
                            incoming_idx = idx
                            break
                    
                    if incoming_idx is None:
                        break
                    
                    # "左转" = incoming_idx + 1 (循环)
                    out_idx = (incoming_idx + 1) % len(nbr_list)
                    next_node, next_edge_id, _ = nbr_list[out_idx]
                    
                    if (nxt, next_node) in used:
                        break
                    
                    path_edges.append(next_edge_id)
                    current = nxt
                    nxt = next_node
                
                if too_long:
                    continue
        
        # 去重
        unique_faces = []
        seen = set()
        
        for face in faces:
            nodes_tuple = tuple(face['node_indices'])
            # 规范化
            min_idx = nodes_tuple.index(min(nodes_tuple))
            rotated = nodes_tuple[min_idx:] + nodes_tuple[:min_idx]
            reversed_rotated = tuple(reversed(rotated))
            key = min(rotated, reversed_rotated)
            
            if key not in seen:
                seen.add(key)
                unique_faces.append(face)
        
        return unique_faces
    
    # ================================================================
    # 步骤5: 去掉外轮廓面
    # ================================================================
    
    def _remove_outer_face(self, faces: List[dict], nodes: List[np.ndarray]) -> List[dict]:
        """
        去掉外轮廓面（整个图形的边界）
        
        外轮廓面 = 所有面的并集（面积最大的面）
        面面积用多边形面积公式计算
        """
        if len(faces) <= 1:
            return faces
        
        def face_area(face):
            indices = face['node_indices']
            pts = [nodes[i] for i in indices]
            area = 0.0
            n = len(pts)
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]
                area += x1 * y2 - x2 * y1
            return abs(area) / 2
        
        # 找面积最大的面
        areas = [face_area(f) for f in faces]
        max_area_idx = np.argmax(areas)
        
        # 检查最大面是否明显大于其他面
        # 外轮廓面通常远大于内部面
        areas_sorted = sorted(areas, reverse=True)
        if len(areas_sorted) >= 2:
            ratio = areas_sorted[0] / areas_sorted[1] if areas_sorted[1] > 0 else float('inf')
            if ratio > 1.5:
                # 去掉最大的面（外轮廓）
                inner_faces = [f for i, f in enumerate(faces) if i != max_area_idx]
                return inner_faces
        
        return faces
    
    # ================================================================
    # 步骤6: 创建OrigamiHandDesign
    # ================================================================
    
    # 替换 origami_parser.py 中的整个 _create_design 方法

    def _create_design(
        self,
        nodes: List[np.ndarray],
        edges: List[dict],
        faces: List[dict]
    ) -> OrigamiHandDesign:
        
        design = OrigamiHandDesign(name="imported_hand", material_thickness=0.002)
        
        # 建字典：node_id -> Point2D
        node_to_point = {}
        for i, pt in enumerate(nodes):
            node_to_point[i] = Point2D(float(pt[0]), float(pt[1]))
        
        # ===== 过滤碎片面 =====
        def face_area(face):
            pts = [nodes[i] for i in face['node_indices']]
            n = len(pts)
            if n < 3:
                return 0.0
            area = 0.0
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]
                area += x1 * y2 - x2 * y1
            return abs(area) / 2
        
        all_areas = [face_area(f) for f in faces]
        if all_areas:
            max_area = max(all_areas)
            area_threshold = max_area * 0.01
        else:
            area_threshold = 0
        
        valid_faces = []
        removed_count = 0
        for f in faces:
            a = face_area(f)
            if a >= area_threshold:
                valid_faces.append(f)
            else:
                removed_count += 1
        
        if removed_count > 0:
            print(f"  Filtered out {removed_count} tiny faces (area < {area_threshold:.1f})")
        
        faces = valid_faces
        
        # ===== 添加折痕线 =====
        for edge in edges:
            u, v = edge['u'], edge['v']
            start_pt = node_to_point[u]
            end_pt = node_to_point[v]
            
            line = FoldLine(
                id=edge['id'],
                start=start_pt,
                end=end_pt,
                fold_type=edge['fold_type']
            )
            design.add_fold_line(line)
        
        # ===== 添加面片 =====
        for i, face in enumerate(faces):
            vertices = []
            for node_idx in face['node_indices']:
                vertices.append(node_to_point[node_idx])
            
            edge_ids = face['edge_ids']
            
            origami_face = OrigamiFace(
                id=i,
                vertices=vertices,
                edge_ids=edge_ids
            )
            design.add_face(origami_face)
        
        # ===== 建立边→面映射 =====
        edge_to_faces = {}
        for face_id in range(len(faces)):
            for edge_id in faces[face_id]['edge_ids']:
                if edge_id not in edge_to_faces:
                    edge_to_faces[edge_id] = []
                edge_to_faces[edge_id].append(face_id)
        
        # ===== 创建关节：为所有共享折痕的面片对创建关节 =====
        joint_id = 0
        
        for edge_id, face_list in edge_to_faces.items():
            # 找到这条边对应的信息
            edge_info = None
            for e in edges:
                if e['id'] == edge_id:
                    edge_info = e
                    break
            
            if edge_info is None:
                continue
            
            # 只处理折痕（峰/谷），轮廓不产生关节
            if edge_info['fold_type'] == FoldType.OUTLINE:
                continue
            
            if len(face_list) == 2:
                a, b = face_list[0], face_list[1]
                joint = JointConnection(
                    id=joint_id,
                    fold_line_id=edge_id,
                    face_a_id=a,
                    face_b_id=b,
                    fold_type=edge_info['fold_type']
                )
                design.add_joint(joint)
                joint_id += 1
                
            elif len(face_list) > 2:
                # 超过2个面片共享同一条折痕，为每对面片创建关节
                for i in range(len(face_list)):
                    for j in range(i + 1, len(face_list)):
                        joint = JointConnection(
                            id=joint_id,
                            fold_line_id=edge_id,
                            face_a_id=face_list[i],
                            face_b_id=face_list[j],
                            fold_type=edge_info['fold_type']
                        )
                        design.add_joint(joint)
                        joint_id += 1
        
        print(f"  Created {joint_id} joints ({len(edges)} total edges)")
        
        # ===== 找root face：面积最大的面 =====
        if faces:
            def get_area(face_id):
                f = design.faces[face_id]
                return f.area
            
            root_id = max(design.faces.keys(), key=get_area)
            design.set_root_face(root_id)
            design.build_face_tree()
        
        # ===== 诊断连通性 =====
        if len(design.face_parent) < len(design.faces) - 1:
            unconnected = set(design.faces.keys()) - set(design.face_parent.keys())
            if design.root_face_id in unconnected:
                unconnected.remove(design.root_face_id)
            if unconnected:
                print(f"  Warning: {len(unconnected)} faces not connected to root: {sorted(unconnected)}")
        
        return design