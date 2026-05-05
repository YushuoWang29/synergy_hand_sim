# src/models/origami_parser.py

import ezdxf
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from .origami_design import (
    Point2D, FoldLine, FoldType, OrigamiFace, 
    JointConnection, OrigamiHandDesign
)
from itertools import combinations


# =============================================================================
# 独立工具函数：可在不创建 OrigamiParser 实例的情况下使用
# =============================================================================

def split_at_intersections(segments: List[dict], tolerance: float = 1.0) -> List[dict]:
    """
    独立版：在所有线段交点处分割。
    tolerance: 点合并容差
    """
    all_points = []
    for seg in segments:
        all_points.append(seg['start'])
        all_points.append(seg['end'])
    
    unique_points = _merge_points_core(all_points, tolerance)
    
    for i, seg_a in enumerate(segments):
        for j, seg_b in enumerate(segments):
            if i >= j:
                continue
            
            intersection = _line_intersection_core(
                seg_a['start'], seg_a['end'],
                seg_b['start'], seg_b['end'],
                tolerance
            )
            
            if intersection is not None:
                is_new = True
                for pt in unique_points:
                    if np.linalg.norm(intersection - pt) < tolerance:
                        is_new = False
                        break
                if is_new:
                    unique_points.append(intersection)
    
    split_segments = []
    
    for seg in segments:
        points_on_line = []
        for pt in unique_points:
            if _point_on_segment_core(pt, seg['start'], seg['end'], tolerance):
                points_on_line.append(pt)
        
        line_dir = seg['end'] - seg['start']
        line_dir_norm = line_dir / np.linalg.norm(line_dir)
        
        points_on_line.sort(key=lambda p: np.dot(p - seg['start'], line_dir_norm))
        
        for k in range(len(points_on_line) - 1):
            sub_start = points_on_line[k]
            sub_end = points_on_line[k + 1]
            sub_length = np.linalg.norm(sub_end - sub_start)
            
            if sub_length > tolerance:
                split_segments.append({
                    'start': sub_start,
                    'end': sub_end,
                    'fold_type': seg['fold_type'],
                    'length': sub_length
                })
    
    return split_segments


def _merge_points_core(points: List[np.ndarray], tolerance: float) -> List[np.ndarray]:
    """合并距离小于tolerance的点"""
    if not points:
        return []
    
    clusters = []
    for p in points:
        found = False
        for cluster in clusters:
            if np.linalg.norm(p - cluster[0]) < tolerance:
                cluster.append(p)
                found = True
                break
        if not found:
            clusters.append([p])
    
    return [cluster[0] for cluster in clusters]


def _line_intersection_core(p1, p2, p3, p4, tolerance: float) -> Optional[np.ndarray]:
    """计算两条线段的交点（如果在内部的话）"""
    d1 = p2 - p1
    d2 = p4 - p3
    
    cross = np.cross(d1, d2)
    if abs(cross) < 1e-10:
        return None
    
    t = np.cross(p3 - p1, d2) / cross
    u = np.cross(p3 - p1, d1) / cross
    
    eps = tolerance / max(np.linalg.norm(d1), np.linalg.norm(d2), 1.0)
    
    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        t = np.clip(t, 0, 1)
        return p1 + t * d1
    
    return None


def _point_on_segment_core(point, seg_start, seg_end, tolerance: float) -> bool:
    """判断点是否在线段上"""
    d = seg_end - seg_start
    length = np.linalg.norm(d)
    if length < tolerance:
        return np.linalg.norm(point - seg_start) < tolerance
    
    t = np.dot(point - seg_start, d) / (length * length)
    
    if t < -tolerance / length or t > 1 + tolerance / length:
        return False
    
    projection = seg_start + t * d
    return np.linalg.norm(point - projection) < tolerance


def build_graph_from_segments(segments: List[dict]) -> Tuple[List[np.ndarray], List[dict]]:
    """
    独立版：建立无向图。
    
    Returns:
        nodes: 所有顶点坐标的列表
        edges: [{'u': node_idx, 'v': node_idx, 'fold_type': ..., 'id': ...}]
    """
    all_points = []
    for seg in segments:
        all_points.append(seg['start'])
        all_points.append(seg['end'])
    
    unique_points = _merge_points_core(all_points, 1.0)
    
    coord_to_idx = {}
    for i, pt in enumerate(unique_points):
        key = (round(pt[0], 3), round(pt[1], 3))
        coord_to_idx[key] = i
    
    nodes = unique_points
    
    edges = []
    seen_edges = set()
    next_line_id = 0
    
    for seg in segments:
        key_start = (round(seg['start'][0], 3), round(seg['start'][1], 3))
        key_end = (round(seg['end'][0], 3), round(seg['end'][1], 3))
        
        u = coord_to_idx[key_start]
        v = coord_to_idx[key_end]
        
        if u == v:
            continue
        
        edge_key = (min(u, v), max(u, v))
        if edge_key in seen_edges:
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
            'id': next_line_id
        })
        next_line_id += 1
    
    return nodes, edges


def find_minimal_cycles(nodes: List[np.ndarray], edges: List[dict]) -> List[dict]:
    """
    独立版：使用平面图的面遍历算法找所有最小面。
    """
    adj = {i: [] for i in range(len(nodes))}
    
    for edge in edges:
        u, v = edge['u'], edge['v']
        adj[u].append((v, edge['id'], edge['fold_type']))
        adj[v].append((u, edge['id'], edge['fold_type']))
    
    for node_id, neighbors in adj.items():
        p0 = nodes[node_id]
        neighbors.sort(key=lambda x: np.arctan2(
            nodes[x[0]][1] - p0[1],
            nodes[x[0]][0] - p0[0]
        ))
    
    used = set()
    faces = []
    
    for u in adj:
        for (v, edge_id, fold_type) in adj[u]:
            if (u, v) in used:
                continue
            
            path_nodes = [u]
            path_edges = [edge_id]
            
            current = u
            nxt = v
            
            too_long = False
            
            while True:
                used.add((current, nxt))
                
                if nxt == u:
                    if len(path_nodes) >= 3:
                        faces.append({
                            'node_indices': path_nodes[:],
                            'edge_ids': path_edges[:]
                        })
                    break
                
                if nxt in path_nodes:
                    break
                
                if len(path_nodes) > len(nodes) + 1:
                    too_long = True
                    break
                
                path_nodes.append(nxt)
                
                incoming = current
                nbr_list = adj[nxt]
                
                incoming_idx = None
                for idx, (neighbor, _, _) in enumerate(nbr_list):
                    if neighbor == incoming:
                        incoming_idx = idx
                        break
                
                if incoming_idx is None:
                    break
                
                out_idx = (incoming_idx + 1) % len(nbr_list)
                next_node, next_edge_id, _ = nbr_list[out_idx]
                
                if (nxt, next_node) in used:
                    break
                
                path_edges.append(next_edge_id)
                current = nxt
                nxt = next_node
            
            if too_long:
                continue
    
    unique_faces = []
    seen = set()
    
    for face in faces:
        nodes_tuple = tuple(face['node_indices'])
        min_idx = nodes_tuple.index(min(nodes_tuple))
        rotated = nodes_tuple[min_idx:] + nodes_tuple[:min_idx]
        reversed_rotated = tuple(reversed(rotated))
        key = min(rotated, reversed_rotated)
        
        if key not in seen:
            seen.add(key)
            unique_faces.append(face)
    
    return unique_faces


def remove_outer_face(faces: List[dict], nodes: List[np.ndarray]) -> List[dict]:
    """
    独立版：去掉外轮廓面（面积最大的面）。
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
    
    areas = [face_area(f) for f in faces]
    max_area_idx = np.argmax(areas)
    
    areas_sorted = sorted(areas, reverse=True)
    if len(areas_sorted) >= 2:
        ratio = areas_sorted[0] / areas_sorted[1] if areas_sorted[1] > 0 else float('inf')
        if ratio > 1.5:
            inner_faces = [f for i, f in enumerate(faces) if i != max_area_idx]
            return inner_faces
    
    return faces


def rebuild_design_topology(design: OrigamiHandDesign, nodes, edges, faces):
    """
    向已有的 OrigamiHandDesign 中添加面片和关节（不添加折痕线）。
    用于 build_topology() 中从已有的 fold_lines 重建拓扑。
    """
    node_to_point = {}
    for i, pt in enumerate(nodes):
        node_to_point[i] = Point2D(float(pt[0]), float(pt[1]))
    
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
    
    for i, face in enumerate(faces):
        vertices = []
        for node_idx in face['node_indices']:
            vertices.append(node_to_point[node_idx])
        edge_ids = face['edge_ids']
        origami_face = OrigamiFace(id=i, vertices=vertices, edge_ids=edge_ids)
        design.add_face(origami_face)
    
    edge_to_faces = {}
    for face_id in range(len(faces)):
        for edge_id in faces[face_id]['edge_ids']:
            if edge_id not in edge_to_faces:
                edge_to_faces[edge_id] = []
            edge_to_faces[edge_id].append(face_id)
    
    joint_id = 0
    for edge_id, face_list in edge_to_faces.items():
        edge_info = None
        for e in edges:
            if e['id'] == edge_id:
                edge_info = e
                break
        
        if edge_info is None:
            continue
        
        if edge_info['fold_type'] == FoldType.OUTLINE:
            continue
        
        if len(face_list) == 2:
            a, b = face_list[0], face_list[1]
            joint = JointConnection(id=joint_id, fold_line_id=edge_id,
                                    face_a_id=a, face_b_id=b,
                                    fold_type=edge_info['fold_type'])
            design.add_joint(joint)
            joint_id += 1
        elif len(face_list) > 2:
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    joint = JointConnection(id=joint_id, fold_line_id=edge_id,
                                            face_a_id=face_list[i], face_b_id=face_list[j],
                                            fold_type=edge_info['fold_type'])
                    design.add_joint(joint)
                    joint_id += 1
    
    print(f"  Rebuilt topology: {len(faces)} faces, {joint_id} joints")
    
    if faces:
        def get_area(face_id):
            return design.faces[face_id].area
        
        root_id = max(design.faces.keys(), key=get_area)
        design.set_root_face(root_id)
        design.build_face_tree()
    
    if len(design.face_parent) < len(design.faces) - 1:
        unconnected = set(design.faces.keys()) - set(design.face_parent.keys())
        if design.root_face_id in unconnected:
            unconnected.remove(design.root_face_id)
        if unconnected:
            print(f"  Warning: {len(unconnected)} faces not connected to root: {sorted(unconnected)}")


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
        self.tolerance = point_tolerance
        self._next_line_id = 0
    
    def parse(self, dxf_path: str) -> OrigamiHandDesign:
        """主入口：解析DXF，返回完整设计"""
        print(f"Reading DXF: {dxf_path}")
        doc = ezdxf.readfile(dxf_path)
        modelspace = doc.modelspace()
        
        raw_segments = self._extract_segments(modelspace)
        print(f"  Extracted {len(raw_segments)} segments")
        
        if not raw_segments:
            raise ValueError("No line segments found in DXF")
        
        split_segments = self._split_at_intersections(raw_segments)
        print(f"  Split into {len(split_segments)} non-intersecting segments")
        
        nodes, edges = self._build_graph(split_segments)
        print(f"  Graph: {len(nodes)} nodes, {len(edges)} edges")
        
        faces = self._find_minimal_cycles(nodes, edges)
        print(f"  Found {len(faces)} faces")
        
        faces = self._remove_outer_face(faces, nodes)
        print(f"  After removing outer face: {len(faces)} faces")
        
        design = self._create_design(nodes, edges, faces)
        print(f"  Created design with {len(design.faces)} faces, "
              f"{len(design.joints)} joints")
        
        return design
    
    # ================================================================
    # 步骤1: 提取线段
    # ================================================================
    
    def _extract_segments(self, modelspace) -> List[dict]:
        segments = []
        
        for entity in modelspace:
            start, end, color = None, None, 7
            
            if entity.dxftype() == 'LINE':
                start = np.array([float(entity.dxf.start.x), 
                                float(entity.dxf.start.y)])
                end = np.array([float(entity.dxf.end.x), 
                              float(entity.dxf.end.y)])
                color = entity.dxf.color
                
            elif entity.dxftype() == 'LWPOLYLINE':
                points = list(entity.get_points())
                color = entity.dxf.color
                
                for i in range(len(points) - 1):
                    start = np.array([float(points[i][0]), float(points[i][1])])
                    end = np.array([float(points[i+1][0]), float(points[i+1][1])])
                    
                    length = np.linalg.norm(end - start)
                    if length < self.tolerance:
                        continue
                    
                    fold_type = self._dxf_color_to_fold_type(color)
                    segments.append({
                        'start': start, 'end': end,
                        'fold_type': fold_type, 'length': length
                    })
                continue
            
            else:
                continue
            
            if start is None or end is None:
                continue
            
            length = np.linalg.norm(end - start)
            if length < self.tolerance:
                continue
            
            fold_type = self._dxf_color_to_fold_type(color)
            segments.append({
                'start': start, 'end': end,
                'fold_type': fold_type, 'length': length
            })
        
        return segments

    def _dxf_color_to_fold_type(self, color: int) -> FoldType:
        """DXF颜色索引转折痕类型（扩展版）"""
        if color in (1, 10, 11, 12, 13):
            return FoldType.MOUNTAIN
        if color in (4, 5, 140, 141, 170, 171):
            return FoldType.VALLEY
        if color in (0, 7, 18, 250, 251, 252, 253, 254, 255):
            return FoldType.OUTLINE
        return FoldType.OUTLINE
    
    # ================================================================
    # 步骤2: 在交点处分割线段
    # ================================================================
    
    def _split_at_intersections(self, segments: List[dict]) -> List[dict]:
        all_points = []
        for seg in segments:
            all_points.append(seg['start'])
            all_points.append(seg['end'])
        
        unique_points = self._merge_points(all_points)
        
        for i, seg_a in enumerate(segments):
            for j, seg_b in enumerate(segments):
                if i >= j:
                    continue
                
                intersection = self._line_intersection(
                    seg_a['start'], seg_a['end'],
                    seg_b['start'], seg_b['end']
                )
                
                if intersection is not None:
                    is_new = True
                    for pt in unique_points:
                        if np.linalg.norm(intersection - pt) < self.tolerance:
                            is_new = False
                            break
                    if is_new:
                        unique_points.append(intersection)
        
        split_segments = []
        
        for seg in segments:
            points_on_line = []
            for pt in unique_points:
                if self._point_on_segment(pt, seg['start'], seg['end']):
                    points_on_line.append(pt)
            
            line_dir = seg['end'] - seg['start']
            line_dir_norm = line_dir / np.linalg.norm(line_dir)
            
            points_on_line.sort(key=lambda p: np.dot(p - seg['start'], line_dir_norm))
            
            for k in range(len(points_on_line) - 1):
                sub_start = points_on_line[k]
                sub_end = points_on_line[k + 1]
                sub_length = np.linalg.norm(sub_end - sub_start)
                
                if sub_length > self.tolerance:
                    split_segments.append({
                        'start': sub_start, 'end': sub_end,
                        'fold_type': seg['fold_type'], 'length': sub_length
                    })
        
        return split_segments
    
    def _merge_points(self, points: List[np.ndarray]) -> List[np.ndarray]:
        if not points:
            return []
        
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
        
        return [cluster[0] for cluster in clusters]
    
    def _line_intersection(self, p1, p2, p3, p4) -> Optional[np.ndarray]:
        d1 = p2 - p1
        d2 = p4 - p3
        
        cross = np.cross(d1, d2)
        if abs(cross) < 1e-10:
            return None
        
        t = np.cross(p3 - p1, d2) / cross
        u = np.cross(p3 - p1, d1) / cross
        
        eps = self.tolerance / max(np.linalg.norm(d1), np.linalg.norm(d2), 1.0)
        
        if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
            t = np.clip(t, 0, 1)
            return p1 + t * d1
        
        return None
    
    def _point_on_segment(self, point, seg_start, seg_end) -> bool:
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
        all_points = []
        for seg in segments:
            all_points.append(seg['start'])
            all_points.append(seg['end'])
        
        unique_points = self._merge_points(all_points)
        
        coord_to_idx = {}
        for i, pt in enumerate(unique_points):
            key = (round(pt[0], 3), round(pt[1], 3))
            coord_to_idx[key] = i
        
        nodes = unique_points
        
        edges = []
        seen_edges = set()
        
        for seg in segments:
            key_start = (round(seg['start'][0], 3), round(seg['start'][1], 3))
            key_end = (round(seg['end'][0], 3), round(seg['end'][1], 3))
            
            u = coord_to_idx[key_start]
            v = coord_to_idx[key_end]
            
            if u == v:
                continue
            
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                for existing_edge in edges:
                    if (min(existing_edge['u'], existing_edge['v']), 
                        max(existing_edge['u'], existing_edge['v'])) == edge_key:
                        if existing_edge['fold_type'] == FoldType.OUTLINE:
                            existing_edge['fold_type'] = seg['fold_type']
                        break
                continue
            
            seen_edges.add(edge_key)
            edges.append({
                'u': u, 'v': v,
                'fold_type': seg['fold_type'],
                'id': self._next_line_id
            })
            self._next_line_id += 1
        
        return nodes, edges
    
    # ================================================================
    # 步骤4: 找最小面片
    # ================================================================
    
    def _find_minimal_cycles(self, nodes, edges) -> List[dict]:
        adj = {i: [] for i in range(len(nodes))}
        
        for edge in edges:
            u, v = edge['u'], edge['v']
            adj[u].append((v, edge['id'], edge['fold_type']))
            adj[v].append((u, edge['id'], edge['fold_type']))
        
        for node_id, neighbors in adj.items():
            p0 = nodes[node_id]
            neighbors.sort(key=lambda x: np.arctan2(
                nodes[x[0]][1] - p0[1],
                nodes[x[0]][0] - p0[0]
            ))
        
        used = set()
        faces = []
        
        for u in adj:
            for (v, edge_id, fold_type) in adj[u]:
                if (u, v) in used:
                    continue
                
                path_nodes = [u]
                path_edges = [edge_id]
                
                current = u
                nxt = v
                
                too_long = False
                
                while True:
                    used.add((current, nxt))
                    
                    if nxt == u:
                        if len(path_nodes) >= 3:
                            faces.append({
                                'node_indices': path_nodes[:],
                                'edge_ids': path_edges[:]
                            })
                        break
                    
                    if nxt in path_nodes:
                        break
                    
                    if len(path_nodes) > len(nodes) + 1:
                        too_long = True
                        break
                    
                    path_nodes.append(nxt)
                    
                    incoming = current
                    nbr_list = adj[nxt]
                    
                    incoming_idx = None
                    for idx, (neighbor, _, _) in enumerate(nbr_list):
                        if neighbor == incoming:
                            incoming_idx = idx
                            break
                    
                    if incoming_idx is None:
                        break
                    
                    out_idx = (incoming_idx + 1) % len(nbr_list)
                    next_node, next_edge_id, _ = nbr_list[out_idx]
                    
                    if (nxt, next_node) in used:
                        break
                    
                    path_edges.append(next_edge_id)
                    current = nxt
                    nxt = next_node
                
                if too_long:
                    continue
        
        unique_faces = []
        seen = set()
        
        for face in faces:
            nodes_tuple = tuple(face['node_indices'])
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
        
        areas = [face_area(f) for f in faces]
        max_area_idx = np.argmax(areas)
        
        areas_sorted = sorted(areas, reverse=True)
        if len(areas_sorted) >= 2:
            ratio = areas_sorted[0] / areas_sorted[1] if areas_sorted[1] > 0 else float('inf')
            if ratio > 1.5:
                inner_faces = [f for i, f in enumerate(faces) if i != max_area_idx]
                return inner_faces
        
        return faces
    
    # ================================================================
    # 步骤6: 创建OrigamiHandDesign
    # ================================================================
    
    def _create_design(self, nodes, edges, faces) -> OrigamiHandDesign:
        design = OrigamiHandDesign(name="imported_hand")
        
        node_to_point = {}
        for i, pt in enumerate(nodes):
            node_to_point[i] = Point2D(float(pt[0]), float(pt[1]))
        
        # 过滤碎片面
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
        
        # 添加折痕线
        for edge in edges:
            u, v = edge['u'], edge['v']
            line = FoldLine(id=edge['id'],
                            start=node_to_point[u], end=node_to_point[v],
                            fold_type=edge['fold_type'])
            design.add_fold_line(line)
        
        # 添加面片
        for i, face in enumerate(faces):
            vertices = [node_to_point[idx] for idx in face['node_indices']]
            origami_face = OrigamiFace(id=i, vertices=vertices, edge_ids=face['edge_ids'])
            design.add_face(origami_face)
        
        # 边→面映射
        edge_to_faces = {}
        for face_id in range(len(faces)):
            for edge_id in faces[face_id]['edge_ids']:
                if edge_id not in edge_to_faces:
                    edge_to_faces[edge_id] = []
                edge_to_faces[edge_id].append(face_id)
        
        # 创建关节
        joint_id = 0
        for edge_id, face_list in edge_to_faces.items():
            edge_info = None
            for e in edges:
                if e['id'] == edge_id:
                    edge_info = e
                    break
            
            if edge_info is None or edge_info['fold_type'] == FoldType.OUTLINE:
                continue
            
            if len(face_list) == 2:
                a, b = face_list[0], face_list[1]
                design.add_joint(JointConnection(id=joint_id, fold_line_id=edge_id,
                                                  face_a_id=a, face_b_id=b,
                                                  fold_type=edge_info['fold_type']))
                joint_id += 1
            elif len(face_list) > 2:
                for i in range(len(face_list)):
                    for j in range(i + 1, len(face_list)):
                        design.add_joint(JointConnection(id=joint_id, fold_line_id=edge_id,
                                                          face_a_id=face_list[i],
                                                          face_b_id=face_list[j],
                                                          fold_type=edge_info['fold_type']))
                        joint_id += 1
        
        print(f"  Created {joint_id} joints ({len(edges)} total edges)")
        
        # root face
        if faces:
            root_id = max(design.faces.keys(), key=lambda fid: design.faces[fid].area)
            design.set_root_face(root_id)
            design.build_face_tree()
        
        # 连通性诊断
        if len(design.face_parent) < len(design.faces) - 1:
            unconnected = set(design.faces.keys()) - set(design.face_parent.keys())
            if design.root_face_id in unconnected:
                unconnected.remove(design.root_face_id)
            if unconnected:
                print(f"  Warning: {len(unconnected)} faces not connected to root: {sorted(unconnected)}")
        
        return design
