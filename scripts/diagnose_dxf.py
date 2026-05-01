# scripts/diagnose_dxf.py
"""
诊断DXF文件的面片识别和连接情况
"""

import sys
sys.path.insert(0, '.')

import argparse
from src.models.origami_parser import OrigamiParser


def diagnose(dxf_path, tolerance=1.0):
    print("=" * 60)
    print(f"Diagnosing: {dxf_path}")
    print("=" * 60)
    
    parser = OrigamiParser(point_tolerance=tolerance)
    
    # 手动执行解析步骤
    import ezdxf
    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()
    
    raw_segments = parser._extract_segments(modelspace)
    print(f"\n1. Raw segments: {len(raw_segments)}")
    for i, seg in enumerate(raw_segments):
        print(f"   {i}: ({seg['start'][0]:.0f},{seg['start'][1]:.0f}) -> "
              f"({seg['end'][0]:.0f},{seg['end'][1]:.0f}) "
              f"[{seg['fold_type'].value}]")
    
    split_segments = parser._split_at_intersections(raw_segments)
    print(f"\n2. After splitting: {len(split_segments)} segments")
    
    nodes, edges = parser._build_graph(split_segments)
    print(f"\n3. Graph: {len(nodes)} nodes, {len(edges)} edges")
    
    # 对每条边标注类型
    print("\n   Edges:")
    for e in edges:
        u, v = e['u'], e['v']
        n1, n2 = nodes[u], nodes[v]
        print(f"   E{e['id']}: ({n1[0]:.0f},{n1[1]:.0f}) -> "
              f"({n2[0]:.0f},{n2[1]:.0f}) [{e['fold_type'].value}]")
    
    faces = parser._find_minimal_cycles(nodes, edges)
    print(f"\n4. Faces found: {len(faces)}")
    for i, face in enumerate(faces):
        indices = face['node_indices']
        pts = [f"({nodes[j][0]:.0f},{nodes[j][1]:.0f})" for j in indices]
        eids = face['edge_ids']
        print(f"   Face {i}: vertices={pts}")
        print(f"          edges={eids}")
    
    # 找面片之间的共享边
    print("\n5. Edge-to-face mapping:")
    edge_to_faces = {}
    for fid, face in enumerate(faces):
        for eid in face['edge_ids']:
            if eid not in edge_to_faces:
                edge_to_faces[eid] = []
            edge_to_faces[eid].append(fid)
    
    for eid, flist in edge_to_faces.items():
        print(f"   Edge {eid}: shared by faces {flist}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf_file")
    parser.add_argument("--tolerance", type=float, default=1.0)
    args = parser.parse_args()
    
    diagnose(args.dxf_file, args.tolerance)