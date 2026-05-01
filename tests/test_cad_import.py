# tests/test_cad_import.py
"""
端到端测试：
1. 用代码生成一个DXF文件（模拟CAD绘图）
2. 用OrigamiParser解析
3. 可视化结果
"""

import sys
sys.path.insert(0, '.')

import ezdxf
import numpy as np
import os
from src.models.origami_parser import OrigamiParser
from src.models.origami_kinematics import OrigamiForwardKinematics
from src.visualization.origami_visualizer import OrigamiVisualizer


def create_test_dxf(output_path: str):
    """
    创建一个测试DXF文件
    
    画一个简单的手掌带一根手指：
    - 手掌：矩形
    - 手指近端：矩形，通过谷折连接手掌
    - 手指远端：矩形，通过谷折连接近端
    
    颜色编码：
    7 = 黑/白（轮廓）
    5 = 蓝（谷折，关节在板上表面）
    1 = 红（峰折，关节在板下表面）
    """
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # === 手掌 (50x30的矩形) ===
    # 左下(0,0), 右下(50,0), 右上(50,30), 左上(0,30)
    palm = [
        (0, 0),
        (50, 0),
        (50, 30),
        (0, 30),
    ]
    
    # === 手指近端 (20x25的矩形，在手掌上面) ===
    finger_proximal = [
        (15, 30),   # 左下
        (35, 30),   # 右下
        (35, 55),   # 右上
        (15, 55),   # 左上
    ]
    
    # === 手指远端 (20x25的矩形，在近端上面) ===
    finger_distal = [
        (15, 55),   # 左下
        (35, 55),   # 右下
        (35, 80),   # 右上
        (15, 80),   # 左上
    ]
    
    def draw_polygon(points, edge_colors):
        """
        画多边形的各条边，每条边可以不同颜色
        
        Args:
            points: 顶点列表 [(x,y), ...]
            edge_colors: 每条边的颜色 [color_idx, ...]
        """
        n = len(points)
        for i in range(n):
            start = points[i]
            end = points[(i + 1) % n]
            msp.add_line(start, end, dxfattribs={'color': edge_colors[i]})
    
    # 手掌：4条边都是黑色轮廓
    draw_polygon(palm, [7, 7, 7, 7])
    
    # 手指近端：
    # 下边(y=30, x:15-35) = 谷折（与手掌共用边界）
    # 上边(y=55, x:15-35) = 谷折（与远端共用边界）
    # 左右边 = 黑色轮廓
    draw_polygon(finger_proximal, [5, 7, 5, 7])  # 下,右,上,左
    # 注意：下边和上边都是 5（蓝色谷折）
    
    # 手指远端：
    # 下边(y=55, x:15-35) = 谷折（与近端共用边界）
    # 其余三边 = 黑色轮廓
    draw_polygon(finger_distal, [5, 7, 7, 7])  # 下,右,上,左
    
    doc.saveas(output_path)
    print(f"Saved test DXF to {output_path}")


def main():
    # 1. 生成测试DXF
    test_dxf = "tests/test_hand.dxf"
    create_test_dxf(test_dxf)
    
    # 2. 解析DXF
    print("\n" + "=" * 60)
    print("Parsing DXF...")
    print("=" * 60)
    parser = OrigamiParser(point_tolerance=1.0)
    design = parser.parse(test_dxf)
    
    # 3. 打印摘要
    print("\n" + "=" * 60)
    design.summary()
    print("=" * 60)
    
    # 4. 验证
    print("\nValidating design...")
    if not design.validate():
        print("\nDesign has errors. Face details for debugging:")
        for face_id, face in design.faces.items():
            print(f"\n  Face {face_id}:")
            print(f"    Vertices: {len(face.vertices)}")
            for i, v in enumerate(face.vertices):
                print(f"      v{i}: ({v.x:.1f}, {v.y:.1f})")
            print(f"    Area: {face.area:.1f}")
            print(f"    Edges: {face.edge_ids}")
        
        print("\nFold lines:")
        for lid, line in design.fold_lines.items():
            print(f"  Line {lid}: ({line.start.x:.1f},{line.start.y:.1f}) -> "
                  f"({line.end.x:.1f},{line.end.y:.1f}) [{line.fold_type.value}]")
        
        print("\nJoints:")
        for joint in design.joints:
            fl = design.fold_lines[joint.fold_line_id]
            print(f"  Joint {joint.id}: face{joint.face_a_id} -- face{joint.face_b_id} "
                  f"via line {joint.fold_line_id} [{fl.fold_type.value}]")
        
        print("\nFace tree:")
        for fid, pid in design.face_parent.items():
            print(f"  Face {fid} -> parent {pid}")
    
    # 5. 创建正向运动学
    print("\n" + "=" * 60)
    print("Creating forward kinematics...")
    print("=" * 60)
    fk = OrigamiForwardKinematics(design)
    
    # 6. 可视化
    print("\nStarting visualizer...")
    viz = OrigamiVisualizer()
    viz.open_browser()
    
    import time
    time.sleep(1.0)
    
    # 显示展开状态
    print("Displaying flat state (all joints at 0°)...")
    viz.display_hand(fk, {})
    
    # 7. 逐关节动画
    if design.joints:
        print(f"\nThere are {len(design.joints)} joints to animate.")
        for joint in design.joints:
            fl = design.fold_lines[joint.fold_line_id]
            input(f"\nPress Enter to fold joint {joint.id} "
                  f"(faces {joint.face_a_id}-{joint.face_b_id}, "
                  f"{fl.fold_type.value}) from 0° to 60°...")
            
            viz.animate_folding(fk, joint.id, 0.0, np.pi/3, steps=30)
    else:
        print("\nNo joints found! Check the design.")
    
    input("\nPress Enter to exit...")
    print("Done!")


if __name__ == "__main__":
    main()