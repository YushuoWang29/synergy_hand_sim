# scripts/export_urdf.py
"""
将DXF文件导出为URDF格式

输出结构：
    models/<hand_name>/
        ├── <hand_name>.urdf
        └── meshes/
            ├── face_0.stl
            ├── face_1.stl
            └── ...
"""

import sys
sys.path.insert(0, '.')

import argparse
import os
from src.models.origami_parser import OrigamiParser
from src.models.origami_to_urdf import export_urdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf_file")
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--thickness", "-t", type=float, default=3.0, help="Material thickness for STL generation (in mm)")
    args = parser.parse_args()
    
    # 从DXF文件名提取手部名称
    dxf_path = os.path.abspath(args.dxf_file)
    hand_name = os.path.splitext(os.path.basename(dxf_path))[0]
    
    # 输出目录：models/<hand_name>/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "models", hand_name)
    os.makedirs(output_dir, exist_ok=True)
    
    urdf_path = os.path.join(output_dir, f"{hand_name}.urdf")
    
    print(f"Parsing: {args.dxf_file}")
    dxf_parser = OrigamiParser(point_tolerance=args.tolerance)
    design = dxf_parser.parse(args.dxf_file)
    design.summary()
    
    print(f"\nExporting to: {output_dir}/")
    export_urdf(design, urdf_path, thickness=args.thickness)
    print(f"Done! URDF: {urdf_path}")


if __name__ == "__main__":
    main()