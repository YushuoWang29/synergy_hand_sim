# scripts/run_pinocchio_simulator.py

import sys
sys.path.insert(0, '.')

import argparse
import os
from PyQt5 import QtWidgets

from src.models.origami_parser import OrigamiParser
from src.interactive.pinocchio_simulator import PinocchioSimulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf_file", help="Path to URDF file")
    parser.add_argument("--dxf", default=None, help="Path to DXF (auto-detected if not specified)")
    parser.add_argument("--tolerance", type=float, default=2.0)
    args = parser.parse_args()

    # 确定 DXF 路径
    if args.dxf:
        dxf_path = args.dxf
    else:
        # 从 URDF 路径推断 DXF：models/<name>/<name>.urdf → tests/<name>.dxf
        urdf_dir = os.path.dirname(os.path.abspath(args.urdf_file))
        hand_name = os.path.splitext(os.path.basename(args.urdf_file))[0]
        dxf_path = os.path.join(urdf_dir, "..", "..", "tests", f"{hand_name}.dxf")
        dxf_path = os.path.normpath(dxf_path)

    if not os.path.exists(dxf_path):
        print(f"DXF not found: {dxf_path}")
        print("Use --dxf to specify the DXF path manually.")
        return

    urdf_path = os.path.abspath(args.urdf_file)
    mesh_dir = os.path.join(os.path.dirname(urdf_path), "meshes")

    print(f"URDF: {urdf_path}")
    print(f"DXF:  {dxf_path}")
    print(f"Mesh: {mesh_dir}")

    # 解析 DXF（仅用于 2D 视图）
    print("Parsing DXF for 2D view...")
    dxf_parser = OrigamiParser(point_tolerance=args.tolerance)
    design = dxf_parser.parse(dxf_path)
    design.summary()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    simulator = PinocchioSimulator(design, urdf_path, mesh_dir)
    simulator.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()