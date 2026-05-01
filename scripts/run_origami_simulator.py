# scripts/run_origami_simulator.py
"""
交互式折纸手仿真器启动脚本

用法：
    python scripts/run_origami_simulator.py tests/test_hand.dxf
"""

import sys
sys.path.insert(0, '.')

import argparse
from PyQt5 import QtWidgets
from src.models.origami_parser import OrigamiParser
from src.interactive.origami_simulator import OrigamiSimulator


def main():
    parser = argparse.ArgumentParser(description="Origami Hand Interactive Simulator")
    parser.add_argument(
        "dxf_file",
        help="Path to DXF file with origami hand design"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Point merging tolerance (default: 1.0 mm)"
    )
    args = parser.parse_args()
    
    # 解析DXF
    print("=" * 60)
    print("Parsing DXF file...")
    print("=" * 60)
    
    dxf_parser = OrigamiParser(point_tolerance=args.tolerance)
    design = dxf_parser.parse(args.dxf_file)
    
    design.summary()
    
    if not design.validate():
        print("\nDesign validation failed. Some features may not work correctly.")
    
    if len(design.joints) == 0:
        print("\nWARNING: No joints found in the design!")
        print("Make sure your DXF has red (mountain) or blue (valley) fold lines.")
    
    # 启动交互式仿真器
    print("\n" + "=" * 60)
    print("Starting interactive simulator...")
    print("=" * 60)
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    
    simulator = OrigamiSimulator(design)
    simulator.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()