# scripts/run_mujoco_simulator.py
"""
启动基于 MuJoCo 的交互式折纸手仿真器

用法：
    conda activate synergy_hand_sim
    python scripts/run_mujoco_simulator.py models/test_3/test_3.urdf
    python scripts/run_mujoco_simulator.py models/test_3/test_3.urdf --timeout 5
"""

import sys
sys.path.insert(0, '.')

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo 交互式仿真器 - 用于折纸手 URDF 可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 加载 test_3 模型并可视化
  python scripts/run_mujoco_simulator.py models/test_3/test_3.urdf

  # 自动化：启动后 5 秒自动退出（用于无交互测试）
  python scripts/run_mujoco_simulator.py models/test_3/test_3.urdf --timeout 5
        """
    )
    parser.add_argument("urdf_file", help="URDF 文件路径")
    parser.add_argument("--no-left-ui", action="store_true",
                        help="隐藏查看器左侧 UI 面板")
    parser.add_argument("--no-right-ui", action="store_true",
                        help="隐藏查看器右侧关节滑块面板")
    parser.add_argument("--timeout", type=float, default=None,
                        help="自动退出超时秒数（默认无限等待）")
    args = parser.parse_args()

    urdf_path = os.path.abspath(args.urdf_file)
    mesh_dir = os.path.join(os.path.dirname(urdf_path), "meshes")

    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found: {urdf_path}")
        sys.exit(1)

    if not os.path.exists(mesh_dir):
        print(f"Error: Mesh directory not found: {mesh_dir}")
        print("Please run 'scripts/export_urdf.py' first to generate meshes.")
        sys.exit(1)

    print(f"URDF: {urdf_path}")
    print(f"Mesh: {mesh_dir}")

    # 启动 MuJoCo 仿真器
    from src.interactive.mujoco_simulator import MuJoCoSimulator

    simulator = MuJoCoSimulator(
        urdf_path=urdf_path,
        mesh_dir=mesh_dir,
        show_left_ui=not args.no_left_ui,
        show_right_ui=not args.no_right_ui
    )

    print("\nMuJoCo 仿真器已启动！")
    print("  - MuJoCo 3D 查看器窗口（独立窗口）")
    print("  - 右侧面板可调节关节角度：")
    print("      「Position」滑块 === 可交互拖动控制关节角度")
    print("      「Joint」滑块（灰色）=== 只读显示，忽略即可")
    print("  - 鼠标滚轮可任意拉远/拉近视角")
    print("  - 关闭查看器窗口即可退出\n")

    simulator.run(timeout=args.timeout)


if __name__ == "__main__":
    main()
