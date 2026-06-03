#!/usr/bin/env python3
"""
三步分段动力学仿真示例（ohd_2）

流程:
  第一步 (0~0.3s): sigma=5,  sigma_f=0  — 协同捏取
  第二步 (0.3~0.6s): sigma=5, sigma_f=-2 — 差动偏转 (反向)
  第三步 (0.6~0.9s): sigma=5, sigma_f=2  — 差动偏转 (正向)

用法:
    python scripts/run_three_phase_demo.py
    python scripts/run_three_phase_demo.py --viz
    python scripts/run_three_phase_demo.py --t-per 0.5 --save result.npz
"""

import sys, os, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.origami_design import OrigamiHandDesign
from src.simulation.config import SimulationConfig
from src.simulation.simulator import HandSimulator


def build_piecewise_sigma_f(t1=0.3, t2=0.6, val1=0.0, val2=-2.0, val3=2.0):
    """
    构造 sigma_f(t) 的分段常数表达式字符串。
    在 simulator._compute_inputs 中被 eval(t, np) 求值。
    """
    return (f"(t < {t1}) * ({val1}) + "
            f"(t >= {t1} and t < {t2}) * ({val2}) + "
            f"(t >= {t2}) * ({val3})")


def main():
    parser = argparse.ArgumentParser(description="三步分段动力学仿真")
    parser.add_argument("--ohd", default="models/ohd test/ohd_2.ohd",
                        help=".ohd 设计文件路径")
    parser.add_argument("--tau-M", type=float, default=5.0,
                        help="电机拉力 tau_M (默认 5.0)")
    parser.add_argument("--t-per", type=float, default=0.3,
                        help="每步时长 (秒, 默认 0.3)")
    parser.add_argument("--dt", type=float, default=1e-4,
                        help="时间步长 (默认 1e-4)")
    parser.add_argument("--save", type=str, default=None,
                        help="保存 .npz 路径")
    parser.add_argument("--viz", action="store_true",
                        help="显示可视化")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="详细输出")
    args = parser.parse_args()

    ohd_path = os.path.abspath(args.ohd)
    t_per = args.t_per
    t_total = t_per * 3

    # ================================================================
    # 1. 加载设计
    # ================================================================
    print("=" * 60)
    print(f"  三步分段动力学仿真")
    print(f"  设计: {ohd_path}")
    print(f"  tau_M = {args.tau_M}")

    print(f"  每步时长: {t_per}s × 3 = {t_total}s")
    print("=" * 60)

    design = OrigamiHandDesign.load(ohd_path)
    # 确保 face tree 建立（用于质量估计，避免 0 质量导致惯性矩阵退化）
    if design.faces and design.root_face_id is None:
        design.root_face_id = min(design.faces.keys())
        design.build_face_tree()
    print(f"  面片: {len(design.faces)}, "
          f"腱绳: {len(design.tendons)}, 滑轮: {len(design.pulleys)}")



    # ================================================================
    # 2. 配置 — 三分段控制输入
    # ================================================================
    cfg = SimulationConfig(
        dt=args.dt,
        t_end=t_total,
        phase=2,
        verbose=1 if args.verbose else 0,
    )

    # sigma(t) = 5 (常数，全行程协同激活)
    cfg.sigma_func = "5.0"
    # tau_M(t) = args.tau_m (常数，电机拉力)
    cfg.tau_M_func = f"{args.tau_M}"
    # sigma_f 三段: 0 → -2 → +2
    sigma_f_expr = build_piecewise_sigma_f(
        t1=t_per, t2=2*t_per,
        val1=0.0, val2=-2.0, val3=2.0
    )
    cfg.sigma_f_func = sigma_f_expr

    print(f"\n  sigma(t)     = 5.0 (常量)")
    print(f"  tau_M(t)     = {args.tau_M} (常量)")

    print(f"  sigma_f(t)   = {sigma_f_expr}")

    # ================================================================
    # 3. 构建仿真器并执行
    # ================================================================
    print("\n构建仿真引擎...")
    sim = HandSimulator(design, cfg)
    n_joints = sim.rbs.n_joints
    print(f"  关节数: {n_joints}, 总质量: {sim.rbs.get_total_mass()*1000:.1f}g")

    print(f"\n执行三步动力学仿真 (dt={cfg.dt}, 总步数约 {int(t_total/cfg.dt)})...")
    t0 = time.time()
    traj = sim.run(timeout=120.0)
    elapsed = time.time() - t0

    # ================================================================
    # 4. 输出结果摘要
    # ================================================================
    print(f"\n  [结果]")
    print(f"  物理时间: {traj.t[-1]*1000:.1f} ms (共 {len(traj.t)} 个采样点)")
    print(f"  仿真耗时: {elapsed*1000:.1f} ms")

    idx_step1 = np.searchsorted(traj.t, t_per)
    idx_step2 = np.searchsorted(traj.t, 2*t_per)
    idx_end = len(traj.t) - 1

    print(f"\n  三步结果摘要:")
    print(f"  {'步骤':>6s} | {'时间(s)':>8s} | {'sigma':>6s} | "
          f"{'sigma_f':>6s} | {'最终 q 向量':>30s}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*30}")

    for label, idx in [("起始", 0), ("步1(σ)", idx_step1),
                        ("步2(σ_f=-2)", idx_step2), ("步3(σ_f=+2)", idx_end)]:
        q_str = np.array2string(traj.q[idx], precision=3,
                                  suppress_small=True, max_line_width=80)
        print(f"  {label:>12s} | {traj.t[idx]:8.4f} |  5.0  | "
              f"{' 0.0' if idx==0 or idx<=idx_step1 else ('-2.0' if idx<=idx_step2 else ' 2.0')} | {q_str}")
        if traj.q_dot is not None:
            qd_str = np.array2string(traj.q_dot[idx], precision=3,
                                       suppress_small=True, max_line_width=80)
            print(f"  {'':>12s} | {'':>8s} | {'':>6s} | {'':>6s} | q_dot={qd_str}")

    # 保存
    if args.save:
        save_path = os.path.abspath(args.save)
        from src.simulation.io import SimulationWriter
        writer = SimulationWriter(save_path, overwrite=True)
        writer.save(traj, metadata={
            'ohd_file': os.path.basename(ohd_path),
            'sigma_func': '5.0',
            'sigma_f_func': sigma_f_expr,
            'tau_M': args.tau_M,
            't_per': t_per,

        })
        print(f"\n  已保存到: {save_path}")

    # 可视化
    if args.viz:
        from src.visualization.simulation_visualizer import SimulationVisualizer
        traj_data = {
            't': traj.t,
            'q': traj.q,
            'q_dot': traj.q_dot,
            'energy_kinetic': traj.energy_kinetic,
            'info': {'phase': 2, 'method': 'RK4', 'dt': cfg.dt,
                     'n_steps_total': len(traj.t), 'elapsed_s': elapsed},
            'metadata': f'{{"ohd": "{os.path.basename(ohd_path)}"}}',
        }
        viz = SimulationVisualizer(traj_data, design=design)
        viz.plot_dashboard(title=f"三步分段仿真 (σ=5, σ_f: 0→-2→+2)")
        viz.show_all(block=True)

    print("\n完成 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
