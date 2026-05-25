#!/usr/bin/env python3
"""
力学数值求解仿真入口
======================
从 .ohd 文件加载设计，运行三层递进力学仿真（Phase 1-3）。
支持通过 --interactive 分步输入 sigma 和 sigma_f。

用法:
    # Phase 1 - 准静态力平衡（默认）
    python scripts/run_numerical_simulation.py "models/ohd test/ohd_1.ohd"

    # Phase 2 - 完整动力学 ODE 积分
    python scripts/run_numerical_simulation.py "models/ohd test/ohd_1.ohd" --phase 2

    # 指定驱动器输入 u=[MotorA, MotorB]
    python scripts/run_numerical_simulation.py "models/ohd test/ohd_1.ohd" --u 10 0

    # 交互式 step-by-step 输入 sigma 和 sigma_f (缩写 S 和 F)
    python scripts/run_numerical_simulation.py "models/ohd test/ohd_1.ohd" --interactive --phase 1

    # 设置仿真时长和时步
    python scripts/run_numerical_simulation.py "models/ohd test/ohd_1.ohd" --phase 2 --dt 1e-4 --t-end 0.2

    # 保存仿真结果到 npz 文件
    python scripts/run_numerical_simulation.py "models/ohd test/ohd_1.ohd" --phase 2 --save results.npz

    # 详细输出
    python scripts/run_numerical_simulation.py "models/ohd test/ohd_1.ohd" --phase 2 -v
"""

import sys, os, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.origami_design import OrigamiHandDesign
from src.simulation.config import SimulationConfig
from src.simulation.simulator import HandSimulator, SimulationTrajectory
from src.simulation.io import SimulationWriter
from src.simulation.transmission_force import compute_Q_matrix


def build_u(motor_a: float, motor_b: float, n_tendons: int) -> np.ndarray:
    """构建驱动器输入向量 u (3 分量: tau_M, sigma, sigma_f).
    
    从两个电机输入 A, B 推导:
        sigma   = (A + B) / 2  (协同分量)
        sigma_f = (A - B) / 2  (差动分量)
        tau_M   = A            (电机拉力幅度)
    """
    if n_tendons == 0:
        return np.array([0.0, 0.0, 0.0])
    sigma = (motor_a + motor_b) / 2.0
    sigma_f = (motor_a - motor_b) / 2.0
    return np.array([motor_a, sigma, sigma_f])


def build_u_from_sigma(sigma: float, sigma_f: float,
                       tau_M: float = 5.0) -> np.ndarray:
    """从 sigma/sigma_f 直接构建驱动器输入向量 u.
    
    u[0] = tau_M * sigma   (电机拉力 × 协同激活)
    u[1] = sigma           (协同激活)
    u[2] = sigma_f         (差动激活)
    
    这是对 --interactive 模式的支持函数。
    """
    return np.array([tau_M * sigma, sigma, sigma_f])


def interactive_input():
    """
    交互式分步输入 sigma 和 sigma_f.
    
    流程:
        第一步: 输入 sigma (或缩写 S), 例如 'S 0.5' 或 'sigma 0.5'
        第二步: 输入 sigma_f (或缩写 F), 例如 'F 0.1' 或 'sigma_f 0.1'
        
    返回:
        sigma, sigma_f : float
    """
    print("\n" + "=" * 60)
    print("  交互式驱动器输入")
    print("  输入格式: <名称> <数值>")
    print("  名称支持: sigma (S), sigma_f (F)")
    print("  示例: S 0.5   或   sigma 0.5")
    print("         F 0.1  或   sigma_f 0.1")
    print("  输入 'q' 或 'quit' 退出")
    print("=" * 60)

    sigma = None
    sigma_f = None

    # 第一步: 输入 sigma
    while sigma is None:
        try:
            raw = input("\n第一步 (sigma / S) = ").strip()
            if raw.lower() in ('q', 'quit', 'exit'):
                sys.exit(0)
            parts = raw.split()
            if len(parts) == 1:
                # 仅输入数值，默认名称是 sigma
                val = float(parts[0])
                sigma = val
            elif len(parts) >= 2:
                name = parts[0].lower()
                val = float(parts[1])
                if name in ('sigma', 's'):
                    sigma = val
                else:
                    print(f"  未知名称 '{parts[0]}'，请使用 sigma 或 S")
                    continue
            print(f"  设置 sigma = {sigma}")
        except (ValueError, IndexError):
            print("  输入无效，请重新输入。例如: S 0.5")

    # 第二步: 输入 sigma_f
    while sigma_f is None:
        try:
            raw = input("\n第二步 (sigma_f / F) = ").strip()
            if raw.lower() in ('q', 'quit', 'exit'):
                sys.exit(0)
            parts = raw.split()
            if len(parts) == 1:
                val = float(parts[0])
                sigma_f = val
            elif len(parts) >= 2:
                name = parts[0].lower()
                val = float(parts[1])
                if name in ('sigma_f', 'f'):
                    sigma_f = val
                else:
                    print(f"  未知名称 '{parts[0]}'，请使用 sigma_f 或 F")
                    continue
            print(f"  设置 sigma_f = {sigma_f}")
        except (ValueError, IndexError):
            print("  输入无效，请重新输入。例如: F 0.1")

    print(f"\n  最终: sigma={sigma}, sigma_f={sigma_f}")
    return sigma, sigma_f


def main():
    parser = argparse.ArgumentParser(
        description="折纸手力学数值求解器 - 三层递进架构 (Phase 1/2/3)")
    parser.add_argument("ohd_file", help=".ohd 设计文件路径")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="仿真阶段: 1=准静态 (默认), 2=动力学, 3=动力学+接触")
    parser.add_argument("--u", nargs=2, type=float, default=[5.0, 0.0],
                        metavar=("MOTOR_A", "MOTOR_B"),
                        help="驱动器输入 A B (默认: 5.0 0.0)")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式分步输入 sigma 和 sigma_f (缩写 S 和 F)")
    parser.add_argument("--tau-m", type=float, default=5.0,
                        help="电机拉力 tau_M (仅 --interactive 模式使用, 默认: 5.0)")
    parser.add_argument("--dt", type=float, default=1e-4,
                        help="时间步长 (默认: 1e-4)")
    parser.add_argument("--t-end", type=float, default=None,
                        help="仿真结束时间 (默认: 0.1)")
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="准静态求解容差 (默认: 1e-8)")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="准静态最大迭代次数 (默认: 50)")
    parser.add_argument("--record-every", type=int, default=10,
                        help="记录步长间隔 (默认: 10)")
    parser.add_argument("--save", type=str, default=None,
                        help="保存轨迹到 .npz 文件路径")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="仿真超时秒数 (默认: 60)")
    parser.add_argument("--viz", action="store_true",
                        help="打开可视化 (2D 轨迹图 + 能量图)")
    parser.add_argument("--viz-save", type=str, default=None,
                        help="保存可视化结果 PNG 到指定前缀路径")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="详细输出")
    args = parser.parse_args()

    # ================================================================
    # 自动设定默认值
    # ================================================================
    if args.t_end is None:
        args.t_end = 0.1

    # ================================================================
    # 1. 加载设计
    # ================================================================
    ohd_path = os.path.abspath(args.ohd_file)
    if not os.path.exists(ohd_path):
        print(f"[错误] 文件不存在: {ohd_path}")
        return 1

    print("=" * 60)
    print(f"  数值力学仿真入口")
    print(f"  设计文件: {ohd_path}")
    print(f"  仿真阶段: Phase {args.phase}", end="")
    if args.phase == 1:
        print(" (准静态力平衡)")
    elif args.phase == 2:
        print(" (完整动力学 ODE)")
    else:
        print(" (动力学)")
    print("=" * 60)

    print("\n[1/4] 加载设计...")
    design = OrigamiHandDesign.load(ohd_path)
    n_fold = len(design.fold_lines)
    n_joints_from_design = 0
    if hasattr(design, 'joints') and design.joints:
        n_joints_from_design = len(design.joints)
    print(f"  折痕线: {n_fold} 条, 关节: {n_joints_from_design} 个, "
          f"腱绳: {len(design.tendons)} 条, 滑轮: {len(design.pulleys)} 个")

    # ================================================================
    # 2. 构建配置和仿真器
    # ================================================================
    print("\n[2/4] 配置仿真参数...")
    cfg = SimulationConfig(
        dt=args.dt,
        t_end=args.t_end,
        phase=args.phase,
        quasi_static_tol=args.tol,
        quasi_static_max_iter=args.max_iter,
        record_every=args.record_every,
        verbose=1 if args.verbose else 0,
    )
    n_steps = int(cfg.t_end / cfg.dt) + 1
    print(f"  dt={cfg.dt}, t_end={cfg.t_end}s, "
          f"steps={n_steps}, record_every={cfg.record_every}")
    print(f"  tol={cfg.quasi_static_tol:.0e}, max_iter={cfg.quasi_static_max_iter}")

    if not design.faces or len(design.faces) == 0:
        print("\n  重建面片拓扑 (用于可视化)...")
        try:
            design.build_topology()
            if design.faces:
                print(f"  成功重建 {len(design.faces)} 个面片")
        except Exception as e:
            print(f"  面片重建跳过 ({e})")

    print("\n[3/4] 构建仿真引擎...")
    t0 = time.time()
    sim = HandSimulator(design, cfg)
    n_joints = sim.rbs.n_joints
    build_time = time.time() - t0
    print(f"  求解器引擎, 关节数: {n_joints}, "
          f"构建耗时: {build_time*1000:.1f} ms")

    # ================================================================
    # 交互式 / 命令行输入构建 u
    # ================================================================
    if args.interactive:
        sigma, sigma_f = interactive_input()
        u = build_u_from_sigma(sigma, sigma_f, tau_M=args.tau_m)
        print(f"  驱动器输入 u (tau_M=sigma*tauM, sigma, sigma_f): {u}")
    else:
        u = build_u(args.u[0], args.u[1], len(design.tendons))
        print(f"  驱动器输入 u (A={args.u[0]}, B={args.u[1]}): {u}")

    # 设置控制函数
    ramp_time = max(cfg.t_end * 0.3, 0.01)
    cfg.sigma_func = f"min(t/{ramp_time:.4f}, 1.0)"
    # u = [tau_M * sigma, sigma, sigma_f]; use sigma_f from u[2]
    cfg.sigma_f_func = f"{u[2]:.2f}"

    # ================================================================
    # 4. 执行仿真
    # ================================================================
    print(f"\n[4/4] 运行 Phase {args.phase}...")
    t0 = time.time()

    if args.phase == 1:
        q0 = np.zeros(n_joints)
        result = sim.run_quasistatic(q0, u=u)
        elapsed = time.time() - t0

        print(f"\n  [准静态求解结果]")
        print(f"  收敛: {'✓' if result.converged else '✗'}")
        print(f"  迭代次数: {result.n_iterations}")
        print(f"  残差范数: {np.linalg.norm(result.residual):.4e}")
        print(f"  求解耗时: {elapsed*1000:.1f} ms")
        print(f"  平衡 q: {np.array2string(result.q, precision=4, suppress_small=True)}")

        for i in range(n_joints):
            print(f"    关节[{i}]: {result.q[i]:+.4f} rad = "
                  f"{np.degrees(result.q[i]):+.1f}°")

    elif args.phase in (2, 3):
        traj = sim.run(timeout=args.timeout)
        elapsed = time.time() - t0

        print(f"\n  [动力学仿真结果]")
        print(f"  积分步数: {traj.n_steps}")
        print(f"  物理时间: {traj.t[-1]*1000:.2f} ms  (规划: {cfg.t_end*1000:.2f} ms)")
        print(f"  仿真耗时: {elapsed*1000:.1f} ms")
        print(f"  采样点数: {len(traj.t)}")

        print(f"\n  初始 q: {np.array2string(traj.q[0], precision=4, suppress_small=True)}")
        print(f"  最终 q: {np.array2string(traj.q[-1], precision=4, suppress_small=True)}")

        if traj.q_dot is not None and len(traj.q_dot) > 0:
            print(f"  最终 q_dot: {np.array2string(traj.q_dot[-1], precision=4, suppress_small=True)}")

        if args.save:
            save_path = os.path.abspath(args.save)
            print(f"\n  保存轨迹到: {save_path}")
            writer = SimulationWriter(save_path, overwrite=True)
            writer.save(traj, metadata={
                'ohd_file': os.path.basename(ohd_path),
                'phase': args.phase,
                'u': u.tolist(),
                'dt': cfg.dt,
                't_end': cfg.t_end,
            })
            print(f"  已保存 ({os.path.getsize(save_path)/1024:.1f} KB)")

    # ================================================================
    # 5. 可视化
    # ================================================================
    if args.viz and traj is not None:
        from src.visualization.simulation_visualizer import SimulationVisualizer
        print("\n[5/4] 可视化仿真结果...")

        if args.phase == 1:
            traj_data = {
                't': np.array([0.0, 1.0]),
                'q': np.array([result.q, result.q]),
                'q_dot': np.zeros((2, n_joints)),
                'energy_kinetic': np.array([0.0, 0.0]),
                'info': {
                    'phase': 1,
                    'converged': result.converged,
                    'n_iterations': result.n_iterations,
                    'residual': float(np.linalg.norm(result.residual)),
                }
            }
            viz = SimulationVisualizer(traj_data, design=design)
            viz.plot_quasistatic_bars(q_eq=result.q, q0=q0)
            viz.plot_dashboard(title="Phase 1 - 准静态力平衡")
        else:
            traj_data = {
                't': traj.t,
                'q': traj.q,
                'q_dot': traj.q_dot if hasattr(traj, 'q_dot') and traj.q_dot is not None else None,
                'q_ddot': traj.q_ddot if hasattr(traj, 'q_ddot') else None,
                'inputs': traj.inputs if hasattr(traj, 'inputs') else None,
                'energy_kinetic': traj.energy_kinetic,
                'energy_potential': traj.energy_potential if hasattr(traj, 'energy_potential') else None,
                'energy_dissipated': traj.energy_dissipated if hasattr(traj, 'energy_dissipated') else None,
                'info': {
                    'phase': args.phase,
                    'method': 'RK4',
                    'dt': cfg.dt,
                    'n_steps_total': traj.n_steps,
                    'elapsed_s': elapsed,
                },
                'metadata': f'{{"ohd_file": "{os.path.basename(ohd_path)}", "phase": {args.phase}, "u": {u.tolist()}}}',
            }
            viz = SimulationVisualizer(traj_data, design=design)
            viz.plot_dashboard(title=f"Phase {args.phase} - 力学仿真 Dashboard")

        if args.viz_save:
            viz.save_all(prefix=args.viz_save)
            print(f"  图形已保存到 {args.viz_save}_*.png")

        viz.show_all(block=True)
        print("  可视化完成.")

    # ================================================================
    # 完成
    # ================================================================
    print("\n" + "=" * 60)
    print("  完成 ✓")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
