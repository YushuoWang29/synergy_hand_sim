# MuJoCo SDAS Numerical Simulation Upgrade

## Scope

The upgraded path converts the previous geometry-only hand motion into a MuJoCo
numerical simulation driven by SDAS distributions. The implementation does not
use damper components, velocity damping, viscous tendon friction, or any manual
`-D qdot` term.

## Literature Basis

The SoftHand 2 paper gives the baseline synergy relation:

```text
q = S sigma
```

and the soft synergy equilibrium:

```text
q = S sigma - C J^T f_ext
```

For adaptive synergies, a tendon transmission matrix maps actuator displacement
to joint configuration:

```text
R q = x
tau = R^T tau_M
J^T f_ext = R^T tau_M - E q
```

In free motion, this yields the transmission-derived synergy matrix:

```text
S_R = E^{-1} R^T (R E^{-1} R^T)^{-1}
q = S_R x
```

SoftHand 2 further introduces a second direction from sliding/friction. For this
project, that idea is retained as a configurable distribution matrix:

```text
q_ref(t) = B_sigma sigma(t)
```

MuJoCo then integrates the articulated model numerically. The full SoftHand 2
dynamic expression

```text
B(q) qddot + W(q, qdot) qdot + Gamma(q) = Q(q) u + J(q)^T f_ext
```

is used as motivation for moving to MuJoCo, but the implemented path excludes
the dissipative parts of `W`, joint-level velocity friction, tendon viscous
friction, static-friction memory states, damper transmissions, and motor PD
velocity terms.

## Implementation Path

The new entry point is:

```text
python scripts/run_mujoco_sdas.py "models/ohd test/mujoco_sdas_step.ohd"
```

Core module:

```text
src/simulation/mujoco_sdas.py
```

Main layers:

1. OHD simulation parser: loads either a hand-design OHD or a simulation-definition OHD.
2. Distribution builder: supports `sdas`, `transmission`, `joint_space`, `endpoint`, `uniform`, and custom matrices.
3. Driver evaluator: supports `position`, `velocity`, and `force` time-series inputs.
4. MuJoCo executor: builds a no-damper MJCF from URDF, applies generalized joint forces, calls `mj_step`, and records logs/renders.

## Control Law

For position and velocity inputs, driver values are accumulated into the synergy
coordinate vector `sigma_pos`. Geometry provides the target:

```text
q_ref = B_sigma sigma_pos
```

MuJoCo receives a spring-like generalized force:

```text
tau_pos = Kp (q_ref - q)
```

For force inputs:

```text
tau_force = force_scale B_sigma sigma_force
```

The applied generalized force is:

```text
tau = tau_pos + tau_force
```

No velocity feedback is used.

## OHD Simulation Definition Schema

```json
{
  "hand_model_path": "three_finger_gripper_b.ohd",
  "urdf_path": "../three_finger_gripper_b/three_finger_gripper_b.urdf",
  "simulation": {
    "label": "mujoco_sdas_step",
    "duration": 1.2,
    "dt": 0.002,
    "distribution": {
      "type": "sdas"
    },
    "drivers": [
      {
        "name": "sigma",
        "type": "position",
        "samples": [[0.0, 0.0], [0.2, 8.0], [1.2, 8.0]]
      }
    ],
    "output": {
      "dir": "../../outputs/mujoco_sdas/step_demo",
      "screenshots": [0.35, 1.05]
    }
  }
}
```

## Outputs

The runner writes:

- `*_log.csv`: time, joint angles, geometry targets, driver inputs, applied generalized forces, and contact count.
- `*_log.npz`: compressed numerical trajectory arrays.
- `*_geometry_vs_mujoco.png`: geometry target vs. MuJoCo state comparison.
- `*_t<time>.png`: MuJoCo offscreen render screenshots.
- `*.xml`: generated no-damper MJCF for audit.

## Verification Result

The demo run uses `three_finger_gripper_b` with two SDAS inputs. Generated MJCF
was checked to contain no `damping`, `damper`, or `velocity` terms. The demo run
produces two MuJoCo rendered postures:

```text
outputs/mujoco_sdas/step_demo/mujoco_sdas_step_t0.350.png
outputs/mujoco_sdas/step_demo/mujoco_sdas_step_t1.050.png
```

The geometry-vs-MuJoCo comparison reports:

```text
RMS difference: 0.1848 rad
Max absolute difference: 0.5661 rad
```

