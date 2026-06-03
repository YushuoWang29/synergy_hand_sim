# MuJoCo 接触与自适应抓取 Demo

本文记录本项目新增的接触/碰撞接口、GitHub 物体资产来源，以及四个可复现的折纸灵巧手自适应抓取 demo。

## 资产来源

本次检索了两个 MuJoCo 资产来源：

| 来源 | 链接 | 采用情况 |
|---|---|---|
| MuJoCo Menagerie | https://github.com/google-deepmind/mujoco_menagerie | 作为参考来源，偏机器人本体模型 |
| MuJoCo Scanned Objects | https://github.com/kevinzakka/mujoco_scanned_objects | 已采用，适合抓取物体 demo |

本项目实际 vendored 了 `mujoco_scanned_objects` 中的三个物体：

| 物体 | 本地路径 |
|---|---|
| coffee mug | `assets/mujoco_scanned_objects/ACE_Coffee_Mug_Kristen_16_oz_cup` |
| Jenga block | `assets/mujoco_scanned_objects/2_of_Jenga_Classic_Game` |
| hammer ball | `assets/mujoco_scanned_objects/HAMMER_BALL` |

该资产目录中的 `README.md`、`UPSTREAM_README.md` 和 `LICENSE` 记录了来源与许可。上游 XML 为 MIT license，3D 资产按上游说明为 CC-BY 4.0。

## 新增接口

`.ohd` 仿真定义现在支持：

```json
"contact": {
  "enabled": true,
  "hand_contact": true,
  "object_contact": true,
  "floor_contact": true,
  "hand_floor_contact": false,
  "friction": [1.1, 0.01, 0.0001],
  "margin": 0.0002,
  "floor_z": -0.015
}
```

以及物体定义：

```json
"objects": [
  {
    "name": "adaptive_cylinder",
    "type": "cylinder",
    "pos": [0.045, 0.125, 0.0785],
    "size": [0.018, 0.03],
    "rgba": [0.20, 0.62, 0.58, 1.0],
    "mass": 0.5,
    "freejoint": true
  }
]
```

外部 scanned object 可写为：

```json
"objects": [
  {
    "name": "scanned_mug",
    "type": "scanned",
    "model_path": "../../assets/mujoco_scanned_objects/ACE_Coffee_Mug_Kristen_16_oz_cup/model.xml",
    "scale": 0.45,
    "pos": [0.045, 0.125, 0.0785],
    "mass": 0.5,
    "freejoint": true
  }
]
```

## 碰撞分组

生成的 MJCF 使用 MuJoCo `contype/conaffinity` 进行分组：

| 几何体 | `contype` | `conaffinity` | 说明 |
|---|---:|---:|---|
| hand | 1 | 2 | 手只与物体碰撞，默认不自碰撞 |
| object | 2 | 5 | 物体与手和地面碰撞 |
| floor | 4 | 2 | 地面与物体碰撞 |

这样做可以避免折纸手 mesh 自碰撞把运动卡住，同时保留手-物体、物体-地面的接触。

实现仍然不包含阻尼器：不生成 `<damper>`，不写 `damping`，也没有 `-D q_dot` 速度阻尼反馈。

## Demo 命令

```powershell
python scripts\run_mujoco_sdas.py "models\ohd test\mujoco_sdas_grasp_box.ohd"
python scripts\run_mujoco_sdas.py "models\ohd test\mujoco_sdas_grasp_cylinder.ohd"
python scripts\run_mujoco_sdas.py "models\ohd test\mujoco_sdas_grasp_sphere.ohd"
python scripts\run_mujoco_sdas.py "models\ohd test\mujoco_sdas_grasp_scanned_mug.ohd"
```

## 结果汇总

| Demo | 分布 | 接触步数 | 最大接触数 | 最大总接触力 | 截图目录 |
|---|---|---:|---:|---:|---|
| box | `sdas` | 146 | 5 | 4.395 | `outputs/mujoco_sdas/grasp_box` |
| cylinder | `sdas` | 158 | 5 | 4.415 | `outputs/mujoco_sdas/grasp_cylinder` |
| sphere | `uniform` | 531 | 2 | 120.234 | `outputs/mujoco_sdas/grasp_sphere` |
| scanned mug | `sdas` | 209 | 7 | 0.623 | `outputs/mujoco_sdas/grasp_scanned_mug` |

推荐优先展示 cylinder 和 scanned mug：它们既有清楚的手-物体接触画面，也有比较温和的接触力数值。

## 展示截图

![box grasp](../outputs/mujoco_sdas/grasp_box/mujoco_sdas_grasp_box_t1.250.png)

![cylinder grasp](../outputs/mujoco_sdas/grasp_cylinder/mujoco_sdas_grasp_cylinder_t1.250.png)

![sphere grasp](../outputs/mujoco_sdas/grasp_sphere/mujoco_sdas_grasp_sphere_t1.250.png)

![scanned mug grasp](../outputs/mujoco_sdas/grasp_scanned_mug/mujoco_sdas_grasp_scanned_mug_t1.250.png)

## 输出日志

CSV/NPZ 中新增：

| 字段 | 含义 |
|---|---|
| `ncon` | 当前 MuJoCo 接触点数量 |
| `contact_force_total` | 当前所有接触点的力范数求和 |
| `object_<name>_x/y/z` | 物体世界坐标 |

`summary.json` 中新增：

| 字段 | 含义 |
|---|---|
| `object_count` | 物体数量 |
| `contact_enabled` | 是否打开接触 |
| `contact_steps` | 出现接触的仿真步数 |
| `max_contacts` | 单步最大接触点数量 |
| `max_contact_force` | 单步最大总接触力 |

## 当前解释边界

这些 demo 的目标是证明 `.ohd -> SDAS/control distribution -> MuJoCo contact simulation` 已经打通。它们还不是完整硬件级抓取评估，因为当前折纸手 link 仍使用简化 mesh 碰撞，指尖厚度、柔性材料、真实重力抓取和接触力标定还需要后续细化。

更适合论文中的表述是：

> 我们实现了基于 MuJoCo 的接触感知 SDAS 数值仿真接口，并通过 box、cylinder、sphere 和 scanned mug 四类物体验证了折纸灵巧手在低维协同输入下能够与外部物体发生可记录的碰撞和接触响应。
