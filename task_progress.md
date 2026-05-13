# Task: 提升 MuJoCo 仿真可视化质量

## Completed ✓
- [x] 检查 MuJoCo 版本和渲染能力（3.8.0 = 完整版，支持阴影/反光/抗锯齿）
- [x] 分析当前渲染问题（纯黑背景、巨大白色平面、无材质、光照差）
- [x] 调查 MuJoCo 可视化API
- [x] 修改 `_urdf_to_mjcf()` - MJCF 生成部分
  - [x] 添加 `<visual>` 块配置暖灰色背景 (haze/fog)
  - [x] 添加 3 盏场景灯光（三点照明：主光/补光/背光，带投射阴影）
  - [x] 移除巨大白色平面矩形背景
  - [x] 添加材质定义 (origami_plastic, origami_accent) 并应用到几何体
  - [x] 设置 8x 抗锯齿、大阴影贴图 (8192)
  - [x] 优化雾/景深参数
- [x] 修改 `_load_model()` - 应用渲染质量设置（offsamples, shadowsize, linewidth）
- [x] 清理临时调试脚本
- [x] 语法验证和MJCF生成测试（全部16检查项通过）

## 验证结果
- 模块导入正常: `import` OK
- MJCF 生成: 全部 16 项视觉检查通过
- 产生的 XML 包含: 背景色、3盏灯光、材质定义、阴影、抗锯齿、雾效等
