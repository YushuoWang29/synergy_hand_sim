# MuJoCo Scanned Object Assets

This directory vendors a small subset of graspable MuJoCo objects from:

https://github.com/kevinzakka/mujoco_scanned_objects

Selected models:

- `ACE_Coffee_Mug_Kristen_16_oz_cup`
- `2_of_Jenga_Classic_Game`
- `HAMMER_BALL`

The upstream project provides MJCF object files generated from the Google Scanned Objects dataset. Each object directory includes a visual mesh, a texture, and multiple convex V-HACD collision meshes.

License notes:

- Upstream MJCF XML files are MIT licensed.
- The 3D object assets are distributed under CC-BY 4.0 according to the upstream repository.
- `LICENSE` and `UPSTREAM_README.md` are copied here for attribution and provenance.

In `synergy_hand_sim`, these assets can be referenced from an `.ohd` simulation definition using:

```json
{
  "name": "scanned_mug",
  "type": "scanned",
  "model_path": "../../assets/mujoco_scanned_objects/ACE_Coffee_Mug_Kristen_16_oz_cup/model.xml",
  "scale": 0.25,
  "pos": [0.045, 0.125, 0.0785]
}
```
