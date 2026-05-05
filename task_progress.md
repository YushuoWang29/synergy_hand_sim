# Task Progress: Reproduce Paper Results for 3-Finger Gripper

## Results

### ✅ R Matrix (both ohd_2 and ohd_3)
```
R = [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
```
All 6 joints have pulleys correctly mapped. Previously was [7,7,7,0,0,0].

### ✅ R_f Matrix
```
ohd_2 (Routing 1): Rf = [13.65,  3.15, -11.55,  3.15, -11.55,  13.65]
ohd_3 (Routing 2): Rf = [ 3.15, 13.65, -11.55, 13.65, -11.55,   3.15]
```
R_f differs between the two routings as expected.

### ✅ Synergy Directions (paper match!)
After reordering joints to paper convention [F1_prox, F1_dist, F2_prox, F2_dist, F3_prox, F3_dist]:
```
ohd_2 Direction 1: [17, 17, 2, 2, -19, -19] ← MATCHES paper (33)
ohd_3 Direction 1: [2, 2, 17, 17, -19, -19] ← MATCHES paper (36)
```

### ✅ MuJoCo Simulation
- Updated transmission_builder.py: no longer calls build_topology(), preserves original fold line IDs
- MuJoCo simulator already uses dummy joints with kp=0 for synergy mode

## What Was Fixed
1. **Bug 1 - Pulley mapping**: `get_joint_list()` used `design.build_topology()` which destroyed fold line IDs. Fixed by creating joints directly from fold-type fold lines.
2. **Bug 2 - R_f calculation**: Old code had wrong combinatorial sums. Fixed to use proper M^{-1} * V_max * e_v formulation from paper.
3. **Bug 3 - MuJoCo not moving**: Already fixed in previous iteration (dummy joints with kp=0).

## Still To Do
- [x] Fix pulley-to-joint mapping
- [x] Fix R_f computation  
- [x] Verify R = [7,7,7,7,7,7] for both models
- [x] Verify R_f differs between ohd_2 and ohd_3
- [x] Verify synergy directions match paper predictions
- [ ] User to test MuJoCo visualization
