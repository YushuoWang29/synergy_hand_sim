#!/usr/bin/env python3
"""
v4: KEY DIAGNOSTIC - Compare Q matrix BEFORE and AFTER topology rebuild.
The hypothesis is that build_topology() changes fold_line IDs, breaking the 
pulley->joint mapping, leading to wrong Q matrices inside the simulator.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.origami_design import OrigamiHandDesign
from src.models.transmission_builder import get_joint_list
from src.simulation.transmission_force import compute_Q_matrix
from src.simulation.config import SimulationConfig
from src.simulation.simulator import HandSimulator

ALL_OHD = ['ohd_2', 'ohd_3', 'ohd_5', 'ohd_6']

def diagnose_ohd(name):
    path = f'models/ohd test/{name}.ohd'
    if not os.path.exists(path):
        print(f"[SKIP] {name}: file not found")
        return
    
    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS: {name}")
    print(f"{'='*70}")
    
    # ========================
    # 1. Load and inspect original design
    # ========================
    design = OrigamiHandDesign.load(path)
    
    print(f"\n  Original design (before rebuild):")
    print(f"  Fold lines: {len(design.fold_lines)}")
    print(f"  Has faces: {len(design.faces)}")
    print(f"  Has joints: {len(design.joints)}")
    print(f"  Pulleys: {len(design.pulleys)}")
    print(f"  Tendons: {len(design.tendons)}")
    
    # Check pulley fold_line attachments
    for pid, p in sorted(design.pulleys.items()):
        fl_id = p.attached_fold_line_id
        if fl_id is not None:
            fl = design.fold_lines.get(fl_id)
            fl_type = fl.fold_type.value if fl else "MISSING!"
            print(f"    Pulley {pid}: attached to fold_line {fl_id} ({fl_type})")
    
    # Check if original design has joints
    orig_has_joints = len(design.joints) > 0
    
    # Get Q matrix from ORIGINAL design
    orig_joints, orig_jid_to_idx = get_joint_list(design)
    print(f"\n  Original joints: {len(orig_joints)}")
    print(f"  Original jid_to_idx: {sorted(orig_jid_to_idx.items())}")
    
    Q_tauM_orig, Q_s_orig, Q_sdot_orig = compute_Q_matrix(design)
    print(f"  Q_tauM (original): {Q_tauM_orig}")
    
    # ========================
    # 2. Build topology and inspect modified design
    # ========================
    print(f"\n  Rebuilding topology...")
    try:
        design.build_topology()
        print(f"  After rebuild:")
        print(f"  Fold lines: {len(design.fold_lines)}")
        print(f"  Faces: {len(design.faces)}")
        print(f"  Joints: {len(design.joints)}")
        
        # Check new fold_line IDs
        new_fold_ids = sorted(design.fold_lines.keys())
        print(f"  New fold_line IDs: {new_fold_ids}")
        
        # Check pulley fold_line attachments after rebuild
        print(f"  Pulley fold_line attachments (after rebuild):")
        for pid, p in sorted(design.pulleys.items()):
            fl_id = p.attached_fold_line_id
            if fl_id is not None:
                fl = design.fold_lines.get(fl_id)
                if fl:
                    print(f"    Pulley {pid}: attached to fold_line {fl_id} -> OK")
                else:
                    print(f"    ⚠ Pulley {pid}: attached to fold_line {fl_id} -> MISSING after rebuild!")
        
        # Get Q matrix from REBUILT design
        new_joints, new_jid_to_idx = get_joint_list(design)
        print(f"\n  Rebuilt joints: {len(new_joints)}")
        print(f"  Rebuilt jid_to_idx: {sorted(new_jid_to_idx.items())}")
        
        Q_tauM_new, Q_s_new, Q_sdot_new = compute_Q_matrix(design)
        print(f"  Q_tauM (rebuilt): {Q_tauM_new}")
        
        # Compare
        print(f"\n  Comparison:")
        print(f"  Q_tauM original: {Q_tauM_orig}")
        print(f"  Q_tauM rebuilt:  {Q_tauM_new}")
        print(f"  Q_s original:    {Q_s_orig}")
        print(f"  Q_s rebuilt:     {Q_s_new}")
        
        if np.allclose(Q_tauM_orig, Q_tauM_new):
            print(f"  ✓ Q_tauM matches")
        else:
            print(f"  ✗ Q_tauM DIFFERS! (rebuilt design has DIFFERENT transmission matrix)")
            print(f"    This means the simulator uses a different Q matrix from what's printed!")
        
        # ========================
        # 3. Check what HandSimulator actually uses
        # ========================
        print(f"\n  Creating HandSimulator (with rebuilt design)...")
        cfg = SimulationConfig(dt=1e-4, t_end=0.1, phase=1, verbose=0)
        sim = HandSimulator(design, cfg)
        
        print(f"  Simulator n_joints: {sim.rbs.n_joints}")
        print(f"  Simulator.Q_tauM: {sim.Q_tauM}")
        print(f"  Simulator.Q_s: {sim.Q_s}")
        print(f"  Simulator.Q_sdot: {sim.Q_sdot}")
        
        if np.allclose(sim.Q_tauM, Q_tauM_new):
            print(f"  ✓ Simulator Q matches rebuilt Q")
        else:
            print(f"  ✗ Simulator Q DIFFERS from rebuilt Q!")
            print(f"    (This shouldn't happen)")
        
        if np.allclose(sim.Q_tauM, Q_tauM_orig):
            print(f"  ✓ Simulator Q matches ORIGINAL Q (pre-rebuild)")
        else:
            print(f"  ✗ Simulator Q differs from original Q")
            print(f"    (Expected - design was modified by rebuild)")
        
        # ========================
        # 4. Check URDF joint mapping
        # ========================
        print(f"\n  Checking URDF joint mapping:")
        from src.models.origami_to_urdf import build_urdf_from_design
        urdf_path = f'models/{name}/{name}.urdf'
        if os.path.exists(urdf_path):
            from lxml import etree
            tree = etree.parse(urdf_path)
            root = tree.getroot()
            for i, joint in enumerate(root.findall('.//joint')):
                child = joint.get('name')
                parent = None
                for item in joint:
                    if item.tag == 'parent':
                        parent = item.get('link')
                    if item.tag == 'child':
                        child = item.get('link')
                        # Try to match child face to fold line
                        print(f"    joint_{i}: parent={parent}, child={child}")
        else:
            print(f"    URDF file not found: {urdf_path}")
        
    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()


def main():
    print("="*70)
    print("  DIAGNOSTIC v4: Q Matrix Before/After Topology Rebuild")
    print("="*70)
    
    for name in ALL_OHD:
        diagnose_ohd(name)
    
    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
