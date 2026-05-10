# src/optimization/design_evaluator.py
"""
Design evaluator: maps a parameter vector to an OrigamiHandDesign,
then computes the synergy model and extracts relevant matrices.

This is the core bridge between the optimization parameter space and
the physical/model space.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import copy

from src.models.origami_design import (
    OrigamiHandDesign, FoldLine, FoldType, Point2D,
    Hole, Pulley, Tendon, Damper,
    HOLE_ID_OFFSET, is_hole_id,
)
from src.models.transmission_builder import (
    build_synergy_model, build_dynamic_synergy_model, get_joint_list
)
from src.models.hole_transmission import (
    find_hole_pairs, compute_hole_equivalent_R_row
)
from src.synergy.base_adaptive import AdaptiveSynergyModel
from src.synergy.augmented_adaptive import AugmentedAdaptiveSynergyModel
from src.synergy.dynamic_synergy import DynamicSynergyModel


class DesignEvaluator:
    """
    Evaluates a design by:
    1. Taking a base OrigamiHandDesign
    2. Applying design parameter modifications
    3. Computing the synergy model and returning the synergy matrices
    
    Parameters
    ----------
    base_design : OrigamiHandDesign
        The original hand design with geometry (fold lines, outlines,
        hole positions, pulleys, dampers, tendons).
    joint_fold_map : Dict[int, int]
        Mapping from fold_line_id -> joint_index.
    """
    
    def __init__(self, base_design: OrigamiHandDesign,
                 joint_fold_map: Optional[Dict[int, int]] = None):
        self.base_design = base_design
        self._joints, self._jid_to_idx = get_joint_list(base_design)
        self.n_joints = len(self._joints)
        self.n_holes = len(base_design.holes)
        self.n_dampers = len(base_design.dampers)
        
        # Build a list of hole IDs sorted for consistent indexing
        self.hole_ids = sorted(base_design.holes.keys())
        self.damper_ids = sorted(base_design.dampers.keys())
    
    # ------------------------------------------------------------------
    #  Apply design parameters to create a modified design
    # ------------------------------------------------------------------
    def apply_parameters(self, param_dict: Dict[str, float]) -> OrigamiHandDesign:
        """
        Create a new design by applying parameters to a deep copy of the base.
        
        Parameters
        ----------
        param_dict : dict
            Named parameter values (e.g., {'fold_stiffness_0': 2.5, ...})
        
        Returns
        -------
        design : OrigamiHandDesign
            Modified design ready for evaluation.
        """
        design = copy.deepcopy(self.base_design)
        
        # ---- Apply fold stiffness modifications ----
        for i, joint in enumerate(self._joints):
            key = f'fold_stiffness_{i}'
            if key in param_dict:
                fid = joint.fold_line_id
                if fid in design.fold_lines:
                    design.fold_lines[fid].stiffness = param_dict[key]
        
        # ---- Apply hole parameter modifications ----
        for idx, hid in enumerate(self.hole_ids):
            if hid not in design.holes:
                continue
            hole = design.holes[hid]
            
            # Plate offset (h)
            key_h = f'hole_offset_{idx}'
            if key_h in param_dict:
                hole.plate_offset = param_dict[key_h]
            
            # Radius
            key_r = f'hole_radius_{idx}'
            if key_r in param_dict:
                hole.radius = param_dict[key_r]
            
            # Distance to crease (d): modify position relative to the crease
            key_d = f'hole_distance_{idx}'
            if key_d in param_dict and hole.attached_fold_line_id is not None:
                self._set_hole_distance(design, hid, param_dict[key_d])
            
            # Friction coefficient
            key_mu = f'hole_friction_{idx}'
            if key_mu in param_dict:
                hole.friction_coefficient = param_dict[key_mu]
        
        # ---- Apply damping modifications ----
        for idx, did in enumerate(self.damper_ids):
            key = f'damping_{idx}'
            if key in param_dict and did in design.dampers:
                design.dampers[did].damping_coefficient = param_dict[key]
        
        return design
    
    def _set_hole_distance(self, design: OrigamiHandDesign,
                           hole_id: int, new_d: float):
        """
        Adjust hole position along the perpendicular direction to its
        attached fold line to achieve a desired distance 'd' from the crease.
        
        The hole is moved along the normal of the fold line while keeping
        its projection position on the fold line constant.
        """
        hole = design.holes[hole_id]
        fl_id = hole.attached_fold_line_id
        if fl_id is None or fl_id not in design.fold_lines:
            return
        
        fold = design.fold_lines[fl_id]
        start = np.array([fold.start.x, fold.start.y])
        end = np.array([fold.end.x, fold.end.y])
        direction = end - start
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            return
        direction = direction / dir_norm
        
        # Normal (perpendicular)
        normal = np.array([-direction[1], direction[0]])
        
        # Current hole position relative to fold line start
        pos = np.array([hole.position.x, hole.position.y])
        v = pos - start
        
        # Current signed distance from fold line
        current_dist = np.dot(v, normal)
        sign = 1.0 if current_dist >= 0 else -1.0
        
        # Projection along fold line
        t = np.dot(v, direction)
        
        # New position: keep t same, set distance to new_d
        new_pos = start + t * direction + sign * new_d * normal
        
        hole.position = Point2D(float(new_pos[0]), float(new_pos[1]))
    
    # ------------------------------------------------------------------
    #  Evaluate: apply parameters, build synergy model, return matrices
    # ------------------------------------------------------------------
    def evaluate(self, param_dict: Dict[str, float],
                 use_dynamic: bool = False) -> dict:
        """
        Full evaluation: apply parameters, compute synergy model.
        
        Parameters
        ----------
        param_dict : dict
            Design parameter values.
        use_dynamic : bool
            If True, also compute dynamic synergy (S_s, S_f).
        
        Returns
        -------
        result : dict with keys:
            'design': modified OrigamiHandDesign
            'R': transmission matrix
            'Rf': friction transmission matrix
            'S_s': slow synergy matrix (or base S if no dynamic)
            'S_f': fast synergy matrix (or None if no dynamic)
            'S_aug': augmented synergy matrix (vstack([R, Rf]) version)
            'E_vec': joint stiffness vector
            'T_mat': damper transmission matrix
            'C_diag': damping coefficient vector
            'joint_names': list of joint names
        """
        design = self.apply_parameters(param_dict)
        
        # Build base (augmented) synergy model
        try:
            model, info = build_synergy_model(design)
            # model is AugmentedAdaptiveSynergyModel
            # Extract R and Rf internally:
            joints, jid_to_idx = get_joint_list(design)
            n = len(joints)
            from src.models.transmission_builder import compute_R, compute_Rf
            R_mat = compute_R(design)
            Rf_mat = compute_Rf(design)
            E_vec = np.array([
                design.fold_lines[j.fold_line_id].stiffness
                for j in joints
            ])
            
            result = {
                'design': design,
                'R': R_mat,
                'Rf': Rf_mat,
                'S_aug': model.S_aug,
                'E_vec': E_vec,
                'joint_names': list(info.get('joint_names', [])),
            }
            
            if use_dynamic:
                dyn_model, dyn_info = build_dynamic_synergy_model(
                    design, use_augmented=True
                )
                result['S_s'] = dyn_model.S_s
                result['S_f'] = dyn_model.S_f
                result['T_mat'] = dyn_model.T
                result['C_diag'] = np.diag(dyn_model.C)
                result['has_dynamic'] = dyn_model.n_d > 0
            else:
                # Non-dynamic: compute S_s from base
                base = AdaptiveSynergyModel(n, R_mat, E_vec)
                result['S_s'] = base.S
                result['S_f'] = None
                result['T_mat'] = None
                result['C_diag'] = None
                result['has_dynamic'] = False
            
            return result
        
        except Exception as e:
            # Return a minimal result with error info
            return {
                'error': str(e),
                'design': design,
                'R': None, 'Rf': None,
                'S_s': None, 'S_f': None, 'S_aug': None,
                'E_vec': None,
                'joint_names': [],
            }
    
    # ------------------------------------------------------------------
    #  Convenience: evaluate from vector
    # ------------------------------------------------------------------
    def evaluate_from_vector(self, x: np.ndarray,
                             param_names: List[str],
                             use_dynamic: bool = False) -> dict:
        """Evaluate from a flat optimization vector and name list."""
        param_dict = {name: float(x[i]) for i, name in enumerate(param_names)}
        return self.evaluate(param_dict, use_dynamic=use_dynamic)
