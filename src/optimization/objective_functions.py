# src/optimization/objective_functions.py
"""
Objective functions for the computational design framework.

Define the design targets and quantify how well a given design
parameterization achieves them.

Two main target types:
1. DirectionTarget: Match specific synergy directions (Direction 0, 1)
   from S_aug (sigma and sigma_f). Directions are normalized so only
   relative element magnitudes matter.
2. SpeedDependentTarget: Match different synergy behaviors at slow vs fast
   speeds using the dynamic synergy model.
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np


def normalize_direction(v: np.ndarray) -> np.ndarray:
    """
    Normalize a direction vector to unit length.
    This makes the objective focus on relative magnitudes rather than
    absolute scaling.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v
    return v / norm


def sign_consistency(a: np.ndarray, b: np.ndarray) -> float:
    """
    Penalty for sign mismatches between two vectors.
    Returns fraction of elements with same sign.
    """
    a_sign = np.sign(a)
    b_sign = np.sign(b)
    return np.mean(a_sign == b_sign)


class ObjectiveFunction:
    """
    Base class for design objective functions.
    
    The objective is formulated as a minimization problem.
    Lower values = better design.
    """
    
    def __init__(self, name: str = "objective"):
        self.name = name
        self._weights: Dict[str, float] = {}
    
    def evaluate(self, eval_result: dict) -> float:
        """
        Evaluate the objective given a design evaluation result.
        Must be implemented by subclasses.
        
        Returns a scalar cost (lower is better).
        """
        raise NotImplementedError
    
    def evaluate_with_details(self, eval_result: dict) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate and return both total cost and per-component breakdown.
        
        Returns
        -------
        total_cost : float
        details : dict of component_name -> cost_value
        """
        cost = self.evaluate(eval_result)
        return cost, {self.name: cost}
    
    def set_weight(self, component: str, weight: float):
        """Set weight for a named cost component."""
        self._weights[component] = weight


class DirectionTarget(ObjectiveFunction):
    """
    Objective to match specific synergy directions from the S_aug matrix.
    
    Design target: The synergy directions (columns of S_aug) should
    match desired directional patterns.
    
    This supports two modes:
    1. Direction 0 (σ) and Direction 1 (σ_f) targeting:
       Match both columns of S_aug to desired patterns.
    2. Only Direction 0 targeting.
    
    Parameters
    ----------
    target_dir0 : np.ndarray, shape (n_joints,)
        Desired Direction 0 (sigma) pattern. Will be normalized internally.
    target_dir1 : np.ndarray, shape (n_joints,), optional
        Desired Direction 1 (sigma_f) pattern. Will be normalized internally.
    weight_dir0 : float
        Weight for Direction 0 matching.
    weight_dir1 : float
        Weight for Direction 1 matching.
    weight_ortho : float
        Weight for encouraging orthogonality between the two directions.
        Set to 0 to disable.
    """
    
    def __init__(self,
                 target_dir0: np.ndarray,
                 target_dir1: Optional[np.ndarray] = None,
                 weight_dir0: float = 1.0,
                 weight_dir1: float = 1.0,
                 weight_ortho: float = 0.1):
        super().__init__(name="direction_target")
        
        self.target_dir0 = normalize_direction(np.asarray(target_dir0, dtype=float))
        self.n_joints = len(self.target_dir0)
        
        if target_dir1 is not None:
            self.target_dir1 = normalize_direction(np.asarray(target_dir1, dtype=float))
            self.has_dir1 = True
        else:
            self.target_dir1 = None
            self.has_dir1 = False
        
        self.weight_dir0 = weight_dir0
        self.weight_dir1 = weight_dir1
        self.weight_ortho = weight_ortho
    
    def evaluate_with_details(self, eval_result: dict) -> Tuple[float, Dict[str, float]]:
        S = eval_result.get('S_aug')
        if S is None:
            return 1e10, {'error': 1e10}
        
        n_j = S.shape[0]
        details = {}
        total_cost = 0.0
        
        # ---- Direction 0 matching ----
        if S.shape[1] >= 1:
            actual_dir0 = normalize_direction(S[:, 0])
            
            # Cosine similarity as cost (1 - cos = 0 if aligned, 1 if orthogonal, 2 if opposite)
            cos_sim0 = np.dot(actual_dir0, self.target_dir0)
            # Clamp to avoid numerical issues
            cos_sim0 = np.clip(cos_sim0, -1.0, 1.0)
            # Cost: 0 when aligned, 1 when orthogonal, 2 when opposite
            cost_dir0 = 1.0 - cos_sim0
            
            # Sign penalty: add extra cost if sign pattern is wrong
            sign_cost0 = 1.0 - sign_consistency(actual_dir0, self.target_dir0)
            
            dir0_cost = self.weight_dir0 * (cost_dir0 + 0.5 * sign_cost0)
            details['dir0_alignment'] = cost_dir0
            details['dir0_sign'] = sign_cost0
            details['dir0_cost'] = dir0_cost
            total_cost += dir0_cost
        else:
            total_cost += self.weight_dir0 * 1.0
        
        # ---- Direction 1 matching ----
        if self.has_dir1 and S.shape[1] >= 2:
            actual_dir1 = normalize_direction(S[:, 1])
            
            cos_sim1 = np.dot(actual_dir1, self.target_dir1)
            cos_sim1 = np.clip(cos_sim1, -1.0, 1.0)
            cost_dir1 = 1.0 - cos_sim1
            sign_cost1 = 1.0 - sign_consistency(actual_dir1, self.target_dir1)
            
            dir1_cost = self.weight_dir1 * (cost_dir1 + 0.5 * sign_cost1)
            details['dir1_alignment'] = cost_dir1
            details['dir1_sign'] = sign_cost1
            details['dir1_cost'] = dir1_cost
            total_cost += dir1_cost
        
        # ---- Orthogonality bonus ----
        if self.weight_ortho > 0 and S.shape[1] >= 2:
            d0 = normalize_direction(S[:, 0])
            d1 = normalize_direction(S[:, 1])
            ortho_cos = np.abs(np.dot(d0, d1))
            # Penalize if directions are too similar or too opposite
            ortho_cost = ortho_cos  # 0 = orthogonal (good), 1 = colinear (bad)
            details['orthogonality'] = ortho_cost
            total_cost += self.weight_ortho * ortho_cost
        
        # ---- Degeneracy penalty ----
        # If the second direction is near-zero (rank deficient), penalize
        if self.has_dir1 and S.shape[1] >= 2:
            dir1_norm = np.linalg.norm(S[:, 1])
            if dir1_norm < 0.01:
                details['dir1_degenerate'] = 1.0
                total_cost += self.weight_dir1 * 2.0
        
        return total_cost, details
    
    def evaluate(self, eval_result: dict) -> float:
        cost, _ = self.evaluate_with_details(eval_result)
        return cost


class SpeedDependentTarget(ObjectiveFunction):
    """
    Objective to achieve different synergy behaviors at slow vs fast speeds.
    
    Design target: The synergy directions should differ between slow (S_s)
    and fast (S_f) regimes, with specific patterns.
    
    This supports two sub-modes:
    Mode A: Target specific slow and fast direction patterns
    Mode B: Target a specific "direction change" (S_f - S_s) pattern
    
    Parameters
    ----------
    target_slow : np.ndarray, optional
        Desired slow synergy direction (will be normalized).
    target_fast : np.ndarray, optional
        Desired fast synergy direction (will be normalized).
    target_diff : np.ndarray, optional
        Desired difference direction (S_f - S_s) pattern.
    weight_slow : float
    weight_fast : float
    weight_diff : float
        Weight for the difference direction matching.
    weight_separation : float
        Penalty if S_f and S_s are too similar (no speed-dependent effect).
    """
    
    def __init__(self,
                 target_slow: Optional[np.ndarray] = None,
                 target_fast: Optional[np.ndarray] = None,
                 target_diff: Optional[np.ndarray] = None,
                 weight_slow: float = 1.0,
                 weight_fast: float = 1.0,
                 weight_diff: float = 0.5,
                 weight_separation: float = 0.3):
        super().__init__(name="speed_dependent_target")
        
        self.target_slow = normalize_direction(np.asarray(target_slow, dtype=float)) if target_slow is not None else None
        self.target_fast = normalize_direction(np.asarray(target_fast, dtype=float)) if target_fast is not None else None
        self.target_diff = normalize_direction(np.asarray(target_diff, dtype=float)) if target_diff is not None else None
        
        self.weight_slow = weight_slow
        self.weight_fast = weight_fast
        self.weight_diff = weight_diff
        self.weight_separation = weight_separation
    
    def evaluate_with_details(self, eval_result: dict) -> Tuple[float, Dict[str, float]]:
        S_s = eval_result.get('S_s')
        S_f = eval_result.get('S_f')
        
        if S_s is None:
            return 1e10, {'error_no_S_s': 1e10}
        
        details = {}
        total_cost = 0.0
        
        # Direction 0 only (column 0) for both slow and fast
        if S_s.shape[1] >= 1:
            s_slow = normalize_direction(S_s[:, 0])
        else:
            s_slow = normalize_direction(S_s[:, 0]) if S_s.ndim == 1 else np.zeros(S_s.shape[0])
        
        has_fast = S_f is not None and S_f.shape[1] >= 1
        s_fast = normalize_direction(S_f[:, 0]) if has_fast else s_slow
        
        # ---- Slow direction matching ----
        if self.target_slow is not None:
            cos_slow = np.dot(s_slow, self.target_slow)
            cos_slow = np.clip(cos_slow, -1.0, 1.0)
            cost_slow = 1.0 - cos_slow
            sign_slow = 1.0 - sign_consistency(s_slow, self.target_slow)
            slow_cost = self.weight_slow * (cost_slow + 0.5 * sign_slow)
            details['slow_alignment'] = cost_slow
            details['slow_sign'] = sign_slow
            details['slow_cost'] = slow_cost
            total_cost += slow_cost
        
        # ---- Fast direction matching ----
        if self.target_fast is not None and has_fast:
            cos_fast = np.dot(s_fast, self.target_fast)
            cos_fast = np.clip(cos_fast, -1.0, 1.0)
            cost_fast = 1.0 - cos_fast
            sign_fast = 1.0 - sign_consistency(s_fast, self.target_fast)
            fast_cost = self.weight_fast * (cost_fast + 0.5 * sign_fast)
            details['fast_alignment'] = cost_fast
            details['fast_sign'] = sign_fast
            details['fast_cost'] = fast_cost
            total_cost += fast_cost
        
        # ---- Difference direction matching ----
        if self.target_diff is not None and has_fast:
            actual_diff = normalize_direction(S_f - S_s)
            cos_diff = np.dot(actual_diff[:, 0] if actual_diff.ndim == 2 else actual_diff,
                            self.target_diff)
            cos_diff = np.clip(cos_diff, -1.0, 1.0)
            cost_diff = 1.0 - cos_diff
            details['diff_alignment'] = cost_diff
            total_cost += self.weight_diff * cost_diff
        
        # ---- Separation penalty ----
        # If S_f ≈ S_s, the speed-dependent effect is weak
        if has_fast and self.weight_separation > 0:
            separation = np.linalg.norm(S_f - S_s) / max(np.linalg.norm(S_s), 1e-10)
            # Normalize: separation ~ 0 means identical, larger means more different
            # We want separation > some threshold
            sep_cost = np.exp(-2.0 * separation)  # 1 when identical, ~0 when well-separated
            details['separation'] = separation
            details['sep_cost'] = sep_cost
            total_cost += self.weight_separation * sep_cost
        
        # ---- Degeneracy check ----
        if has_fast and np.linalg.norm(S_f) < 0.001:
            details['fast_degenerate'] = 1.0
            total_cost += 5.0
        
        return total_cost, details
    
    def evaluate(self, eval_result: dict) -> float:
        cost, _ = self.evaluate_with_details(eval_result)
        return cost


class CompositeObjective(ObjectiveFunction):
    """
    Combine multiple objective functions with weights.
    
    Parameters
    ----------
    objectives : list of (ObjectiveFunction, weight) tuples
    """
    
    def __init__(self, objectives: List[Tuple[ObjectiveFunction, float]] = None):
        super().__init__(name="composite")
        self.objectives: List[Tuple[ObjectiveFunction, float]] = objectives or []
    
    def add(self, obj: ObjectiveFunction, weight: float = 1.0):
        """Add an objective component with weight."""
        self.objectives.append((obj, weight))
    
    def evaluate_with_details(self, eval_result: dict) -> Tuple[float, Dict[str, float]]:
        total_cost = 0.0
        all_details = {}
        
        for obj, weight in self.objectives:
            cost, details = obj.evaluate_with_details(eval_result)
            total_cost += weight * cost
            for k, v in details.items():
                all_details[f"{obj.name}.{k}"] = v
        
        all_details['total'] = total_cost
        return total_cost, all_details
    
    def evaluate(self, eval_result: dict) -> float:
        cost, _ = self.evaluate_with_details(eval_result)
        return cost
