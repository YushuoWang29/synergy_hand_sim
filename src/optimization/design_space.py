# src/optimization/design_space.py
"""
Design space definition for origami hand optimization.

Defines what variables can be edited, their types, ranges, and
how they map to an actual OrigamiHandDesign.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np


class VariableType(Enum):
    """Types of design variables for differentiation."""
    SCALAR = "scalar"           # Single float value
    CONTINUOUS = "continuous"   # Continuous range
    INTEGER = "integer"         # Integer range
    CATEGORICAL = "categorical" # Discrete choices
    PERMUTATION = "permutation" # Ordering of items (tendon path)


@dataclass
class DesignVariable:
    """
    A single design variable with its bounds/choices.
    
    Parameters
    ----------
    name : str
        Human-readable name, e.g. 'fold_stiffness_0'
    var_type : VariableType
    low : float, optional
        Lower bound (for continuous/integer)
    high : float, optional
        Upper bound (for continuous/integer)
    categories : list, optional
        Possible values (for categorical)
    default : Any, optional
        Default value
    description : str, optional
        Explanation of what this variable controls
    """
    name: str
    var_type: VariableType
    low: Optional[float] = None
    high: Optional[float] = None
    categories: Optional[List[Any]] = None
    default: Optional[Any] = None
    description: str = ""

    def validate(self, value: float) -> bool:
        """Check if a value is within bounds."""
        if self.var_type == VariableType.CONTINUOUS:
            return self.low <= value <= self.high
        elif self.var_type == VariableType.INTEGER:
            return self.low <= value <= self.high and float(value).is_integer()
        elif self.var_type == VariableType.CATEGORICAL:
            return value in self.categories
        elif self.var_type == VariableType.SCALAR:
            return True  # fixed, no validation needed
        return True

    def clip(self, value: float) -> float:
        """Clip a value to the valid range."""
        if self.var_type in (VariableType.CONTINUOUS, VariableType.INTEGER):
            return float(np.clip(value, self.low, self.high))
        return value

    @property
    def dimension(self) -> int:
        """Number of real-valued parameters needed to encode this variable."""
        if self.var_type == VariableType.PERMUTATION:
            return 0  # handled separately
        return 1


class DesignSpace:
    """
    Collection of design variables that define the search space.
    
    Grouped by category:
    - fold_stiffness: stiffness of each fold line
    - hole_params: radius, plate_offset, position offset for each hole
    - tendon_count: number of tendons
    - tendon_path: routing path for each tendon (permutation problem)
    - damping: damping coefficients
    
    Parameters
    ----------
    n_joints : int
        Number of joints in the hand
    n_holes : int
        Number of available hole positions
    n_tendons_max : int
        Maximum number of tendons to consider
    """
    
    def __init__(self, n_joints: int, n_holes: int, n_tendons_max: int = 3):
        self.n_joints = n_joints
        self.n_holes = n_holes
        self.n_tendons_max = n_tendons_max
        
        self.variables: Dict[str, DesignVariable] = {}
        self._groups: Dict[str, List[str]] = {
            'fold_stiffness': [],
            'hole_params': [],
            'tendon_count': [],
            'tendon_path': [],
            'damping': [],
        }
    
    def add_variable(self, var: DesignVariable, group: str = 'other'):
        """Add a design variable to the space."""
        if var.name in self.variables:
            raise ValueError(f"Variable '{var.name}' already exists")
        self.variables[var.name] = var
        if group in self._groups:
            self._groups[group].append(var.name)
        else:
            self._groups.setdefault(group, []).append(var.name)
    
    # ------------------------------------------------------------------
    #  Convenience builders
    # ------------------------------------------------------------------
    def add_fold_stiffness(self, fold_idx: int,
                           low: float = 0.1, high: float = 10.0,
                           default: float = 1.0):
        """Add fold line stiffness as editable variable."""
        self.add_variable(DesignVariable(
            name=f'fold_stiffness_{fold_idx}',
            var_type=VariableType.CONTINUOUS,
            low=low, high=high,
            default=default,
            description=f'Stiffness of fold line {fold_idx}'
        ), group='fold_stiffness')
    
    def add_hole_radius(self, hole_idx: int,
                        low: float = 0.5, high: float = 5.0,
                        default: float = 1.5):
        """Add hole radius as editable variable."""
        self.add_variable(DesignVariable(
            name=f'hole_radius_{hole_idx}',
            var_type=VariableType.CONTINUOUS,
            low=low, high=high,
            default=default,
            description=f'Radius of hole {hole_idx}'
        ), group='hole_params')
    
    def add_hole_plate_offset(self, hole_idx: int,
                              low: float = 0.5, high: float = 8.0,
                              default: float = 3.0):
        """Add hole plate offset (h) as editable variable."""
        self.add_variable(DesignVariable(
            name=f'hole_offset_{hole_idx}',
            var_type=VariableType.CONTINUOUS,
            low=low, high=high,
            default=default,
            description=f'Plate offset (h) of hole {hole_idx}'
        ), group='hole_params')
    
    def add_hole_distance_to_crease(self, hole_idx: int,
                                    low: float = 2.0, high: float = 15.0,
                                    default: float = 5.0):
        """Add hole distance to crease (d) as editable variable."""
        self.add_variable(DesignVariable(
            name=f'hole_distance_{hole_idx}',
            var_type=VariableType.CONTINUOUS,
            low=low, high=high,
            default=default,
            description=f'Distance from hole {hole_idx} to crease'
        ), group='hole_params')
    
    def add_tendon_count(self, low: int = 1, high: int = 3, default: int = 1):
        """Add number of tendons as editable integer variable."""
        self.add_variable(DesignVariable(
            name='tendon_count',
            var_type=VariableType.INTEGER,
            low=low, high=high,
            default=default,
            description='Number of tendons'
        ), group='tendon_count')
    
    def add_damping(self, damper_idx: int,
                    low: float = 0.0, high: float = 20.0,
                    default: float = 1.0):
        """Add damping coefficient as editable variable."""
        self.add_variable(DesignVariable(
            name=f'damping_{damper_idx}',
            var_type=VariableType.CONTINUOUS,
            low=low, high=high,
            default=default,
            description=f'Damping coefficient for damper {damper_idx}'
        ), group='damping')
    
    # ------------------------------------------------------------------
    #  Conversion to/from optimization vectors
    # ------------------------------------------------------------------
    @property
    def continuous_variable_names(self) -> List[str]:
        """Names of all continuous/integer variables."""
        return [name for name, v in self.variables.items()
                if v.var_type in (VariableType.CONTINUOUS, VariableType.INTEGER)]
    
    @property
    def continuous_bounds(self) -> List[Tuple[float, float]]:
        """Bounds for continuous optimization variables."""
        bounds = []
        for name in self.continuous_variable_names:
            v = self.variables[name]
            bounds.append((v.low, v.high))
        return bounds
    
    def get_default_vector(self) -> np.ndarray:
        """Get default parameter vector."""
        vec = []
        for name in self.continuous_variable_names:
            v = self.variables[name]
            vec.append(v.default if v.default is not None else (v.low + v.high) / 2.0)
        return np.array(vec)
    
    def vector_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        """Convert optimization vector to named dict."""
        names = self.continuous_variable_names
        return {name: float(x[i]) for i, name in enumerate(names)}
    
    def summary(self) -> str:
        """Print a summary of the design space."""
        lines = [f"Design Space: {len(self.variables)} variables",
                 f"  Joints: {self.n_joints}, Holes: {self.n_holes}"]
        for group, var_names in self._groups.items():
            if not var_names:
                continue
            lines.append(f"  [{group}]")
            for vn in var_names:
                v = self.variables[vn]
                bounds = f"[{v.low:.2f}, {v.high:.2f}]" if v.low is not None else ""
                lines.append(f"    {vn}: {v.var_type.value} {bounds}")
        return "\n".join(lines)
