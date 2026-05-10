# src/optimization/__init__.py
"""
Computational design optimization framework for origami hand synergies.

Modules:
    design_space: Define editable design variables and their constraints
    design_evaluator: Map parameters -> OrigamiHandDesign -> synergy matrices
    objective_functions: Define design targets (direction matching, speed-dependent)
    optimization_engine: Run search algorithms (DE, SHGO, grid, random)
    tendon_routing: Combinatorial tendon path optimization
"""

from .design_space import DesignSpace, DesignVariable, VariableType
from .design_evaluator import DesignEvaluator
from .objective_functions import (
    ObjectiveFunction, DirectionTarget, SpeedDependentTarget,
    CompositeObjective, normalize_direction, sign_consistency
)
from .optimization_engine import (
    OptimizationEngine, OptimizationResult, AlgorithmType
)

__all__ = [
    'DesignSpace', 'DesignVariable', 'VariableType',
    'DesignEvaluator',
    'ObjectiveFunction', 'DirectionTarget', 'SpeedDependentTarget',
    'CompositeObjective', 'normalize_direction', 'sign_consistency',
    'OptimizationEngine', 'OptimizationResult', 'AlgorithmType',
]
