# src/optimization/optimization_engine.py
"""
Optimization engine: runs search algorithms to find optimal design parameters.

Supports multiple optimization algorithms:
- Differential Evolution (scipy): Good for global search in continuous spaces
- Particle Swarm Optimization (scipy): Alternative global search
- Brute-force grid search: For small parameter spaces
- Random sampling: For exploration and initialization

For combinatorial tendon routing, a separate enumeration-based search is used.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import time
import json


class AlgorithmType(Enum):
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    SHGO = "shgo"                # Simplicial homology global optimization
    DUAL_ANNEALING = "dual_annealing"
    BASIN_HOPPING = "basin_hopping"
    GRID_SEARCH = "grid_search"
    RANDOM_SAMPLING = "random_sampling"


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_x: np.ndarray           # Best parameter vector
    best_cost: float             # Best objective value
    best_params: Dict[str, float]  # Named parameters
    history: List[float]         # Cost history over iterations
    n_evaluations: int
    total_time: float
    algorithm: str
    success: bool
    message: str = ""
    
    # Detailed evaluation result
    eval_result: Optional[dict] = None
    details: Optional[Dict[str, float]] = None
    
    def summary(self) -> str:
        """Print a formatted summary."""
        lines = [
            f"Optimization Result: {self.algorithm}",
            f"  Success: {self.success}",
            f"  Best cost: {self.best_cost:.6f}",
            f"  Evaluations: {self.n_evaluations}",
            f"  Time: {self.total_time:.2f}s",
            f"  Best parameters:",
        ]
        for k, v in sorted(self.best_params.items()):
            lines.append(f"    {k} = {v:.4f}")
        
        if self.details:
            lines.append(f"  Cost breakdown:")
            for k, v in sorted(self.details.items()):
                lines.append(f"    {k} = {v:.6f}")
        
        return "\n".join(lines)


class OptimizationEngine:
    """
    Optimization engine that searches the design space.
    
    Parameters
    ----------
    objective_fn : ObjectiveFunction
        The objective function to minimize.
    evaluator : DesignEvaluator
        The evaluator that maps parameters to synergy matrices.
    param_names : List[str]
        Names of parameters, corresponding to the optimization vector.
    bounds : List[Tuple[float, float]]
        Bounds for each parameter.
    use_dynamic : bool
        Whether to use the dynamic synergy model in evaluation.
    """
    
    def __init__(self,
                 objective_fn,
                 evaluator,
                 param_names: List[str],
                 bounds: List[Tuple[float, float]],
                 use_dynamic: bool = False):
        self.objective_fn = objective_fn
        self.evaluator = evaluator
        self.param_names = param_names
        self.bounds = bounds
        self.use_dynamic = use_dynamic
        self.n_dims = len(param_names)
        
        # Callback tracking
        self._eval_count = 0
        self._history = []
        self._verbose = False
    
    # ------------------------------------------------------------------
    #  Core objective wrapper
    # ------------------------------------------------------------------
    def _objective(self, x: np.ndarray) -> float:
        """Internal objective function for optimization."""
        self._eval_count += 1
        
        param_dict = {name: float(x[i]) for i, name in enumerate(self.param_names)}
        
        eval_result = self.evaluator.evaluate(param_dict, use_dynamic=self.use_dynamic)
        
        cost = self.objective_fn.evaluate(eval_result)
        
        self._history.append(cost)
        
        if self._verbose and self._eval_count % 10 == 0:
            print(f"  Eval {self._eval_count}: cost = {cost:.6f}")
        
        return cost
    
    def _objective_with_details(self, x: np.ndarray) -> Tuple[float, dict]:
        """Return both cost and detailed breakdown."""
        param_dict = {name: float(x[i]) for i, name in enumerate(self.param_names)}
        eval_result = self.evaluator.evaluate(param_dict, use_dynamic=self.use_dynamic)
        cost, details = self.objective_fn.evaluate_with_details(eval_result)
        return cost, details
    
    # ------------------------------------------------------------------
    #  Differential Evolution (recommended default)
    # ------------------------------------------------------------------
    def run_differential_evolution(self,
                                   maxiter: int = 100,
                                   popsize: int = 15,
                                   tol: float = 1e-6,
                                   mutation: float = 0.5,
                                   recombination: float = 0.7,
                                   seed: Optional[int] = None,
                                   verbose: bool = False) -> OptimizationResult:
        """
        Run Differential Evolution global optimization.
        
        Best for: 3-20 continuous design variables with nonlinear objectives.
        """
        from scipy.optimize import differential_evolution
        
        self._eval_count = 0
        self._history = []
        self._verbose = verbose
        
        start_time = time.time()
        
        result = differential_evolution(
            self._objective,
            bounds=self.bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            mutation=mutation,
            recombination=recombination,
            seed=seed,
            callback=self._de_callback if verbose else None,
        )
        
        total_time = time.time() - start_time
        
        best_x = result.x
        best_cost = result.fun
        
        # Get detailed evaluation
        _, details = self._objective_with_details(best_x)
        param_dict = {name: float(best_x[i]) for i, name in enumerate(self.param_names)}
        
        return OptimizationResult(
            best_x=best_x,
            best_cost=best_cost,
            best_params=param_dict,
            history=self._history,
            n_evaluations=self._eval_count,
            total_time=total_time,
            algorithm="differential_evolution",
            success=result.success,
            message=str(result.message),
            details=details,
        )
    
    def _de_callback(self, xk, convergence=None):
        if self._eval_count % 50 == 0:
            print(f"  DE: eval={self._eval_count}, best={self._history[-1]:.6f}, conv={convergence:.4f}")
        return False
    
    # ------------------------------------------------------------------
    #  SHGO (simplicial homology global optimization)
    # ------------------------------------------------------------------
    def run_shgo(self,
                 n: int = 100,
                 iters: int = 1,
                 sampling_method: str = 'sobol',
                 verbose: bool = False) -> OptimizationResult:
        """
        Run SHGO global optimization.
        Good for functions with many local minima.
        """
        from scipy.optimize import shgo
        
        self._eval_count = 0
        self._history = []
        self._verbose = verbose
        
        start_time = time.time()
        
        result = shgo(
            self._objective,
            bounds=self.bounds,
            n=n,
            iters=iters,
            sampling_method=sampling_method,
        )
        
        total_time = time.time() - start_time
        
        best_x = result.x
        best_cost = result.fun
        
        _, details = self._objective_with_details(best_x)
        param_dict = {name: float(best_x[i]) for i, name in enumerate(self.param_names)}
        
        return OptimizationResult(
            best_x=best_x,
            best_cost=best_cost,
            best_params=param_dict,
            history=self._history,
            n_evaluations=self._eval_count,
            total_time=total_time,
            algorithm="shgo",
            success=result.success,
            message=str(getattr(result, 'message', '')),
            details=details,
        )
    
    # ------------------------------------------------------------------
    #  Grid Search
    # ------------------------------------------------------------------
    def run_grid_search(self,
                        n_points_per_dim: int = 5,
                        verbose: bool = False) -> OptimizationResult:
        """
        Brute-force grid search over the parameter space.
        
        Best for: Small number of variables (<= 4) where exhaustive search is feasible.
        Number of evaluations = n_points_per_dim^n_dims.
        """
        self._eval_count = 0
        self._history = []
        self._verbose = verbose
        
        start_time = time.time()
        
        # Generate grid points
        grid_axes = []
        for i, (low, high) in enumerate(self.bounds):
            if n_points_per_dim > 1:
                axis = np.linspace(low, high, n_points_per_dim)
            else:
                axis = np.array([(low + high) / 2.0])
            grid_axes.append(axis)
        
        # Mesh grid
        mesh = np.meshgrid(*grid_axes, indexing='ij')
        dims = [len(axis) for axis in grid_axes]
        
        best_cost = float('inf')
        best_x = None
        best_details = None
        
        # Flatten and iterate
        flat = [m.ravel() for m in mesh]
        n_total = len(flat[0])
        
        for i in range(n_total):
            x = np.array([f[i] for f in flat])
            
            cost, details = self._objective_with_details(x)
            self._eval_count += 1
            self._history.append(cost)
            
            if cost < best_cost:
                best_cost = cost
                best_x = x.copy()
                best_details = details
            
            if verbose and (i + 1) % max(1, n_total // 20) == 0:
                print(f"  Grid: {i+1}/{n_total} evaluated, best={best_cost:.6f}")
        
        total_time = time.time() - start_time
        param_dict = {name: float(best_x[i]) for i, name in enumerate(self.param_names)}
        
        return OptimizationResult(
            best_x=best_x,
            best_cost=best_cost,
            best_params=param_dict,
            history=self._history,
            n_evaluations=self._eval_count,
            total_time=total_time,
            algorithm=f"grid_search_{n_points_per_dim}",
            success=True,
            details=best_details,
        )
    
    # ------------------------------------------------------------------
    #  Random Sampling
    # ------------------------------------------------------------------
    def run_random_sampling(self,
                            n_samples: int = 1000,
                            seed: Optional[int] = None,
                            verbose: bool = False) -> OptimizationResult:
        """
        Random sampling over the parameter space.
        Good for exploration and getting a baseline.
        """
        self._eval_count = 0
        self._history = []
        self._verbose = verbose
        
        rng = np.random.RandomState(seed)
        start_time = time.time()
        
        best_cost = float('inf')
        best_x = None
        best_details = None
        
        for i in range(n_samples):
            x = np.array([
                rng.uniform(low, high) for low, high in self.bounds
            ])
            
            cost, details = self._objective_with_details(x)
            self._eval_count += 1
            self._history.append(cost)
            
            if cost < best_cost:
                best_cost = cost
                best_x = x.copy()
                best_details = details
            
            if verbose and (i + 1) % max(1, n_samples // 10) == 0:
                print(f"  Random: {i+1}/{n_samples}, best={best_cost:.6f}")
        
        total_time = time.time() - start_time
        param_dict = {name: float(best_x[i]) for i, name in enumerate(self.param_names)}
        
        return OptimizationResult(
            best_x=best_x,
            best_cost=best_cost,
            best_params=param_dict,
            history=self._history,
            n_evaluations=self._eval_count,
            total_time=total_time,
            algorithm=f"random_sampling_{n_samples}",
            success=True,
            details=best_details,
        )
    
    # ------------------------------------------------------------------
    #  Run best algorithm automatically
    # ------------------------------------------------------------------
    def run(self,
            algorithm: AlgorithmType = AlgorithmType.DIFFERENTIAL_EVOLUTION,
            **kwargs) -> OptimizationResult:
        """
        Run the optimization with the specified algorithm.
        
        Parameters
        ----------
        algorithm : AlgorithmType
        **kwargs : passed to the specific algorithm.
        
        Returns
        -------
        result : OptimizationResult
        """
        if algorithm == AlgorithmType.DIFFERENTIAL_EVOLUTION:
            return self.run_differential_evolution(**kwargs)
        elif algorithm == AlgorithmType.SHGO:
            return self.run_shgo(**kwargs)
        elif algorithm == AlgorithmType.GRID_SEARCH:
            return self.run_grid_search(**kwargs)
        elif algorithm == AlgorithmType.RANDOM_SAMPLING:
            return self.run_random_sampling(**kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # ------------------------------------------------------------------
    #  Two-phase: random exploration + refinement
    # ------------------------------------------------------------------
    def run_two_phase(self,
                      n_explore: int = 500,
                      de_kwargs: Optional[dict] = None,
                      verbose: bool = False) -> OptimizationResult:
        """
        Two-phase optimization:
        Phase 1: Random sampling for exploration
        Phase 2: Differential Evolution refinement around best point
        
        This is recommended for most real design problems.
        """
        self._verbose = verbose
        
        if verbose:
            print("Phase 1: Random exploration...")
        
        result1 = self.run_random_sampling(n_samples=n_explore, verbose=verbose)
        
        if verbose:
            print(f"  Phase 1 best: {result1.best_cost:.6f}")
            print("Phase 2: DE refinement...")
        
        # Narrow bounds around best point
        best_x = result1.best_x
        refined_bounds = []
        for i, (low, high) in enumerate(self.bounds):
            span = high - low
            # Narrow to 30% of original span around best point
            new_low = max(low, best_x[i] - span * 0.3)
            new_high = min(high, best_x[i] + span * 0.3)
            refined_bounds.append((new_low, new_high))
        
        # Temporarily override bounds for phase 2
        original_bounds = self.bounds
        self.bounds = refined_bounds
        
        de_kwargs = de_kwargs or {}
        de_kwargs.setdefault('popsize', 20)
        de_kwargs.setdefault('maxiter', 80)
        de_kwargs.setdefault('verbose', verbose)
        
        result2 = self.run_differential_evolution(**de_kwargs)
        
        self.bounds = original_bounds
        
        # Combine results
        if result2.best_cost < result1.best_cost:
            result2.history = result1.history + result2.history
            result2.n_evaluations = result1.n_evaluations + result2.n_evaluations
            result2.total_time = result1.total_time + result2.total_time
            return result2
        else:
            result1.algorithm = f"two_phase_de"
            result1.n_evaluations = result1.n_evaluations + result2.n_evaluations
            result1.total_time = result1.total_time + result2.total_time
            return result1
