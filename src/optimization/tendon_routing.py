# src/optimization/tendon_routing.py
"""
Tendon routing optimization: find optimal hole sequences for tendon paths.

Given a set of holes on the origami hand, each with an associated face
and fold line, find the best tendon routing path(s) that achieve the
desired synergy directions.

Problem formulation:
- Each tendon starts from Actuator A (-1), passes through a sequence of holes,
  and ends at Actuator B (-2).
- Each hole can be used at most once.
- Each hole on a face can only be used by one tendon.
- The path must be physically valid (no crossing between faces, etc.)

We formulate this as a graph problem: the holes are nodes, and edges
represent valid transitions between holes on adjacent faces.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from itertools import permutations, combinations
import numpy as np

from src.models.origami_design import (
    OrigamiHandDesign, Hole, FoldLine, FoldType, Point2D,
    Tendon, is_hole_id
)
from src.models.transmission_builder import (
    compute_R, compute_Rf, get_joint_list
)
from src.models.hole_transmission import (
    find_hole_pairs, compute_hole_equivalent_R_row
)
from src.synergy.base_adaptive import AdaptiveSynergyModel


@dataclass
class HoleGraph:
    """
    Graph representation of holes and their connectivity.
    
    Nodes: holes (by ID)
    Edges: valid transitions between holes that are on adjacent faces
           sharing a fold line.
    
    This is used to enumerate or optimize tendon routing paths.
    """
    hole_ids: List[int]
    face_of_hole: Dict[int, int]        # hole_id -> face_id
    fold_of_hole: Dict[int, int]        # hole_id -> attached_fold_line_id
    position_of_hole: Dict[int, Point2D]  # hole_id -> position
    
    # Adjacency: hole_id -> list of (neighbor_hole_id, shared_fold_line_id)
    adjacency: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    
    @classmethod
    def from_design(cls, design: OrigamiHandDesign) -> 'HoleGraph':
        """Build a hole graph from a design."""
        hole_ids = sorted(design.holes.keys())
        face_map = {}
        fold_map = {}
        pos_map = {}
        
        for hid, hole in design.holes.items():
            face_map[hid] = hole.face_id
            fold_map[hid] = hole.attached_fold_line_id
            pos_map[hid] = hole.position
        
        graph = cls(
            hole_ids=hole_ids,
            face_of_hole=face_map,
            fold_of_hole=fold_map,
            position_of_hole=pos_map,
        )
        
        # Build adjacency: two holes are adjacent if they share a fold line
        # OR if they are on adjacent faces (connected via a joint)
        # For simplicity: holes on the same fold line (one on each side) are adjacent.
        
        # Group holes by fold line
        holes_by_fold: Dict[int, List[int]] = {}
        for hid, fid in fold_map.items():
            if fid is not None:
                holes_by_fold.setdefault(fid, []).append(hid)
        
        # Holes on opposite sides of the same fold line are adjacent
        for fid, hids in holes_by_fold.items():
            if len(hids) >= 2:
                # All pairs of holes on this fold line are adjacent
                for i in range(len(hids)):
                    for j in range(i + 1, len(hids)):
                        graph._add_edge(hids[i], hids[j], fid)
        
        # Also connect holes that are on the same face (sequential ordering)
        holes_by_face: Dict[int, List[int]] = {}
        for hid, fid in face_map.items():
            if fid is not None:
                holes_by_face.setdefault(fid, []).append(hid)
        
        for fid, hids in holes_by_face.items():
            if len(hids) >= 2:
                # Holes on the same face can be connected if there's a path
                # along the face surface. Connect all pairs for now.
                for i in range(len(hids)):
                    for j in range(i + 1, len(hids)):
                        graph._add_edge(hids[i], hids[j], fid)
        
        return graph
    
    def _add_edge(self, h1: int, h2: int, via_id: int):
        """Add bidirectional edge."""
        self.adjacency.setdefault(h1, []).append((h2, via_id))
        self.adjacency.setdefault(h2, []).append((h1, via_id))
    
    @property
    def n_holes(self) -> int:
        return len(self.hole_ids)
    
    def get_route_up_to_length(self, max_len: int) -> List[List[int]]:
        """
        Enumerate all possible simple paths (no repeated holes) up to
        max_len holes. Includes start/end actuators.
        
        Returns
        -------
        routes : list of [actuator_A, hole_1, ..., hole_k, actuator_B]
        """
        all_routes = []
        
        # We need to consider all permutations of hole subsets
        # For small n_holes (<= 8), brute force enumeration is feasible
        if self.n_holes > 8:
            # Fall back to random sampling or heuristic
            return self._sample_routes(max_len, n_samples=200)
        
        # For each possible route length k (1 to max_len)
        for k in range(1, min(max_len, self.n_holes) + 1):
            # For each subset of k holes
            for subset in combinations(self.hole_ids, k):
                # For each ordering of the subset
                for perm in permutations(subset):
                    route = [-1] + list(perm) + [-2]  # A -> holes -> B
                    if self._is_valid_route(route):
                        all_routes.append(route)
        
        return all_routes
    
    def _sample_routes(self, max_len: int, n_samples: int = 200) -> List[List[int]]:
        """Sample random valid routes for larger problems."""
        routes = []
        import random
        attempts = 0
        while len(routes) < n_samples and attempts < n_samples * 10:
            k = random.randint(1, min(max_len, self.n_holes))
            subset = random.sample(self.hole_ids, k)
            route = [-1] + subset + [-2]
            if self._is_valid_route(route) and route not in routes:
                routes.append(route)
            attempts += 1
        return routes
    
    def _is_valid_route(self, route: List[int]) -> bool:
        """
        Check if a route is physically valid.
        Rule: consecutive holes must be adjacent in the graph.
        """
        # Extract hole-only part
        holes = [e for e in route if is_hole_id(e)]
        if len(holes) <= 1:
            return True
        
        for i in range(len(holes) - 1):
            h1, h2 = holes[i], holes[i + 1]
            neighbors = {n for n, _ in self.adjacency.get(h1, [])}
            if h2 not in neighbors:
                return False
        
        return True


class TendonRouter:
    """
    Handles combinatorial optimization of tendon routing paths.
    
    Given a set of possible hole paths, find the combination that
    best achieves the target synergy directions.
    
    Parameters
    ----------
    design : OrigamiHandDesign
        Base design with holes placed.
    n_tendons_range : Tuple[int, int]
        (min_tendons, max_tendons) to consider.
    """
    
    def __init__(self, design: OrigamiHandDesign,
                 n_tendons_range: Tuple[int, int] = (1, 3)):
        self.design = design
        self.n_tendons_min, self.n_tendons_max = n_tendons_range
        self.hole_graph = HoleGraph.from_design(design)
        
        # Pre-enumerate possible routes
        max_holes_per_tendon = max(2, self.hole_graph.n_holes // 2)
        self.possible_routes = self.hole_graph.get_route_up_to_length(
            max_holes_per_tendon
        )
    
    def enumerate_tendon_configs(self) -> List[Dict[int, List[int]]]:
        """
        Enumerate all possible multi-tendon configurations.
        
        Returns
        -------
        configs : list of {tendon_id: [pulley_sequence]}
            Each config is a set of tendons with their paths.
        """
        configs = []
        
        for n_tendons in range(self.n_tendons_min, self.n_tendons_max + 1):
            if n_tendons == 1:
                # Single tendon: each possible route is a config
                for route in self.possible_routes:
                    configs.append({0: route})
            else:
                # Multiple tendons: routes must use disjoint hole sets
                configs.extend(self._enumerate_multi_tendon(n_tendons))
        
        return configs
    
    def _enumerate_multi_tendon(self, n: int) -> List[Dict[int, List[int]]]:
        """Enumerate configurations with n tendons (disjoint hole sets)."""
        configs = []
        
        # For each combination of n routes where holes don't overlap
        for route_combo in combinations(self.possible_routes, n):
            # Check that hole sets are disjoint
            all_holes = []
            for route in route_combo:
                holes = {e for e in route if is_hole_id(e)}
                all_holes.append(holes)
            
            # Check pairwise disjoint
            disjoint = True
            for i in range(len(all_holes)):
                for j in range(i + 1, len(all_holes)):
                    if all_holes[i] & all_holes[j]:
                        disjoint = False
                        break
                if not disjoint:
                    break
            
            if disjoint:
                config = {i: list(route_combo[i]) for i in range(n)}
                configs.append(config)
        
        return configs
    
    def evaluate_config(self, tendon_config: Dict[int, List[int]],
                        joint_stiffness: np.ndarray) -> np.ndarray:
        """
        For a given tendon configuration, compute the R matrix.
        
        Parameters
        ----------
        tendon_config : {tendon_id: pulley_sequence}
        joint_stiffness : shape (n_joints,) 
        
        Returns
        -------
        R : shape (n_tendons, n_joints)
        """
        joints, jid_to_idx = get_joint_list(self.design)
        n = len(joints)
        n_tendons = len(tendon_config)
        R = np.zeros((n_tendons, n))
        
        for tid, sequence in tendon_config.items():
            row = np.zeros(n)
            for eid in sequence:
                if eid < 0:
                    continue  # actuator
                if is_hole_id(eid):
                    hole = self.design.holes.get(eid)
                    if hole and hole.attached_fold_line_id is not None:
                        j_idx = jid_to_idx.get(hole.attached_fold_line_id)
                        if j_idx is not None:
                            row[j_idx] += hole.plate_offset
                else:
                    pulley = self.design.pulleys.get(eid)
                    if pulley and pulley.attached_fold_line_id is not None:
                        j_idx = jid_to_idx.get(pulley.attached_fold_line_id)
                        if j_idx is not None:
                            row[j_idx] += pulley.radius
            R[tid] = row
        
        return R
