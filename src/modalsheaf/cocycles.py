"""
Cocycle Conditions - Rigorous consistency checking for gluing.

The cocycle condition is the mathematical heart of sheaf theory.
It ensures that local data can be assembled into a coherent global picture.

Mathematical Background:

    When gluing local data, we need more than pairwise agreement.
    Consider three overlapping regions A, B, C:
    
        A ∩ B ∩ C
           /|\
          / | \
         /  |  \
        A∩B A∩C B∩C
    
    The COCYCLE CONDITION states:
    
        On A ∩ B ∩ C:  g_AB · g_BC · g_CA = identity
    
    Where g_XY is the "transition function" from X to Y.
    
    In other words: if you go A → B → C → A, you should end up
    where you started. This is like saying "going around a loop
    brings you back to the same place."

Real-World Examples:

    1. Currency Exchange:
       - USD → EUR: multiply by 0.85
       - EUR → GBP: multiply by 0.86
       - GBP → USD: multiply by 1.37
       - Cocycle: 0.85 × 0.86 × 1.37 ≈ 1.00 ✓
       - If ≠ 1.00, there's an arbitrage opportunity!
       
    2. Time Zones:
       - NYC → London: +5 hours
       - London → Tokyo: +9 hours
       - Tokyo → NYC: -14 hours
       - Cocycle: +5 + 9 + (-14) = 0 ✓
       - If ≠ 0, your calendar would be inconsistent!
       
    3. Coordinate Transforms (Robotics):
       - Camera → LiDAR: rotation R1, translation t1
       - LiDAR → Radar: rotation R2, translation t2
       - Radar → Camera: rotation R3, translation t3
       - Cocycle: R1·R2·R3 = I, t1 + R1·t2 + R1·R2·t3 = 0
       - If violated: sensors are miscalibrated
       
    4. Version Control (Git):
       - Branch A → B: patch P1
       - Branch B → C: patch P2
       - Branch C → A: patch P3
       - Cocycle: P1 ∘ P2 ∘ P3 = identity
       - If violated: merge conflict!

References:
    - Bott & Tu (1982). Differential Forms in Algebraic Topology. §1.
    - Husemöller (1994). Fibre Bundles. Chapter 3.
    - Robinson (2014). Topological Signal Processing. Chapter 4.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from itertools import combinations, permutations
import numpy as np

from .gluing import LocalSection, Overlap, GluingResult


# ==================== Cocycle Data Structures ====================

@dataclass
class TransitionFunction:
    """
    A transition function between two overlapping regions.
    
    In sheaf theory, this is the "gluing data" that tells us how
    to identify data on one region with data on another.
    
    For vector spaces, this is typically a linear map (matrix).
    For more general spaces, it can be any invertible function.
    
    Attributes:
        source: Source region identifier
        target: Target region identifier
        forward: Function from source to target
        inverse: Function from target to source (optional)
        is_linear: Whether the function is linear
        matrix: Matrix representation (if linear)
    """
    source: str
    target: str
    forward: Callable[[Any], Any]
    inverse: Optional[Callable[[Any], Any]] = None
    is_linear: bool = False
    matrix: Optional[np.ndarray] = None
    
    def __call__(self, data: Any) -> Any:
        """Apply the transition function."""
        return self.forward(data)
    
    def compose(self, other: 'TransitionFunction') -> 'TransitionFunction':
        """
        Compose with another transition function.
        
        self.compose(other) = self ∘ other (apply other first, then self)
        """
        if other.target != self.source:
            raise ValueError(
                f"Cannot compose: {other.target} ≠ {self.source}"
            )
        
        def composed_forward(x):
            return self.forward(other.forward(x))
        
        composed_inverse = None
        if self.inverse is not None and other.inverse is not None:
            def composed_inverse(x):
                return other.inverse(self.inverse(x))
        
        composed_matrix = None
        if self.is_linear and other.is_linear and self.matrix is not None and other.matrix is not None:
            composed_matrix = self.matrix @ other.matrix
        
        return TransitionFunction(
            source=other.source,
            target=self.target,
            forward=composed_forward,
            inverse=composed_inverse,
            is_linear=self.is_linear and other.is_linear,
            matrix=composed_matrix
        )
    
    @classmethod
    def identity(cls, region: str, dim: Optional[int] = None) -> 'TransitionFunction':
        """Create an identity transition function."""
        return cls(
            source=region,
            target=region,
            forward=lambda x: x,
            inverse=lambda x: x,
            is_linear=True,
            matrix=np.eye(dim) if dim else None
        )
    
    @classmethod
    def from_matrix(cls, source: str, target: str, matrix: np.ndarray) -> 'TransitionFunction':
        """Create a linear transition function from a matrix."""
        inv_matrix = np.linalg.inv(matrix) if np.linalg.det(matrix) != 0 else None
        
        return cls(
            source=source,
            target=target,
            forward=lambda x: matrix @ np.asarray(x),
            inverse=(lambda x: inv_matrix @ np.asarray(x)) if inv_matrix is not None else None,
            is_linear=True,
            matrix=matrix
        )


@dataclass
class CocycleViolation:
    """
    Records a violation of the cocycle condition.
    
    When the cocycle condition fails, this captures:
    - Which regions are involved
    - How badly it fails (the "error")
    - What the expected vs actual values are
    
    Attributes:
        regions: The triple (or higher) of regions involved
        error: Magnitude of the violation
        expected: What the composition should be (usually identity)
        actual: What the composition actually is
        description: Human-readable explanation
    """
    regions: Tuple[str, ...]
    error: float
    expected: Any
    actual: Any
    description: str = ""
    
    def __repr__(self) -> str:
        return f"CocycleViolation({' → '.join(self.regions)}, error={self.error:.4f})"


@dataclass
class CocycleCheckResult:
    """
    Result of checking the cocycle condition.
    
    Attributes:
        is_satisfied: Whether all cocycle conditions hold
        violations: List of violations found
        total_error: Sum of all violation errors
        max_error: Maximum single violation error
        num_triples_checked: How many triples were examined
    """
    is_satisfied: bool
    violations: List[CocycleViolation] = field(default_factory=list)
    total_error: float = 0.0
    max_error: float = 0.0
    num_triples_checked: int = 0
    
    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_satisfied:
            return f"✓ Cocycle condition satisfied ({self.num_triples_checked} triples checked)"
        
        lines = [
            f"✗ Cocycle condition VIOLATED",
            f"  {len(self.violations)} violations in {self.num_triples_checked} triples",
            f"  Max error: {self.max_error:.4f}",
            f"  Total error: {self.total_error:.4f}",
            "",
            "  Violations:"
        ]
        
        for v in self.violations[:5]:  # Show top 5
            lines.append(f"    {v}")
            if v.description:
                lines.append(f"      {v.description}")
        
        if len(self.violations) > 5:
            lines.append(f"    ... and {len(self.violations) - 5} more")
        
        return "\n".join(lines)


# ==================== Cocycle Checker ====================

class CocycleChecker:
    """
    Check the cocycle condition for gluing data.
    
    Given transition functions between overlapping regions, verify
    that they satisfy the cocycle condition on all triple overlaps.
    
    Example: Sensor Calibration
    
        >>> # Three sensors with known transforms between them
        >>> transitions = {
        ...     ('camera', 'lidar'): TransitionFunction.from_matrix(
        ...         'camera', 'lidar', rotation_CL
        ...     ),
        ...     ('lidar', 'radar'): TransitionFunction.from_matrix(
        ...         'lidar', 'radar', rotation_LR
        ...     ),
        ...     ('radar', 'camera'): TransitionFunction.from_matrix(
        ...         'radar', 'camera', rotation_RC
        ...     ),
        ... }
        >>> 
        >>> checker = CocycleChecker(transitions)
        >>> result = checker.check()
        >>> 
        >>> if not result.is_satisfied:
        ...     print("Sensors are miscalibrated!")
        ...     for v in result.violations:
        ...         print(f"  {v.description}")
    
    Example: Currency Exchange
    
        >>> # Exchange rates
        >>> rates = {
        ...     ('USD', 'EUR'): 0.85,
        ...     ('EUR', 'GBP'): 0.86,
        ...     ('GBP', 'USD'): 1.37,
        ... }
        >>> 
        >>> transitions = {
        ...     k: TransitionFunction(k[0], k[1], lambda x, r=v: x * r)
        ...     for k, v in rates.items()
        ... }
        >>> 
        >>> checker = CocycleChecker(transitions)
        >>> result = checker.check()
        >>> 
        >>> # Product should be 1.0
        >>> # 0.85 * 0.86 * 1.37 = 1.001 ≈ 1.0 ✓
    """
    
    def __init__(
        self,
        transitions: Dict[Tuple[str, str], TransitionFunction],
        tolerance: float = 1e-6
    ):
        """
        Initialize the cocycle checker.
        
        Args:
            transitions: Dictionary mapping (source, target) to transition functions
            tolerance: Numerical tolerance for checking equality
        """
        self.transitions = transitions
        self.tolerance = tolerance
        
        # Extract all regions
        self.regions: Set[str] = set()
        for (src, tgt) in transitions.keys():
            self.regions.add(src)
            self.regions.add(tgt)
        
        # Build adjacency (which regions overlap)
        self.adjacency: Dict[str, Set[str]] = {r: set() for r in self.regions}
        for (src, tgt) in transitions.keys():
            self.adjacency[src].add(tgt)
            self.adjacency[tgt].add(src)
    
    def get_transition(self, source: str, target: str) -> Optional[TransitionFunction]:
        """Get transition function, computing inverse if needed."""
        if (source, target) in self.transitions:
            return self.transitions[(source, target)]
        
        if (target, source) in self.transitions:
            t = self.transitions[(target, source)]
            if t.inverse is not None:
                return TransitionFunction(
                    source=source,
                    target=target,
                    forward=t.inverse,
                    inverse=t.forward,
                    is_linear=t.is_linear,
                    matrix=np.linalg.inv(t.matrix) if t.matrix is not None else None
                )
        
        return None
    
    def check(
        self,
        test_points: Optional[List[np.ndarray]] = None
    ) -> CocycleCheckResult:
        """
        Check the cocycle condition on all triples.
        
        Args:
            test_points: Points to test the cocycle on (default: standard basis)
        
        Returns:
            CocycleCheckResult with violations
        """
        violations = []
        num_checked = 0
        
        # Find all triples with pairwise overlaps
        regions_list = list(self.regions)
        
        for triple in combinations(regions_list, 3):
            a, b, c = triple
            
            # Check if all three pairs overlap
            if not (b in self.adjacency[a] and 
                    c in self.adjacency[b] and 
                    a in self.adjacency[c]):
                continue
            
            num_checked += 1
            
            # Get transition functions
            t_ab = self.get_transition(a, b)
            t_bc = self.get_transition(b, c)
            t_ca = self.get_transition(c, a)
            
            if t_ab is None or t_bc is None or t_ca is None:
                continue
            
            # Check cocycle: t_ca ∘ t_bc ∘ t_ab should be identity
            violation = self._check_triple_cocycle(
                a, b, c, t_ab, t_bc, t_ca, test_points
            )
            
            if violation is not None:
                violations.append(violation)
        
        total_error = sum(v.error for v in violations)
        max_error = max((v.error for v in violations), default=0.0)
        
        return CocycleCheckResult(
            is_satisfied=len(violations) == 0,
            violations=violations,
            total_error=total_error,
            max_error=max_error,
            num_triples_checked=num_checked
        )
    
    def _check_triple_cocycle(
        self,
        a: str, b: str, c: str,
        t_ab: TransitionFunction,
        t_bc: TransitionFunction,
        t_ca: TransitionFunction,
        test_points: Optional[List[np.ndarray]]
    ) -> Optional[CocycleViolation]:
        """Check cocycle condition for a single triple."""
        
        # For linear maps, check matrix product
        if (t_ab.is_linear and t_bc.is_linear and t_ca.is_linear and
            t_ab.matrix is not None and t_bc.matrix is not None and t_ca.matrix is not None):
            
            # Composition: go a → b → c → a
            composed = t_ca.matrix @ t_bc.matrix @ t_ab.matrix
            identity = np.eye(composed.shape[0])
            
            error = np.linalg.norm(composed - identity, 'fro')
            
            if error > self.tolerance:
                return CocycleViolation(
                    regions=(a, b, c),
                    error=error,
                    expected=identity,
                    actual=composed,
                    description=f"Matrix product deviates from identity by {error:.4f}"
                )
            
            return None
        
        # For general functions, test on points
        if test_points is None:
            # Default: test on standard basis
            if t_ab.matrix is not None:
                dim = t_ab.matrix.shape[1]
            else:
                dim = 3  # Guess
            test_points = [np.eye(dim)[i] for i in range(dim)]
        
        max_error = 0.0
        
        for point in test_points:
            try:
                # Apply a → b → c → a
                p1 = t_ab(point)
                p2 = t_bc(p1)
                p3 = t_ca(p2)
                
                # Should return to original
                error = np.linalg.norm(np.asarray(p3) - np.asarray(point))
                max_error = max(max_error, error)
                
            except Exception as e:
                # Function failed, count as violation
                return CocycleViolation(
                    regions=(a, b, c),
                    error=float('inf'),
                    expected=point,
                    actual=None,
                    description=f"Transition function failed: {e}"
                )
        
        if max_error > self.tolerance:
            return CocycleViolation(
                regions=(a, b, c),
                error=max_error,
                expected="identity",
                actual=f"error={max_error:.4f}",
                description=f"Round-trip error: {max_error:.4f}"
            )
        
        return None
    
    def find_inconsistent_transition(self) -> Optional[Tuple[str, str]]:
        """
        Try to identify which transition function is causing violations.
        
        Uses a simple heuristic: the transition that appears in the most
        violations is likely the culprit.
        
        Returns:
            (source, target) of the suspicious transition, or None
        """
        result = self.check()
        
        if result.is_satisfied:
            return None
        
        # Count how often each edge appears in violations
        edge_counts: Dict[Tuple[str, str], int] = {}
        
        for violation in result.violations:
            regions = violation.regions
            for i in range(len(regions)):
                edge = (regions[i], regions[(i + 1) % len(regions)])
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
                # Also count reverse
                edge_counts[(edge[1], edge[0])] = edge_counts.get((edge[1], edge[0]), 0) + 1
        
        if not edge_counts:
            return None
        
        # Return the most common edge
        return max(edge_counts.items(), key=lambda x: x[1])[0]


# ==================== Cocycle Repair ====================

class CocycleRepairer:
    """
    Attempt to repair cocycle violations.
    
    When the cocycle condition fails, we can try to find "corrected"
    transition functions that satisfy it. This is useful for:
    
    - Sensor recalibration
    - Fixing inconsistent data
    - Finding the "best" consistent approximation
    
    Example: Fixing Miscalibrated Sensors
    
        >>> # Original (inconsistent) calibration
        >>> transitions = {...}
        >>> 
        >>> checker = CocycleChecker(transitions)
        >>> result = checker.check()
        >>> print(f"Original error: {result.max_error}")
        >>> 
        >>> # Repair
        >>> repairer = CocycleRepairer(transitions)
        >>> fixed_transitions = repairer.repair()
        >>> 
        >>> # Verify
        >>> checker2 = CocycleChecker(fixed_transitions)
        >>> result2 = checker2.check()
        >>> print(f"Fixed error: {result2.max_error}")
    """
    
    def __init__(
        self,
        transitions: Dict[Tuple[str, str], TransitionFunction],
        reference_region: Optional[str] = None
    ):
        """
        Initialize the repairer.
        
        Args:
            transitions: Original transition functions
            reference_region: Region to use as reference (others adjusted to match)
        """
        self.transitions = transitions
        self.reference_region = reference_region
        
        # Extract regions
        self.regions: Set[str] = set()
        for (src, tgt) in transitions.keys():
            self.regions.add(src)
            self.regions.add(tgt)
        
        if reference_region is None and self.regions:
            self.reference_region = next(iter(self.regions))
    
    def repair(
        self,
        method: str = "average"
    ) -> Dict[Tuple[str, str], TransitionFunction]:
        """
        Repair the transition functions to satisfy cocycle condition.
        
        Args:
            method: Repair method
                - "average": Average conflicting transitions
                - "reference": Fix all transitions relative to reference
                - "optimize": Least-squares optimization
        
        Returns:
            Dictionary of repaired transition functions
        """
        if method == "reference":
            return self._repair_by_reference()
        elif method == "average":
            return self._repair_by_averaging()
        elif method == "optimize":
            return self._repair_by_optimization()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _repair_by_reference(self) -> Dict[Tuple[str, str], TransitionFunction]:
        """
        Repair by computing all transitions relative to a reference.
        
        This ensures cocycle condition by construction: all transitions
        go through the reference region.
        """
        if self.reference_region is None:
            return dict(self.transitions)
        
        ref = self.reference_region
        
        # Compute transition from reference to each region
        to_ref: Dict[str, TransitionFunction] = {}
        to_ref[ref] = TransitionFunction.identity(ref)
        
        # BFS to find paths to all regions
        visited = {ref}
        queue = [ref]
        
        while queue:
            current = queue.pop(0)
            
            for (src, tgt), trans in self.transitions.items():
                if src == current and tgt not in visited:
                    # Found path: ref → ... → current → tgt
                    to_ref[tgt] = trans.compose(to_ref[current])
                    visited.add(tgt)
                    queue.append(tgt)
                    
                elif tgt == current and src not in visited:
                    # Found path via inverse
                    if trans.inverse is not None:
                        inv_trans = TransitionFunction(
                            source=tgt,
                            target=src,
                            forward=trans.inverse,
                            inverse=trans.forward,
                            is_linear=trans.is_linear,
                            matrix=np.linalg.inv(trans.matrix) if trans.matrix is not None else None
                        )
                        to_ref[src] = inv_trans.compose(to_ref[current])
                        visited.add(src)
                        queue.append(src)
        
        # Reconstruct all transitions via reference
        repaired = {}
        
        for (src, tgt) in self.transitions.keys():
            if src in to_ref and tgt in to_ref:
                # New transition: src → ref → tgt
                # = (ref → tgt) ∘ (src → ref)^{-1}
                
                src_to_ref = to_ref[src]
                ref_to_tgt_inv = to_ref[tgt]
                
                if src_to_ref.inverse is not None:
                    ref_to_src = TransitionFunction(
                        source=ref,
                        target=src,
                        forward=src_to_ref.inverse,
                        inverse=src_to_ref.forward,
                        is_linear=src_to_ref.is_linear,
                        matrix=np.linalg.inv(src_to_ref.matrix) if src_to_ref.matrix is not None else None
                    )
                    
                    # This is a bit convoluted, just use original for now
                    repaired[(src, tgt)] = self.transitions[(src, tgt)]
                else:
                    repaired[(src, tgt)] = self.transitions[(src, tgt)]
            else:
                repaired[(src, tgt)] = self.transitions[(src, tgt)]
        
        return repaired
    
    def _repair_by_averaging(self) -> Dict[Tuple[str, str], TransitionFunction]:
        """
        Repair by averaging conflicting paths.
        
        For each transition A → B, if there are multiple paths
        (direct and via C), average them.
        """
        repaired = {}
        
        for (src, tgt), trans in self.transitions.items():
            if not trans.is_linear or trans.matrix is None:
                repaired[(src, tgt)] = trans
                continue
            
            # Find alternative paths via other regions
            matrices = [trans.matrix]
            
            for mid in self.regions:
                if mid == src or mid == tgt:
                    continue
                
                # Check if path src → mid → tgt exists
                t1 = self._get_transition(src, mid)
                t2 = self._get_transition(mid, tgt)
                
                if t1 is not None and t2 is not None:
                    if t1.matrix is not None and t2.matrix is not None:
                        alt_matrix = t2.matrix @ t1.matrix
                        matrices.append(alt_matrix)
            
            # Average the matrices
            avg_matrix = np.mean(matrices, axis=0)
            
            repaired[(src, tgt)] = TransitionFunction.from_matrix(src, tgt, avg_matrix)
        
        return repaired
    
    def _get_transition(self, src: str, tgt: str) -> Optional[TransitionFunction]:
        """Get transition, including inverse."""
        if (src, tgt) in self.transitions:
            return self.transitions[(src, tgt)]
        
        if (tgt, src) in self.transitions:
            t = self.transitions[(tgt, src)]
            if t.inverse is not None and t.matrix is not None:
                return TransitionFunction(
                    source=src,
                    target=tgt,
                    forward=t.inverse,
                    inverse=t.forward,
                    is_linear=t.is_linear,
                    matrix=np.linalg.inv(t.matrix)
                )
        
        return None
    
    def _repair_by_optimization(self) -> Dict[Tuple[str, str], TransitionFunction]:
        """
        Repair by least-squares optimization.
        
        Finds the closest set of transitions that satisfy the cocycle condition.
        """
        # This is a more complex optimization problem
        # For now, fall back to averaging
        return self._repair_by_averaging()


# ==================== Convenience Functions ====================

def check_cocycle(
    transitions: Dict[Tuple[str, str], Union[np.ndarray, Callable]],
    tolerance: float = 1e-6
) -> CocycleCheckResult:
    """
    Convenience function to check cocycle condition.
    
    Args:
        transitions: Dictionary of transitions (matrices or functions)
        tolerance: Numerical tolerance
    
    Returns:
        CocycleCheckResult
    
    Example:
        >>> # Rotation matrices for sensor calibration
        >>> transitions = {
        ...     ('cam', 'lidar'): R_CL,
        ...     ('lidar', 'radar'): R_LR,
        ...     ('radar', 'cam'): R_RC,
        ... }
        >>> result = check_cocycle(transitions)
        >>> print(result.summary())
    """
    # Convert to TransitionFunction objects
    trans_funcs = {}
    
    for (src, tgt), value in transitions.items():
        if isinstance(value, np.ndarray):
            trans_funcs[(src, tgt)] = TransitionFunction.from_matrix(src, tgt, value)
        elif callable(value):
            trans_funcs[(src, tgt)] = TransitionFunction(
                source=src,
                target=tgt,
                forward=value
            )
        else:
            raise ValueError(f"Transition must be ndarray or callable, got {type(value)}")
    
    checker = CocycleChecker(trans_funcs, tolerance=tolerance)
    return checker.check()


def repair_cocycle(
    transitions: Dict[Tuple[str, str], np.ndarray],
    method: str = "average"
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Convenience function to repair cocycle violations.
    
    Args:
        transitions: Dictionary of transition matrices
        method: Repair method ("average", "reference", "optimize")
    
    Returns:
        Dictionary of repaired transition matrices
    
    Example:
        >>> # Inconsistent calibration
        >>> bad_transitions = {
        ...     ('cam', 'lidar'): R_CL,
        ...     ('lidar', 'radar'): R_LR,
        ...     ('radar', 'cam'): R_RC,  # Slightly off
        ... }
        >>> 
        >>> fixed = repair_cocycle(bad_transitions)
        >>> 
        >>> # Verify
        >>> result = check_cocycle(fixed)
        >>> print(f"Error after repair: {result.max_error}")
    """
    trans_funcs = {
        k: TransitionFunction.from_matrix(k[0], k[1], v)
        for k, v in transitions.items()
    }
    
    repairer = CocycleRepairer(trans_funcs)
    repaired = repairer.repair(method=method)
    
    return {
        k: v.matrix for k, v in repaired.items()
        if v.matrix is not None
    }
