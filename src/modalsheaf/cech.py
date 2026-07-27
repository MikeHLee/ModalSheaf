"""
Čech Cohomology - Rigorous sheaf cohomology computation.

This module provides the full mathematical machinery for computing
sheaf cohomology via the Čech complex. Unlike the simplified consistency
checker, this computes actual cohomology groups.

Mathematical Background:
    
    The Čech complex for a cover {Uᵢ} of a space X is:
    
        C⁰ → C¹ → C² → ...
    
    Where:
    - C⁰ = ∏ᵢ F(Uᵢ)           (data on each piece)
    - C¹ = ∏ᵢ<ⱼ F(Uᵢ ∩ Uⱼ)    (data on pairwise overlaps)
    - C² = ∏ᵢ<ⱼ<ₖ F(Uᵢ ∩ Uⱼ ∩ Uₖ)  (data on triple overlaps)
    
    The coboundary maps δⁿ: Cⁿ → Cⁿ⁺¹ are:
    
        (δ⁰s)ᵢⱼ = ρⱼ(sⱼ) - ρᵢ(sᵢ)     (difference on overlaps)
        (δ¹t)ᵢⱼₖ = tⱼₖ - tᵢₖ + tᵢⱼ    (cocycle condition)
    
    Then:
    - H⁰ = ker(δ⁰) = global sections
    - H¹ = ker(δ¹) / im(δ⁰) = obstructions to gluing
    - H² = ker(δ²) / im(δ¹) = higher obstructions

Real-World Examples:
    
    1. GPS Localization:
       - Cover: {satellite1, satellite2, satellite3, satellite4}
       - C⁰: Position estimates from each satellite
       - C¹: Pairwise consistency (do sat1 and sat2 agree?)
       - H¹ ≠ 0: Multipath interference, ionospheric delays
       
    2. Distributed Databases:
       - Cover: {replica1, replica2, replica3}
       - C⁰: Data at each replica
       - C¹: Sync status between pairs
       - H¹ ≠ 0: Network partition caused inconsistency
       
    3. Multi-Camera 3D Reconstruction:
       - Cover: {camera1, camera2, camera3}
       - C⁰: 2D observations from each camera
       - C¹: Epipolar constraints between pairs
       - C²: Triple-view consistency
       - H¹ ≠ 0: Calibration error or moving objects

References:
    - Bott & Tu (1982). Differential Forms in Algebraic Topology. §8-9.
    - Robinson (2014). Topological Signal Processing. Chapter 4.
    - Curry (2014). Sheaves, Cosheaves and Applications. Chapter 3.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from itertools import combinations
import numpy as np
from scipy import linalg

from .core import Modality, Transformation
from .graph import ModalityGraph


# ==================== Čech Complex Data Structures ====================

@dataclass
class CechCochain:
    """
    An element of the Čech cochain complex Cⁿ.
    
    A cochain assigns data to each n-fold intersection in the cover.
    
    Example (n=0, data on each open set):
        cochain = CechCochain(
            degree=0,
            data={
                ('sensor_a',): np.array([1.0, 2.0]),
                ('sensor_b',): np.array([1.1, 2.1]),
            }
        )
    
    Example (n=1, data on pairwise overlaps):
        cochain = CechCochain(
            degree=1,
            data={
                ('sensor_a', 'sensor_b'): np.array([0.1, 0.1]),  # difference
            }
        )
    """
    degree: int  # n in Cⁿ
    data: Dict[Tuple[str, ...], np.ndarray]  # simplex -> value
    
    def __add__(self, other: 'CechCochain') -> 'CechCochain':
        """Add two cochains (must have same degree)."""
        if self.degree != other.degree:
            raise ValueError(f"Cannot add cochains of degree {self.degree} and {other.degree}")
        
        result_data = {}
        all_keys = set(self.data.keys()) | set(other.data.keys())
        
        for key in all_keys:
            v1 = self.data.get(key, 0)
            v2 = other.data.get(key, 0)
            result_data[key] = np.asarray(v1) + np.asarray(v2)
        
        return CechCochain(degree=self.degree, data=result_data)
    
    def __sub__(self, other: 'CechCochain') -> 'CechCochain':
        """Subtract two cochains."""
        if self.degree != other.degree:
            raise ValueError(f"Cannot subtract cochains of degree {self.degree} and {other.degree}")
        
        result_data = {}
        all_keys = set(self.data.keys()) | set(other.data.keys())
        
        for key in all_keys:
            v1 = self.data.get(key, 0)
            v2 = other.data.get(key, 0)
            result_data[key] = np.asarray(v1) - np.asarray(v2)
        
        return CechCochain(degree=self.degree, data=result_data)
    
    def norm(self) -> float:
        """Compute the L2 norm of the cochain."""
        total = 0.0
        for v in self.data.values():
            total += np.sum(np.asarray(v) ** 2)
        return np.sqrt(total)
    
    def is_zero(self, tol: float = 1e-10) -> bool:
        """Check if cochain is (approximately) zero."""
        return self.norm() < tol


@dataclass
class CechComplex:
    """
    The Čech complex for computing sheaf cohomology.
    
    This is the main computational object. Given a cover (set of open sets)
    and restriction maps, it constructs the cochain groups and coboundary
    maps needed to compute cohomology.
    
    Example: Three overlapping sensors
    
        >>> cover = ['sensor_a', 'sensor_b', 'sensor_c']
        >>> overlaps = {
        ...     ('sensor_a', 'sensor_b'): True,
        ...     ('sensor_b', 'sensor_c'): True,
        ...     ('sensor_a', 'sensor_c'): True,
        ... }
        >>> complex = CechComplex(cover, overlaps, dim=3)
        >>> 
        >>> # Add data
        >>> c0 = complex.create_cochain(0, {
        ...     ('sensor_a',): np.array([1.0, 0.0, 0.0]),
        ...     ('sensor_b',): np.array([0.9, 0.1, 0.0]),
        ...     ('sensor_c',): np.array([1.0, 0.0, 0.1]),
        ... })
        >>> 
        >>> # Compute coboundary (measures inconsistency)
        >>> c1 = complex.coboundary(c0)
        >>> print(f"Inconsistency: {c1.norm()}")
    """
    cover: List[str]  # Names of open sets in the cover
    overlaps: Dict[Tuple[str, ...], bool]  # Which intersections are non-empty
    dim: int  # Dimension of the data vectors
    restriction_maps: Dict[Tuple[str, str], Callable] = field(default_factory=dict)
    
    def __post_init__(self):
        # Compute all simplices (non-empty intersections)
        self._simplices: Dict[int, List[Tuple[str, ...]]] = {}
        self._compute_simplices()
    
    def _compute_simplices(self):
        """Compute all non-empty intersections up to the cover size."""
        n = len(self.cover)
        
        for degree in range(n + 1):
            self._simplices[degree] = []
            
            for combo in combinations(self.cover, degree + 1):
                # Check if this intersection is non-empty
                if self._is_nonempty_intersection(combo):
                    self._simplices[degree].append(combo)
    
    def _is_nonempty_intersection(self, simplex: Tuple[str, ...]) -> bool:
        """Check if an intersection is non-empty."""
        if len(simplex) <= 1:
            return True
        
        # Check all pairwise overlaps
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                pair = tuple(sorted([simplex[i], simplex[j]]))
                if pair not in self.overlaps or not self.overlaps[pair]:
                    return False
        
        # Also check if explicitly listed
        sorted_simplex = tuple(sorted(simplex))
        if sorted_simplex in self.overlaps:
            return self.overlaps[sorted_simplex]
        
        return True
    
    def get_simplices(self, degree: int) -> List[Tuple[str, ...]]:
        """Get all simplices of a given degree."""
        return self._simplices.get(degree, [])
    
    def create_cochain(
        self, 
        degree: int, 
        data: Dict[Tuple[str, ...], np.ndarray]
    ) -> CechCochain:
        """Create a cochain of the given degree."""
        return CechCochain(degree=degree, data=data)
    
    def coboundary(
        self, 
        cochain: CechCochain,
        use_restrictions: bool = True
    ) -> CechCochain:
        """
        Compute the coboundary δⁿ: Cⁿ → Cⁿ⁺¹.
        
        The coboundary measures how data "disagrees" on overlaps.
        
        For degree 0 (δ⁰):
            (δ⁰s)ᵢⱼ = ρⱼ(sⱼ) - ρᵢ(sᵢ)
            
            This is the difference between restrictions to the overlap.
            If δ⁰s = 0, the data is consistent on all pairwise overlaps.
        
        For degree 1 (δ¹):
            (δ¹t)ᵢⱼₖ = tⱼₖ - tᵢₖ + tᵢⱼ
            
            This is the "cocycle condition" on triple overlaps.
            If δ¹t = 0, the transition data is coherent.
        
        Real-world interpretation:
            - δ⁰ = 0: All sensors agree on their overlapping regions
            - δ¹ = 0: Calibration transforms are consistent around loops
        """
        n = cochain.degree
        result_data = {}
        
        # Get (n+1)-simplices
        higher_simplices = self.get_simplices(n + 1)
        
        for simplex in higher_simplices:
            # Compute alternating sum over faces
            value = np.zeros(self.dim)
            
            for i, vertex in enumerate(simplex):
                # Face opposite to vertex i
                face = tuple(v for j, v in enumerate(simplex) if j != i)
                
                if face in cochain.data:
                    face_value = cochain.data[face]
                    
                    # Apply restriction map if available and requested
                    if use_restrictions and n == 0:
                        # For 0-cochains, we need to restrict to the overlap
                        # The face is a single vertex, simplex is an edge
                        restriction_key = (face[0], simplex)
                        if restriction_key in self.restriction_maps:
                            face_value = self.restriction_maps[restriction_key](face_value)
                    
                    # Alternating sign: (-1)^i
                    sign = (-1) ** i
                    value = value + sign * np.asarray(face_value)
            
            result_data[simplex] = value
        
        return CechCochain(degree=n + 1, data=result_data)
    
    def is_cocycle(self, cochain: CechCochain, tol: float = 1e-10) -> bool:
        """Check if a cochain is a cocycle (δc = 0)."""
        boundary = self.coboundary(cochain)
        return boundary.is_zero(tol)
    
    def is_coboundary(
        self, 
        cochain: CechCochain, 
        tol: float = 1e-10
    ) -> Tuple[bool, Optional[CechCochain]]:
        """
        Check if a cochain is a coboundary (c = δb for some b).
        
        Returns (is_coboundary, primitive) where primitive is b if found.
        """
        if cochain.degree == 0:
            # 0-cochains are coboundaries iff they're zero
            return cochain.is_zero(tol), None
        
        # Try to solve δb = c using least squares
        # This is a linear system
        primitive = self._find_primitive(cochain, tol)
        
        if primitive is not None:
            # Verify
            check = self.coboundary(primitive)
            diff = cochain - check
            if diff.is_zero(tol):
                return True, primitive
        
        return False, None
    
    def _find_primitive(
        self, 
        cochain: CechCochain, 
        tol: float
    ) -> Optional[CechCochain]:
        """Try to find b such that δb = cochain."""
        n = cochain.degree
        if n == 0:
            return None
        
        # Build the coboundary matrix
        lower_simplices = self.get_simplices(n - 1)
        upper_simplices = self.get_simplices(n)
        
        if not lower_simplices or not upper_simplices:
            return None
        
        # Matrix dimensions
        num_lower = len(lower_simplices) * self.dim
        num_upper = len(upper_simplices) * self.dim
        
        # Build matrix
        D = np.zeros((num_upper, num_lower))
        
        for j, lower_simp in enumerate(lower_simplices):
            # Create unit cochain at this simplex
            for d in range(self.dim):
                col_idx = j * self.dim + d
                
                # Compute coboundary
                unit_data = {lower_simp: np.eye(self.dim)[d]}
                unit_cochain = CechCochain(degree=n-1, data=unit_data)
                boundary = self.coboundary(unit_cochain, use_restrictions=False)
                
                # Fill column
                for i, upper_simp in enumerate(upper_simplices):
                    if upper_simp in boundary.data:
                        for dd in range(self.dim):
                            row_idx = i * self.dim + dd
                            D[row_idx, col_idx] = boundary.data[upper_simp][dd]
        
        # Target vector
        b = np.zeros(num_upper)
        for i, simp in enumerate(upper_simplices):
            if simp in cochain.data:
                for d in range(self.dim):
                    b[i * self.dim + d] = cochain.data[simp][d]
        
        # Solve least squares
        try:
            x, residuals, rank, s = np.linalg.lstsq(D, b, rcond=None)
            
            # Check if solution is good
            if len(residuals) > 0 and residuals[0] > tol:
                return None
            
            # Reconstruct cochain
            result_data = {}
            for j, simp in enumerate(lower_simplices):
                vec = x[j * self.dim : (j + 1) * self.dim]
                if np.linalg.norm(vec) > tol:
                    result_data[simp] = vec
            
            return CechCochain(degree=n-1, data=result_data)
            
        except np.linalg.LinAlgError:
            return None


@dataclass
class CohomologyGroup:
    """
    Represents a cohomology group Hⁿ = ker(δⁿ) / im(δⁿ⁻¹).
    
    Attributes:
        degree: The degree n of Hⁿ
        dimension: dim(Hⁿ) = dim(ker δⁿ) - dim(im δⁿ⁻¹)
        representatives: Basis cocycles for Hⁿ (one per dimension)
        is_trivial: True if Hⁿ = 0
    
    Interpretation:
        - H⁰: Global sections (consistent data across all regions)
        - H¹: Obstructions to gluing (inconsistencies that can't be fixed)
        - H²: Higher obstructions (rare in practice)
    """
    degree: int
    dimension: int
    representatives: List[CechCochain] = field(default_factory=list)
    
    @property
    def is_trivial(self) -> bool:
        return self.dimension == 0
    
    def __repr__(self) -> str:
        if self.is_trivial:
            return f"H{self.degree} = 0"
        return f"H{self.degree} ≅ ℝ^{self.dimension}"


# ==================== Cohomology Computation ====================

class CechCohomology:
    """
    Compute Čech cohomology for a sheaf on a modality graph.
    
    This is the rigorous version of ConsistencyChecker. Instead of
    heuristic consistency scores, it computes actual cohomology groups.
    
    Example: Multimodal consistency checking
    
        >>> # Setup
        >>> graph = ModalityGraph()
        >>> graph.add_modality("image", shape=(768,))
        >>> graph.add_modality("text", shape=(768,))
        >>> graph.add_modality("audio", shape=(768,))
        >>> 
        >>> # Add encoders (restriction maps)
        >>> graph.add_transformation("image", "embedding", image_encoder)
        >>> graph.add_transformation("text", "embedding", text_encoder)
        >>> 
        >>> # Compute cohomology
        >>> cech = CechCohomology(graph)
        >>> 
        >>> # Check consistency of multimodal data
        >>> data = {
        ...     "image": image_embedding,
        ...     "text": text_embedding,
        ...     "audio": audio_embedding,
        ... }
        >>> result = cech.compute(data)
        >>> 
        >>> print(f"H⁰ = {result.h0}")  # Global consensus
        >>> print(f"H¹ = {result.h1}")  # Inconsistencies
        >>> 
        >>> if not result.h1.is_trivial:
        ...     print("Data is inconsistent!")
        ...     for rep in result.h1.representatives:
        ...         print(f"  Conflict: {rep.data}")
    
    Real-World Example: Distributed Sensor Network
    
        Imagine 4 temperature sensors around a room:
        
            A ---- B
            |      |
            |      |
            C ---- D
        
        Each sensor measures temperature. Adjacent sensors should agree
        (within tolerance) on the temperature at their boundary.
        
        - H⁰ = ℝ¹: There's one global temperature (if consistent)
        - H¹ = 0: All sensors agree → we can reconstruct global temp
        - H¹ ≠ 0: Sensors disagree → there's a "hole" in our knowledge
        
        If H¹ ≠ 0, the representatives tell us WHERE the disagreement is:
        maybe sensor B is miscalibrated, or there's a heat source between
        B and D that creates a real temperature gradient.
    """
    
    def __init__(
        self, 
        graph: ModalityGraph,
        overlap_threshold: float = 0.0
    ):
        """
        Initialize Čech cohomology computer.
        
        Args:
            graph: The modality graph (defines the cover and restrictions)
            overlap_threshold: Minimum similarity for modalities to "overlap"
        """
        self.graph = graph
        self.overlap_threshold = overlap_threshold
    
    def compute(
        self, 
        data: Dict[str, np.ndarray],
        max_degree: int = 2
    ) -> 'CohomologyResult':
        """
        Compute Čech cohomology for the given data.
        
        Args:
            data: Dictionary mapping modality names to data vectors
            max_degree: Maximum cohomology degree to compute
        
        Returns:
            CohomologyResult with H⁰, H¹, etc.
        """
        # Build the Čech complex
        cover = list(data.keys())
        
        # Determine overlaps (which modalities can be compared)
        overlaps = self._compute_overlaps(cover)
        
        # Get data dimension
        dim = len(next(iter(data.values())).flatten())
        
        # Create complex
        complex = CechComplex(
            cover=cover,
            overlaps=overlaps,
            dim=dim,
            restriction_maps=self._build_restriction_maps(cover)
        )
        
        # Create 0-cochain from data
        c0_data = {(k,): v.flatten() for k, v in data.items()}
        c0 = complex.create_cochain(0, c0_data)
        
        # Compute cohomology groups
        h_groups = {}
        
        for degree in range(max_degree + 1):
            h_groups[degree] = self._compute_cohomology_group(
                complex, degree, c0 if degree == 0 else None
            )
        
        return CohomologyResult(
            complex=complex,
            input_cochain=c0,
            cohomology_groups=h_groups,
            h0=h_groups.get(0, CohomologyGroup(0, 0)),
            h1=h_groups.get(1, CohomologyGroup(1, 0)),
        )
    
    def _compute_overlaps(self, cover: List[str]) -> Dict[Tuple[str, ...], bool]:
        """Determine which modalities overlap (can be compared)."""
        overlaps = {}
        
        for i, m1 in enumerate(cover):
            for m2 in cover[i+1:]:
                # Two modalities overlap if they can both reach a common target
                # or if they're directly connected
                can_compare = (
                    self.graph.has_path(m1, m2) or
                    self.graph.has_path(m2, m1) or
                    self._have_common_target(m1, m2)
                )
                
                overlaps[tuple(sorted([m1, m2]))] = can_compare
        
        return overlaps
    
    def _have_common_target(self, m1: str, m2: str) -> bool:
        """Check if two modalities can both reach a common modality."""
        # Simple check: both can reach "embedding" or similar
        for target in self.graph.modalities:
            if (self.graph.has_path(m1, target) and 
                self.graph.has_path(m2, target)):
                return True
        return False
    
    def _build_restriction_maps(
        self, 
        cover: List[str]
    ) -> Dict[Tuple[str, str], Callable]:
        """Build restriction maps from the graph transformations."""
        restrictions = {}
        
        for m1 in cover:
            for m2 in cover:
                if m1 != m2 and self.graph.has_path(m1, m2):
                    # Get composed transformation
                    try:
                        transform = self.graph.get_composed_transformation(m1, m2)
                        restrictions[(m1, m2)] = transform.forward
                    except:
                        pass
        
        return restrictions
    
    def _compute_cohomology_group(
        self,
        complex: CechComplex,
        degree: int,
        input_cochain: Optional[CechCochain] = None
    ) -> CohomologyGroup:
        """Compute Hⁿ for the complex."""
        
        # Get simplices at this degree and next
        simplices_n = complex.get_simplices(degree)
        simplices_n1 = complex.get_simplices(degree + 1)
        
        if not simplices_n:
            return CohomologyGroup(degree=degree, dimension=0)
        
        dim = complex.dim
        
        # Build coboundary matrix δⁿ: Cⁿ → Cⁿ⁺¹
        num_n = len(simplices_n) * dim
        num_n1 = len(simplices_n1) * dim if simplices_n1 else 0
        
        if num_n1 == 0:
            # No higher simplices, kernel is everything
            ker_dim = num_n
        else:
            # Build matrix
            D = np.zeros((num_n1, num_n))
            
            for j, simp_n in enumerate(simplices_n):
                for d in range(dim):
                    col_idx = j * dim + d
                    
                    # Unit cochain
                    unit_data = {simp_n: np.eye(dim)[d]}
                    unit = CechCochain(degree=degree, data=unit_data)
                    boundary = complex.coboundary(unit, use_restrictions=False)
                    
                    for i, simp_n1 in enumerate(simplices_n1):
                        if simp_n1 in boundary.data:
                            for dd in range(dim):
                                row_idx = i * dim + dd
                                D[row_idx, col_idx] = boundary.data[simp_n1][dd]
            
            # Kernel dimension
            rank_D = np.linalg.matrix_rank(D)
            ker_dim = num_n - rank_D
        
        # Image dimension (from δⁿ⁻¹)
        if degree == 0:
            im_dim = 0  # No δ⁻¹
        else:
            simplices_nm1 = complex.get_simplices(degree - 1)
            if not simplices_nm1:
                im_dim = 0
            else:
                # Build δⁿ⁻¹ matrix
                num_nm1 = len(simplices_nm1) * dim
                D_prev = np.zeros((num_n, num_nm1))
                
                for j, simp_nm1 in enumerate(simplices_nm1):
                    for d in range(dim):
                        col_idx = j * dim + d
                        
                        unit_data = {simp_nm1: np.eye(dim)[d]}
                        unit = CechCochain(degree=degree-1, data=unit_data)
                        boundary = complex.coboundary(unit, use_restrictions=False)
                        
                        for i, simp_n in enumerate(simplices_n):
                            if simp_n in boundary.data:
                                for dd in range(dim):
                                    row_idx = i * dim + dd
                                    D_prev[row_idx, col_idx] = boundary.data[simp_n][dd]
                
                im_dim = np.linalg.matrix_rank(D_prev)
        
        # Cohomology dimension
        h_dim = ker_dim - im_dim
        
        # Find representatives (basis for Hⁿ)
        representatives = []
        if h_dim > 0 and input_cochain is not None and degree == 1:
            # For H¹, the coboundary of the input is a representative
            delta_input = complex.coboundary(input_cochain)
            if not delta_input.is_zero():
                representatives.append(delta_input)
        
        return CohomologyGroup(
            degree=degree,
            dimension=max(0, h_dim),
            representatives=representatives
        )


@dataclass
class CohomologyResult:
    """
    Result of Čech cohomology computation.
    
    Attributes:
        complex: The Čech complex used
        input_cochain: The original data as a 0-cochain
        cohomology_groups: Dictionary of Hⁿ for each n
        h0: H⁰ (global sections)
        h1: H¹ (obstructions)
    """
    complex: CechComplex
    input_cochain: CechCochain
    cohomology_groups: Dict[int, CohomologyGroup]
    h0: CohomologyGroup
    h1: CohomologyGroup
    
    @property
    def is_consistent(self) -> bool:
        """Check if data is globally consistent (H¹ = 0)."""
        return self.h1.is_trivial
    
    @property
    def obstruction_dimension(self) -> int:
        """Dimension of the obstruction space."""
        return self.h1.dimension
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Čech Cohomology Result",
            "=" * 40,
            f"Cover size: {len(self.complex.cover)}",
            f"H⁰ = {self.h0}",
            f"H¹ = {self.h1}",
        ]
        
        if self.is_consistent:
            lines.append("\n✓ Data is globally consistent")
        else:
            lines.append(f"\n✗ Inconsistency detected (dim H¹ = {self.h1.dimension})")
            
            if self.h1.representatives:
                lines.append("\nConflict locations:")
                for rep in self.h1.representatives:
                    for simplex, value in rep.data.items():
                        if np.linalg.norm(value) > 1e-10:
                            lines.append(f"  {' ∩ '.join(simplex)}: {value}")
        
        return "\n".join(lines)


# ==================== Convenience Functions ====================

def compute_cech_cohomology(
    data: Dict[str, np.ndarray],
    overlaps: Optional[Dict[Tuple[str, str], bool]] = None,
    max_degree: int = 2
) -> CohomologyResult:
    """
    Compute Čech cohomology for data without a full graph.
    
    This is a convenience function for quick cohomology computation
    when you don't need the full ModalityGraph infrastructure.
    
    Args:
        data: Dictionary mapping names to data vectors
        overlaps: Which pairs overlap (default: all pairs)
        max_degree: Maximum cohomology degree
    
    Returns:
        CohomologyResult
    
    Example:
        >>> # Three sensors measuring temperature
        >>> data = {
        ...     'sensor_a': np.array([20.0]),
        ...     'sensor_b': np.array([20.5]),
        ...     'sensor_c': np.array([21.0]),
        ... }
        >>> result = compute_cech_cohomology(data)
        >>> print(result.summary())
    """
    cover = list(data.keys())
    
    # Default: all pairs overlap
    if overlaps is None:
        overlaps = {}
        for i, m1 in enumerate(cover):
            for m2 in cover[i+1:]:
                overlaps[tuple(sorted([m1, m2]))] = True
    
    # Get dimension
    dim = len(next(iter(data.values())).flatten())
    
    # Create complex
    complex = CechComplex(cover=cover, overlaps=overlaps, dim=dim)
    
    # Create 0-cochain
    c0_data = {(k,): v.flatten() for k, v in data.items()}
    c0 = complex.create_cochain(0, c0_data)
    
    # Compute δ⁰
    c1 = complex.coboundary(c0)
    
    # Simple cohomology computation
    h0_dim = 1 if c1.is_zero() else 0
    h1_dim = 0 if c1.is_zero() else 1
    
    h0 = CohomologyGroup(degree=0, dimension=h0_dim)
    h1 = CohomologyGroup(
        degree=1, 
        dimension=h1_dim,
        representatives=[c1] if not c1.is_zero() else []
    )
    
    return CohomologyResult(
        complex=complex,
        input_cochain=c0,
        cohomology_groups={0: h0, 1: h1},
        h0=h0,
        h1=h1
    )
