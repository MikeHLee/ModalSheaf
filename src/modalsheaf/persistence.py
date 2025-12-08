"""
Persistent Cohomology - Tracking consistency across scales.

Real data is noisy. A small measurement error shouldn't count as 
"inconsistency," but a large systematic disagreement should. Persistent
cohomology tracks how inconsistencies appear and disappear as we vary
a tolerance threshold.

Mathematical Background:

    Standard cohomology gives a binary answer: consistent or not.
    Persistent cohomology gives a SPECTRUM of answers:
    
    - At tolerance ε = 0: Every tiny difference is an inconsistency
    - At tolerance ε = 0.1: Small differences are ignored
    - At tolerance ε = 1.0: Only large disagreements count
    - At tolerance ε = ∞: Everything is "consistent"
    
    We track the "birth" and "death" of cohomology classes:
    - Birth: The tolerance at which an inconsistency first appears
    - Death: The tolerance at which it gets resolved
    - Persistence: Death - Birth (how "real" the inconsistency is)
    
    Long-lived features (high persistence) are real structure.
    Short-lived features (low persistence) are noise.

Real-World Examples:

    1. Sensor Calibration:
       - 3 temperature sensors: 20.0°C, 20.1°C, 20.5°C
       - At ε = 0.05: All three disagree (H¹ = 2)
       - At ε = 0.15: Sensors 1&2 agree, but 3 disagrees (H¹ = 1)
       - At ε = 0.6: All "agree" within tolerance (H¹ = 0)
       - The 0.5°C difference persists longer → sensor 3 may be miscalibrated
       
    2. Image-Caption Matching:
       - CLIP embeddings: image_emb, text_emb
       - At ε = 0: Almost never exactly equal
       - At ε = 0.3: Good matches are consistent
       - At ε = 0.7: Even poor matches seem consistent
       - Persistence tells us the "quality" of the match
       
    3. Multi-View 3D Reconstruction:
       - 3 cameras observing a scene
       - At ε = 0: Pixel-perfect alignment required
       - At ε = 5px: Reasonable calibration tolerance
       - At ε = 50px: Very loose matching
       - Persistent features are real 3D points; transient ones are noise

References:
    - Edelsbrunner & Harer (2010). Computational Topology. Chapter VII.
    - Carlsson (2009). "Topology and Data." Bull. AMS.
    - Robinson (2014). Topological Signal Processing. Chapter 5.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from bisect import bisect_left

from .cech import CechComplex, CechCochain, CohomologyGroup, CohomologyResult


# ==================== Persistence Data Structures ====================

@dataclass
class PersistenceInterval:
    """
    A persistence interval [birth, death) for a cohomology class.
    
    Represents a feature that:
    - Appears (is born) at tolerance = birth
    - Disappears (dies) at tolerance = death
    - Has persistence = death - birth
    
    Longer persistence = more significant feature.
    
    Attributes:
        birth: Tolerance at which feature appears
        death: Tolerance at which feature disappears (inf if never)
        dimension: Cohomology dimension (0 for H⁰, 1 for H¹, etc.)
        representative: The cocycle representing this class
        metadata: Additional information (e.g., which simplices involved)
    """
    birth: float
    death: float
    dimension: int
    representative: Optional[CechCochain] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def persistence(self) -> float:
        """How long this feature persists."""
        if self.death == float('inf'):
            return float('inf')
        return self.death - self.birth
    
    @property
    def midpoint(self) -> float:
        """Midpoint of the interval (useful for visualization)."""
        if self.death == float('inf'):
            return self.birth + 1.0  # Arbitrary for infinite
        return (self.birth + self.death) / 2
    
    def is_significant(self, threshold: float) -> bool:
        """Check if persistence exceeds threshold."""
        return self.persistence > threshold
    
    def __repr__(self) -> str:
        death_str = "∞" if self.death == float('inf') else f"{self.death:.3f}"
        return f"H{self.dimension}[{self.birth:.3f}, {death_str}) pers={self.persistence:.3f}"


@dataclass
class PersistenceDiagram:
    """
    A persistence diagram: collection of persistence intervals.
    
    The diagram summarizes all topological features across scales:
    - Points near the diagonal (birth ≈ death) are noise
    - Points far from diagonal are significant features
    
    Visualization:
        Plot each interval as a point (birth, death).
        The diagonal y = x represents zero persistence.
        Distance from diagonal = significance.
        
            death
              ^
              |     * (significant)
              |   *
              | *   * (noise)
              |*  *
              +---------> birth
              
    Attributes:
        intervals: List of persistence intervals
        dimension: Which cohomology dimension this diagram represents
    """
    intervals: List[PersistenceInterval] = field(default_factory=list)
    dimension: Optional[int] = None
    
    def add_interval(self, interval: PersistenceInterval) -> None:
        """Add an interval to the diagram."""
        self.intervals.append(interval)
        if self.dimension is None:
            self.dimension = interval.dimension
    
    def filter_by_persistence(self, min_persistence: float) -> 'PersistenceDiagram':
        """Get only intervals with persistence above threshold."""
        filtered = [i for i in self.intervals if i.persistence > min_persistence]
        return PersistenceDiagram(intervals=filtered, dimension=self.dimension)
    
    def total_persistence(self) -> float:
        """Sum of all finite persistences."""
        return sum(
            i.persistence for i in self.intervals 
            if i.persistence != float('inf')
        )
    
    def max_persistence(self) -> float:
        """Maximum finite persistence."""
        finite = [i.persistence for i in self.intervals if i.persistence != float('inf')]
        return max(finite) if finite else 0.0
    
    def num_significant(self, threshold: float) -> int:
        """Count intervals with persistence above threshold."""
        return sum(1 for i in self.intervals if i.is_significant(threshold))
    
    def bottleneck_distance(self, other: 'PersistenceDiagram') -> float:
        """
        Compute bottleneck distance to another diagram.
        
        This is a metric on persistence diagrams that measures
        how "different" two diagrams are. Useful for comparing
        consistency patterns across different data.
        """
        # Simplified: just compare total persistence
        # Full implementation would use Hungarian algorithm
        return abs(self.total_persistence() - other.total_persistence())
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array of (birth, death) pairs."""
        if not self.intervals:
            return np.array([]).reshape(0, 2)
        
        return np.array([
            [i.birth, i.death if i.death != float('inf') else -1]
            for i in self.intervals
        ])
    
    def summary(self) -> str:
        """Human-readable summary."""
        if not self.intervals:
            return f"H{self.dimension}: Empty diagram (no features)"
        
        lines = [
            f"H{self.dimension} Persistence Diagram",
            f"  {len(self.intervals)} intervals",
            f"  Max persistence: {self.max_persistence():.3f}",
            f"  Total persistence: {self.total_persistence():.3f}",
        ]
        
        # Show top intervals
        sorted_intervals = sorted(
            self.intervals, 
            key=lambda i: -i.persistence if i.persistence != float('inf') else float('inf')
        )
        
        lines.append("  Top intervals:")
        for interval in sorted_intervals[:5]:
            lines.append(f"    {interval}")
        
        return "\n".join(lines)


@dataclass
class PersistentCohomologyResult:
    """
    Result of persistent cohomology computation.
    
    Contains persistence diagrams for each cohomology dimension,
    plus utilities for interpretation.
    """
    diagrams: Dict[int, PersistenceDiagram]
    thresholds: np.ndarray  # The tolerance values used
    cohomology_at_threshold: Dict[float, Dict[int, int]]  # threshold -> {dim: rank}
    
    def get_diagram(self, dimension: int) -> PersistenceDiagram:
        """Get persistence diagram for a specific dimension."""
        return self.diagrams.get(dimension, PersistenceDiagram(dimension=dimension))
    
    def consistency_at_threshold(self, threshold: float) -> bool:
        """Check if data is consistent at given tolerance."""
        # Find nearest threshold
        idx = bisect_left(self.thresholds, threshold)
        if idx >= len(self.thresholds):
            idx = len(self.thresholds) - 1
        
        actual_threshold = self.thresholds[idx]
        ranks = self.cohomology_at_threshold.get(actual_threshold, {})
        
        # Consistent if H¹ = 0
        return ranks.get(1, 0) == 0
    
    def recommended_threshold(self, max_noise: float = 0.1) -> float:
        """
        Recommend a tolerance threshold that filters noise.
        
        Args:
            max_noise: Maximum persistence to consider as noise
        
        Returns:
            Threshold that eliminates noise but preserves signal
        """
        h1_diagram = self.get_diagram(1)
        
        if not h1_diagram.intervals:
            return 0.0
        
        # Find the smallest birth time of a significant feature
        significant = [
            i for i in h1_diagram.intervals 
            if i.persistence > max_noise
        ]
        
        if not significant:
            # All features are noise
            return h1_diagram.max_persistence() + 0.01
        
        # Threshold just below the first significant feature
        min_birth = min(i.birth for i in significant)
        return max(0, min_birth - 0.01)
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Persistent Cohomology Result",
            "=" * 40,
            f"Thresholds: {len(self.thresholds)} values from {self.thresholds[0]:.3f} to {self.thresholds[-1]:.3f}",
        ]
        
        for dim, diagram in sorted(self.diagrams.items()):
            lines.append("")
            lines.append(diagram.summary())
        
        # Recommendation
        rec = self.recommended_threshold()
        lines.append("")
        lines.append(f"Recommended threshold: {rec:.3f}")
        lines.append(f"  Consistent at this threshold: {self.consistency_at_threshold(rec)}")
        
        return "\n".join(lines)


# ==================== Persistent Cohomology Computation ====================

class PersistentCohomology:
    """
    Compute persistent cohomology for multimodal data.
    
    This extends the Čech cohomology computation to track how
    cohomology changes as we vary a tolerance threshold.
    
    Example: Noisy sensor readings
    
        >>> sensors = {
        ...     'a': np.array([20.0]),
        ...     'b': np.array([20.1]),
        ...     'c': np.array([20.5]),
        ... }
        >>> 
        >>> pc = PersistentCohomology()
        >>> result = pc.compute(sensors, thresholds=np.linspace(0, 1, 100))
        >>> 
        >>> print(result.summary())
        >>> # Shows which disagreements are noise vs real
        >>> 
        >>> # Get recommended threshold
        >>> threshold = result.recommended_threshold()
        >>> print(f"Use tolerance: {threshold}")
    
    Real-World Application: CLIP Embedding Matching
    
        When matching images to captions using CLIP:
        
        >>> image_emb = clip.encode_image(image)
        >>> text_emb = clip.encode_text(caption)
        >>> 
        >>> # Compute persistent cohomology
        >>> data = {'image': image_emb, 'text': text_emb}
        >>> result = pc.compute(data)
        >>> 
        >>> # The persistence of H¹ tells us match quality:
        >>> # - Low persistence: Good match (disagreement is just noise)
        >>> # - High persistence: Poor match (real semantic difference)
        >>> 
        >>> match_quality = 1.0 - result.get_diagram(1).max_persistence()
    """
    
    def __init__(
        self,
        distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    ):
        """
        Initialize persistent cohomology computer.
        
        Args:
            distance_fn: Function to compute distance between data points.
                        Default: Euclidean distance.
        """
        if distance_fn is None:
            self.distance_fn = lambda a, b: np.linalg.norm(a - b)
        else:
            self.distance_fn = distance_fn
    
    def compute(
        self,
        data: Dict[str, np.ndarray],
        thresholds: Optional[np.ndarray] = None,
        max_degree: int = 1
    ) -> PersistentCohomologyResult:
        """
        Compute persistent cohomology.
        
        Args:
            data: Dictionary mapping names to data vectors
            thresholds: Tolerance values to use (default: auto)
            max_degree: Maximum cohomology degree
        
        Returns:
            PersistentCohomologyResult with diagrams and analysis
        """
        # Flatten data
        flat_data = {k: v.flatten() for k, v in data.items()}
        
        # Compute pairwise distances
        names = list(flat_data.keys())
        distances = {}
        
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                d = self.distance_fn(flat_data[n1], flat_data[n2])
                distances[(n1, n2)] = d
                distances[(n2, n1)] = d
        
        # Auto-generate thresholds if not provided
        if thresholds is None:
            max_dist = max(distances.values()) if distances else 1.0
            thresholds = np.linspace(0, max_dist * 1.5, 50)
        
        # Track cohomology at each threshold
        cohomology_at_threshold = {}
        
        # Track when features are born/die
        feature_births: Dict[int, List[Tuple[float, CechCochain]]] = {d: [] for d in range(max_degree + 1)}
        feature_deaths: Dict[int, List[float]] = {d: [] for d in range(max_degree + 1)}
        
        prev_ranks = {d: 0 for d in range(max_degree + 1)}
        
        for threshold in thresholds:
            # Determine which pairs are "consistent" at this threshold
            overlaps = {}
            for (n1, n2), dist in distances.items():
                if n1 < n2:  # Avoid duplicates
                    overlaps[tuple(sorted([n1, n2]))] = True  # All overlap
            
            # Build complex
            dim = len(next(iter(flat_data.values())))
            complex = CechComplex(
                cover=names,
                overlaps=overlaps,
                dim=dim
            )
            
            # Create thresholded cochain
            # Two values are "equal" if their distance <= threshold
            c0_data = {(k,): v for k, v in flat_data.items()}
            c0 = CechCochain(degree=0, data=c0_data)
            
            # Compute coboundary
            c1 = complex.coboundary(c0)
            
            # Threshold the coboundary: values below threshold become 0
            thresholded_c1_data = {}
            for simplex, value in c1.data.items():
                if np.linalg.norm(value) > threshold:
                    thresholded_c1_data[simplex] = value
            
            thresholded_c1 = CechCochain(degree=1, data=thresholded_c1_data)
            
            # Compute ranks
            h0_rank = 1 if thresholded_c1.is_zero() else 0
            h1_rank = 0 if thresholded_c1.is_zero() else len(thresholded_c1_data)
            
            cohomology_at_threshold[threshold] = {0: h0_rank, 1: h1_rank}
            
            # Track births and deaths
            for d in range(max_degree + 1):
                current_rank = cohomology_at_threshold[threshold].get(d, 0)
                
                if current_rank > prev_ranks[d]:
                    # New features born
                    for _ in range(current_rank - prev_ranks[d]):
                        feature_births[d].append((threshold, thresholded_c1 if d == 1 else None))
                
                elif current_rank < prev_ranks[d]:
                    # Features died
                    for _ in range(prev_ranks[d] - current_rank):
                        feature_deaths[d].append(threshold)
                
                prev_ranks[d] = current_rank
        
        # Build persistence diagrams
        diagrams = {}
        
        for d in range(max_degree + 1):
            diagram = PersistenceDiagram(dimension=d)
            
            births = feature_births[d]
            deaths = feature_deaths[d]
            
            # Match births to deaths (simple: FIFO)
            for i, (birth, rep) in enumerate(births):
                if i < len(deaths):
                    death = deaths[i]
                else:
                    death = float('inf')  # Never dies
                
                interval = PersistenceInterval(
                    birth=birth,
                    death=death,
                    dimension=d,
                    representative=rep
                )
                diagram.add_interval(interval)
            
            diagrams[d] = diagram
        
        return PersistentCohomologyResult(
            diagrams=diagrams,
            thresholds=thresholds,
            cohomology_at_threshold=cohomology_at_threshold
        )
    
    def compute_from_distance_matrix(
        self,
        names: List[str],
        distance_matrix: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        max_degree: int = 1
    ) -> PersistentCohomologyResult:
        """
        Compute persistent cohomology from a precomputed distance matrix.
        
        Useful when you have custom distances or want to avoid
        recomputing distances.
        
        Args:
            names: Names of the data points
            distance_matrix: n x n matrix of pairwise distances
            thresholds: Tolerance values
            max_degree: Maximum cohomology degree
        
        Returns:
            PersistentCohomologyResult
        """
        n = len(names)
        
        # Convert matrix to dictionary
        distances = {}
        for i in range(n):
            for j in range(i + 1, n):
                distances[(names[i], names[j])] = distance_matrix[i, j]
                distances[(names[j], names[i])] = distance_matrix[i, j]
        
        # Create dummy data (we only need distances)
        dummy_data = {name: np.array([0.0]) for name in names}
        
        # Override distance function
        original_fn = self.distance_fn
        self.distance_fn = lambda a, b: 0.0  # Not used
        
        # Manually set distances and compute
        # (This is a bit hacky, but avoids code duplication)
        
        if thresholds is None:
            max_dist = max(distances.values()) if distances else 1.0
            thresholds = np.linspace(0, max_dist * 1.5, 50)
        
        # Simplified computation using distances directly
        cohomology_at_threshold = {}
        feature_births = {d: [] for d in range(max_degree + 1)}
        feature_deaths = {d: [] for d in range(max_degree + 1)}
        prev_ranks = {d: 0 for d in range(max_degree + 1)}
        
        for threshold in thresholds:
            # Count edges above threshold
            edges_above = sum(1 for d in distances.values() if d > threshold) // 2
            
            # Simplified: H¹ rank ≈ number of inconsistent edges
            h0_rank = 1 if edges_above == 0 else 0
            h1_rank = edges_above
            
            cohomology_at_threshold[threshold] = {0: h0_rank, 1: h1_rank}
            
            for d in range(max_degree + 1):
                current_rank = cohomology_at_threshold[threshold].get(d, 0)
                
                if current_rank > prev_ranks[d]:
                    for _ in range(current_rank - prev_ranks[d]):
                        feature_births[d].append((threshold, None))
                elif current_rank < prev_ranks[d]:
                    for _ in range(prev_ranks[d] - current_rank):
                        feature_deaths[d].append(threshold)
                
                prev_ranks[d] = current_rank
        
        # Build diagrams
        diagrams = {}
        for d in range(max_degree + 1):
            diagram = PersistenceDiagram(dimension=d)
            
            births = feature_births[d]
            deaths = feature_deaths[d]
            
            for i, (birth, rep) in enumerate(births):
                death = deaths[i] if i < len(deaths) else float('inf')
                diagram.add_interval(PersistenceInterval(
                    birth=birth, death=death, dimension=d, representative=rep
                ))
            
            diagrams[d] = diagram
        
        self.distance_fn = original_fn
        
        return PersistentCohomologyResult(
            diagrams=diagrams,
            thresholds=thresholds,
            cohomology_at_threshold=cohomology_at_threshold
        )


# ==================== Convenience Functions ====================

def compute_persistent_cohomology(
    data: Dict[str, np.ndarray],
    num_thresholds: int = 50
) -> PersistentCohomologyResult:
    """
    Convenience function for persistent cohomology computation.
    
    Args:
        data: Dictionary mapping names to data vectors
        num_thresholds: Number of threshold values to use
    
    Returns:
        PersistentCohomologyResult
    
    Example:
        >>> data = {
        ...     'sensor_a': np.array([20.0, 30.0]),
        ...     'sensor_b': np.array([20.1, 30.2]),
        ...     'sensor_c': np.array([21.0, 29.5]),
        ... }
        >>> result = compute_persistent_cohomology(data)
        >>> print(result.summary())
    """
    pc = PersistentCohomology()
    
    # Auto-determine threshold range
    flat_data = {k: v.flatten() for k, v in data.items()}
    names = list(flat_data.keys())
    
    max_dist = 0.0
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            d = np.linalg.norm(flat_data[n1] - flat_data[n2])
            max_dist = max(max_dist, d)
    
    thresholds = np.linspace(0, max_dist * 1.2, num_thresholds)
    
    return pc.compute(data, thresholds=thresholds)


def persistence_based_consistency(
    data: Dict[str, np.ndarray],
    noise_threshold: float = 0.1
) -> Tuple[bool, float, str]:
    """
    Check consistency using persistence to filter noise.
    
    Args:
        data: Dictionary mapping names to data vectors
        noise_threshold: Maximum persistence to consider as noise
    
    Returns:
        Tuple of (is_consistent, confidence, explanation)
    
    Example:
        >>> is_consistent, confidence, explanation = persistence_based_consistency(
        ...     {'a': np.array([1.0]), 'b': np.array([1.05]), 'c': np.array([1.1])}
        ... )
        >>> print(f"Consistent: {is_consistent} (confidence: {confidence:.1%})")
    """
    result = compute_persistent_cohomology(data)
    
    h1_diagram = result.get_diagram(1)
    
    # Filter out noise
    significant = h1_diagram.filter_by_persistence(noise_threshold)
    
    is_consistent = len(significant.intervals) == 0
    
    # Confidence based on how clearly we can distinguish signal from noise
    if not h1_diagram.intervals:
        confidence = 1.0
        explanation = "No inconsistencies detected at any threshold"
    else:
        max_pers = h1_diagram.max_persistence()
        if max_pers < noise_threshold:
            confidence = 1.0 - max_pers / noise_threshold
            explanation = f"All inconsistencies below noise threshold ({max_pers:.3f} < {noise_threshold})"
        else:
            confidence = noise_threshold / max_pers
            explanation = f"Significant inconsistency detected (persistence = {max_pers:.3f})"
    
    return is_consistent, confidence, explanation
