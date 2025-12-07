"""
Consistency checking and cohomology computation for multimodal data.

This module implements the sheaf-theoretic consistency analysis:
- H⁰: Global sections (consistent states)
- H¹: Obstructions (inconsistencies)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg

from .core import Modality, Transformation
from .graph import ModalityGraph


@dataclass
class CohomologyResult:
    """
    Result of cohomology computation.
    
    Attributes:
        h0_dim: Dimension of H⁰ (global sections)
        h1_dim: Dimension of H¹ (obstructions)
        h0_basis: Basis vectors for H⁰ (if computed)
        h1_representatives: Representatives of H¹ classes
        consistency_score: Overall consistency (1.0 = perfect, 0.0 = total disagreement)
        diagnosis: Human-readable interpretation
        details: Additional computation details
    """
    h0_dim: int
    h1_dim: int
    h0_basis: Optional[np.ndarray] = None
    h1_representatives: Optional[np.ndarray] = None
    consistency_score: float = 1.0
    diagnosis: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        
        # Generate diagnosis if not provided
        if not self.diagnosis:
            if self.h1_dim == 0:
                self.diagnosis = "Fully consistent: all modalities agree"
            elif self.consistency_score > 0.9:
                self.diagnosis = "Minor inconsistencies detected"
            elif self.consistency_score > 0.5:
                self.diagnosis = "Moderate inconsistencies: some modalities disagree"
            else:
                self.diagnosis = "Major inconsistencies: significant disagreement between modalities"
    
    @property
    def is_consistent(self) -> bool:
        """Check if data is fully consistent (H¹ = 0)."""
        return self.h1_dim == 0
    
    def __repr__(self) -> str:
        return (
            f"CohomologyResult(H⁰={self.h0_dim}, H¹={self.h1_dim}, "
            f"consistency={self.consistency_score:.2f}, "
            f"diagnosis='{self.diagnosis}')"
        )


class ConsistencyChecker:
    """
    Checks consistency of multimodal data using sheaf cohomology.
    
    Given data from multiple modalities, this class:
    1. Transforms all data to a common embedding space
    2. Computes pairwise distances/agreements
    3. Calculates cohomological invariants (H⁰, H¹)
    4. Provides interpretable diagnostics
    
    Example:
        >>> checker = ConsistencyChecker(graph, common_modality="embedding")
        >>> result = checker.check({
        ...     "image": my_image,
        ...     "text": my_caption,
        ...     "audio": my_audio
        ... })
        >>> print(result.diagnosis)
        "Minor inconsistencies detected"
    """
    
    def __init__(
        self,
        graph: ModalityGraph,
        common_modality: str = "embedding",
        distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        consistency_threshold: float = 0.1
    ):
        """
        Initialize the consistency checker.
        
        Args:
            graph: The modality graph with transformations
            common_modality: Modality to use as common comparison space
            distance_fn: Function to compute distance between embeddings
                        (default: cosine distance)
            consistency_threshold: Distance below which data is considered consistent
        """
        self.graph = graph
        self.common_modality = common_modality
        self.consistency_threshold = consistency_threshold
        
        if distance_fn is None:
            self.distance_fn = self._cosine_distance
        else:
            self.distance_fn = distance_fn
    
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two vectors."""
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0  # Maximum distance if either is zero
        
        similarity = np.dot(a, b) / (norm_a * norm_b)
        return 1.0 - similarity
    
    def check(
        self,
        data: Dict[str, Any],
        return_embeddings: bool = False
    ) -> CohomologyResult:
        """
        Check consistency of multimodal data.
        
        Args:
            data: Dictionary mapping modality names to data
            return_embeddings: If True, include embeddings in result details
        
        Returns:
            CohomologyResult with consistency analysis
        
        Example:
            >>> result = checker.check({
            ...     "image": image_array,
            ...     "text": "a photo of a cat"
            ... })
        """
        modalities = list(data.keys())
        n = len(modalities)
        
        if n < 2:
            return CohomologyResult(
                h0_dim=1,
                h1_dim=0,
                consistency_score=1.0,
                diagnosis="Single modality: trivially consistent"
            )
        
        # Transform all data to common modality
        embeddings = {}
        for mod_name, mod_data in data.items():
            if mod_name == self.common_modality:
                embeddings[mod_name] = np.asarray(mod_data)
            elif self.graph.has_path(mod_name, self.common_modality):
                transformed = self.graph.transform(
                    mod_name, self.common_modality, mod_data
                )
                embeddings[mod_name] = np.asarray(transformed)
            else:
                raise ValueError(
                    f"No path from '{mod_name}' to '{self.common_modality}'"
                )
        
        # Compute pairwise distances
        distance_matrix = np.zeros((n, n))
        for i, mod_i in enumerate(modalities):
            for j, mod_j in enumerate(modalities):
                if i < j:
                    dist = self.distance_fn(embeddings[mod_i], embeddings[mod_j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        
        # Compute cohomology
        result = self._compute_cohomology(
            modalities, embeddings, distance_matrix
        )
        
        if return_embeddings:
            result.details["embeddings"] = embeddings
        
        result.details["distance_matrix"] = distance_matrix
        result.details["modalities"] = modalities
        
        return result
    
    def _compute_cohomology(
        self,
        modalities: List[str],
        embeddings: Dict[str, np.ndarray],
        distance_matrix: np.ndarray
    ) -> CohomologyResult:
        """
        Compute sheaf cohomology from embeddings and distances.
        
        For a simple graph (all modalities connected to common space):
        - H⁰ = consensus embedding (average)
        - H¹ = measures deviation from consensus
        
        This is a simplified computation suitable for practical use.
        For full sheaf cohomology on complex graphs, use the advanced API.
        """
        n = len(modalities)
        
        # Stack embeddings
        emb_list = [embeddings[m] for m in modalities]
        emb_matrix = np.vstack([e.flatten() for e in emb_list])
        
        # H⁰: Consensus (mean embedding)
        consensus = np.mean(emb_matrix, axis=0)
        
        # Compute deviations from consensus
        deviations = emb_matrix - consensus
        deviation_norms = np.linalg.norm(deviations, axis=1)
        
        # H¹: Total deviation (simplified)
        # In full sheaf cohomology, this would be ker(δ¹)/im(δ⁰)
        # Here we approximate with total deviation around the "loop"
        total_deviation = np.sum(deviation_norms)
        max_deviation = np.max(deviation_norms)
        
        # Compute consistency score
        # Based on average pairwise distance
        avg_distance = np.sum(distance_matrix) / (n * (n - 1)) if n > 1 else 0
        consistency_score = max(0.0, 1.0 - avg_distance)
        
        # Estimate H¹ dimension
        # Count how many modalities deviate significantly
        significant_deviations = np.sum(
            deviation_norms > self.consistency_threshold * np.mean(deviation_norms)
        )
        h1_dim = max(0, significant_deviations - 1)  # -1 because one deviation is "free"
        
        return CohomologyResult(
            h0_dim=1,  # Always 1 for connected graph
            h1_dim=h1_dim,
            h0_basis=consensus.reshape(1, -1),
            h1_representatives=deviations if h1_dim > 0 else None,
            consistency_score=consistency_score,
            details={
                "deviation_norms": deviation_norms,
                "total_deviation": total_deviation,
                "max_deviation": max_deviation,
                "avg_pairwise_distance": avg_distance,
            }
        )
    
    def diagnose_inconsistency(
        self,
        result: CohomologyResult
    ) -> Dict[str, Any]:
        """
        Provide detailed diagnosis of inconsistencies.
        
        Args:
            result: CohomologyResult from check()
        
        Returns:
            Dictionary with detailed diagnosis
        """
        if result.is_consistent:
            return {
                "status": "consistent",
                "message": "All modalities agree",
                "problematic_modalities": [],
                "recommendations": []
            }
        
        modalities = result.details.get("modalities", [])
        deviation_norms = result.details.get("deviation_norms", [])
        
        # Find most problematic modalities
        if len(deviation_norms) > 0:
            sorted_indices = np.argsort(deviation_norms)[::-1]
            problematic = [
                (modalities[i], float(deviation_norms[i]))
                for i in sorted_indices
                if deviation_norms[i] > np.mean(deviation_norms)
            ]
        else:
            problematic = []
        
        recommendations = []
        if result.consistency_score < 0.5:
            recommendations.append(
                "Consider checking for data quality issues or misaligned inputs"
            )
        if len(problematic) == 1:
            recommendations.append(
                f"Modality '{problematic[0][0]}' is the main source of inconsistency"
            )
        
        return {
            "status": "inconsistent",
            "message": result.diagnosis,
            "problematic_modalities": problematic,
            "recommendations": recommendations,
            "h1_dimension": result.h1_dim,
            "consistency_score": result.consistency_score,
        }


# ==================== Advanced Cohomology ====================

def compute_sheaf_laplacian(
    graph: ModalityGraph,
    embeddings: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute the sheaf Laplacian matrix.
    
    The sheaf Laplacian generalizes the graph Laplacian to account
    for the restriction maps (transformations) between modalities.
    
    ker(L) = H⁰ (global sections)
    
    Args:
        graph: The modality graph
        embeddings: Current embeddings at each modality
    
    Returns:
        The sheaf Laplacian matrix
    """
    modalities = graph.modalities
    n = len(modalities)
    
    # Get embedding dimension (assume all same for now)
    d = len(next(iter(embeddings.values())).flatten())
    
    # Build block Laplacian
    L = np.zeros((n * d, n * d))
    
    for i, mod_i in enumerate(modalities):
        for j, mod_j in enumerate(modalities):
            if i == j:
                # Diagonal: sum of incident edge contributions
                degree = 0
                for k, mod_k in enumerate(modalities):
                    if graph.get_transformation(mod_i, mod_k) is not None:
                        degree += 1
                L[i*d:(i+1)*d, i*d:(i+1)*d] = degree * np.eye(d)
            else:
                # Off-diagonal: -R^T R for restriction map R
                transform = graph.get_transformation(mod_i, mod_j)
                if transform is not None:
                    # For linear transforms, we'd use the matrix
                    # For now, approximate with identity
                    L[i*d:(i+1)*d, j*d:(j+1)*d] = -np.eye(d)
    
    return L


def diffuse_to_consensus(
    graph: ModalityGraph,
    embeddings: Dict[str, np.ndarray],
    num_steps: int = 10,
    step_size: float = 0.1
) -> Dict[str, np.ndarray]:
    """
    Diffuse embeddings toward consensus using sheaf Laplacian.
    
    This implements the heat equation on the sheaf:
        dx/dt = -L x
    
    The steady state is in ker(L) = H⁰ (global sections).
    
    Args:
        graph: The modality graph
        embeddings: Initial embeddings
        num_steps: Number of diffusion steps
        step_size: Step size for Euler integration
    
    Returns:
        Diffused embeddings (closer to consensus)
    """
    modalities = graph.modalities
    n = len(modalities)
    d = len(next(iter(embeddings.values())).flatten())
    
    # Stack into single vector
    x = np.concatenate([embeddings[m].flatten() for m in modalities])
    
    # Compute Laplacian
    L = compute_sheaf_laplacian(graph, embeddings)
    
    # Diffuse
    for _ in range(num_steps):
        x = x - step_size * L @ x
    
    # Unstack
    result = {}
    for i, mod in enumerate(modalities):
        result[mod] = x[i*d:(i+1)*d].reshape(embeddings[mod].shape)
    
    return result
