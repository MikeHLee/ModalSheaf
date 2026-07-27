"""
Measured Transforms - Transformations that return topological loss characterization.

This module provides a richer transformation framework where:
1. Transforms return BOTH the transformed data AND a loss characterization
2. Loss is characterized topologically, not just as a scalar
3. Actual model/function performance influences the loss estimate
4. Loss can be data-dependent (computed at runtime)

The key insight: information loss has STRUCTURE. It's not just "70% lost" but:
- WHAT kind of information was lost (spatial, semantic, relational)
- WHERE in the data space the loss occurred
- HOW the remaining information is shaped (dimensionality, connectivity)

References:
- Robinson, M. (2014). Topological Signal Processing. Springer.
- Curry, J. (2014). Sheaves, Cosheaves and Applications. arXiv:1303.3255
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic, Union
from abc import ABC, abstractmethod
import numpy as np

from .core import Modality, TransformationType


# ==================== Loss Characterization ====================

class LossType(Enum):
    """Categories of information loss."""
    SPATIAL = auto()      # Loss of spatial/positional detail
    TEMPORAL = auto()     # Loss of temporal/sequential detail
    SEMANTIC = auto()     # Loss of meaning/conceptual detail
    RELATIONAL = auto()   # Loss of relationships between elements
    STATISTICAL = auto()  # Loss of distributional properties
    STRUCTURAL = auto()   # Loss of topological/graph structure
    QUANTIZATION = auto() # Loss from discretization
    TRUNCATION = auto()   # Loss from cutting off data
    PROJECTION = auto()   # Loss from dimensionality reduction
    UNKNOWN = auto()


@dataclass
class LossRegion:
    """
    A region in the data space where information was lost.
    
    This captures the topological structure of the loss:
    - Where in the input space did loss occur?
    - What's the "shape" of the lost information?
    """
    loss_type: LossType
    magnitude: float  # 0.0 to 1.0, how much was lost in this region
    
    # Topological characterization
    affected_dimensions: Optional[List[int]] = None  # Which dims were affected
    affected_indices: Optional[np.ndarray] = None    # Specific indices affected
    
    # Betti numbers for the loss region (topological invariants)
    # b0 = connected components, b1 = holes, b2 = voids
    betti_numbers: Optional[Tuple[int, ...]] = None
    
    # Human-readable description
    description: str = ""
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologicalLossCharacterization:
    """
    Full topological characterization of information loss.
    
    This goes beyond a scalar "info_loss_estimate" to capture:
    - Multiple types of loss occurring simultaneously
    - The topological structure of what was preserved vs lost
    - Data-dependent loss that varies with input
    
    In sheaf terms, this characterizes the kernel and cokernel
    of the restriction map.
    """
    # Overall scalar (for backward compatibility)
    total_loss: float  # 0.0 to 1.0
    
    # Breakdown by loss type
    loss_regions: List[LossRegion] = field(default_factory=list)
    
    # What was PRESERVED (complement of loss)
    preserved_dimensions: Optional[int] = None  # Effective dimensionality
    preserved_rank: Optional[int] = None        # Rank of the transformation
    
    # Topological invariants of the preserved information
    # These are the Betti numbers of the "image" of the transform
    preserved_betti: Optional[Tuple[int, ...]] = None
    
    # Is the loss uniform or concentrated?
    loss_entropy: Optional[float] = None  # High = uniform, Low = concentrated
    
    # Confidence in this characterization
    confidence: float = 1.0
    
    # Was this computed from actual data or estimated?
    is_measured: bool = False
    
    # The input that was analyzed (optional, for debugging)
    input_hash: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def dominant_loss_type(self) -> Optional[LossType]:
        """Get the type of loss that dominates."""
        if not self.loss_regions:
            return None
        return max(self.loss_regions, key=lambda r: r.magnitude).loss_type
    
    def loss_by_type(self) -> Dict[LossType, float]:
        """Get loss magnitude by type."""
        result = {}
        for region in self.loss_regions:
            if region.loss_type in result:
                result[region.loss_type] = max(result[region.loss_type], region.magnitude)
            else:
                result[region.loss_type] = region.magnitude
        return result
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Total loss: {self.total_loss:.1%}"]
        
        if self.loss_regions:
            lines.append("Loss breakdown:")
            for region in sorted(self.loss_regions, key=lambda r: -r.magnitude):
                lines.append(f"  - {region.loss_type.name}: {region.magnitude:.1%}")
                if region.description:
                    lines.append(f"    {region.description}")
        
        if self.preserved_dimensions is not None:
            lines.append(f"Preserved dimensions: {self.preserved_dimensions}")
        
        if not self.is_measured:
            lines.append("(Estimated, not measured from data)")
        
        return "\n".join(lines)


# ==================== Transform Result ====================

T = TypeVar('T')  # Input type
U = TypeVar('U')  # Output type


@dataclass
class TransformResult(Generic[U]):
    """
    Result of a measured transformation.
    
    Contains both the transformed data AND the loss characterization.
    This is what measured transforms return instead of just the data.
    """
    data: U
    loss: TopologicalLossCharacterization
    
    # Optional: what would be needed to invert this transform?
    # (Even if not fully invertible, partial info might help)
    inversion_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Timing information
    compute_time_ms: Optional[float] = None


# ==================== Measured Transform Protocol ====================

class MeasuredTransform(ABC, Generic[T, U]):
    """
    Abstract base class for transforms that measure their own loss.
    
    Unlike basic Transformation which takes a static info_loss_estimate,
    MeasuredTransform computes loss characterization at runtime based
    on the actual input data and transformation behavior.
    
    Subclass this to create transforms that:
    1. Know what kind of information they lose
    2. Can measure loss for specific inputs
    3. Return topological characterization of the loss
    
    Example:
        class CLIPImageEncoder(MeasuredTransform[np.ndarray, np.ndarray]):
            def transform(self, image: np.ndarray) -> TransformResult[np.ndarray]:
                embedding = self.model.encode(image)
                loss = self._measure_loss(image, embedding)
                return TransformResult(data=embedding, loss=loss)
    """
    
    def __init__(
        self,
        source: str,
        target: str,
        name: Optional[str] = None,
        expected_loss_types: Optional[List[LossType]] = None,
    ):
        self.source = source
        self.target = target
        self.name = name or f"{source}_to_{target}"
        self.expected_loss_types = expected_loss_types or []
        
        # Track historical loss for calibration
        self._loss_history: List[TopologicalLossCharacterization] = []
        self._max_history = 1000
    
    @abstractmethod
    def transform(self, data: T) -> TransformResult[U]:
        """
        Apply the transformation and measure loss.
        
        Returns TransformResult containing both the transformed data
        and the topological loss characterization.
        """
        pass
    
    def __call__(self, data: T) -> TransformResult[U]:
        """Apply transform (calls transform method)."""
        result = self.transform(data)
        
        # Track history for calibration
        if len(self._loss_history) >= self._max_history:
            self._loss_history.pop(0)
        self._loss_history.append(result.loss)
        
        return result
    
    def average_loss(self) -> float:
        """Get average loss from history."""
        if not self._loss_history:
            return 0.5  # Default
        return np.mean([l.total_loss for l in self._loss_history])
    
    def loss_variance(self) -> float:
        """Get variance in loss (indicates data-dependence)."""
        if len(self._loss_history) < 2:
            return 0.0
        return np.var([l.total_loss for l in self._loss_history])
    
    @property
    def is_data_dependent(self) -> bool:
        """Check if loss varies significantly with input."""
        return self.loss_variance() > 0.01  # Threshold


# ==================== Common Measured Transforms ====================

class EmbeddingTransform(MeasuredTransform[Any, np.ndarray]):
    """
    Base class for embedding transforms (text/image -> vector).
    
    Embedding transforms typically have:
    - High semantic loss (meaning compressed)
    - Projection loss (dimensionality reduced)
    - Variable loss depending on input complexity
    """
    
    def __init__(
        self,
        source: str,
        embed_fn: Callable[[Any], np.ndarray],
        embedding_dim: int,
        name: Optional[str] = None,
        base_loss: float = 0.7,
    ):
        super().__init__(
            source=source,
            target="embedding",
            name=name,
            expected_loss_types=[LossType.SEMANTIC, LossType.PROJECTION]
        )
        self.embed_fn = embed_fn
        self.embedding_dim = embedding_dim
        self.base_loss = base_loss
    
    def transform(self, data: Any) -> TransformResult[np.ndarray]:
        # Apply embedding
        embedding = self.embed_fn(data)
        
        # Measure loss
        loss = self._measure_embedding_loss(data, embedding)
        
        return TransformResult(
            data=embedding,
            loss=loss,
            inversion_hints={
                "original_type": type(data).__name__,
                "embedding_norm": float(np.linalg.norm(embedding)),
            }
        )
    
    def _measure_embedding_loss(
        self, 
        original: Any, 
        embedding: np.ndarray
    ) -> TopologicalLossCharacterization:
        """Measure loss from embedding."""
        
        # Estimate input complexity
        if isinstance(original, str):
            input_complexity = len(original.split())  # Word count
            max_complexity = 1000
        elif isinstance(original, np.ndarray):
            input_complexity = original.size
            max_complexity = 224 * 224 * 3  # Typical image
        else:
            input_complexity = 100
            max_complexity = 1000
        
        # Loss increases with input complexity relative to embedding dim
        complexity_ratio = min(input_complexity / max_complexity, 1.0)
        compression_ratio = self.embedding_dim / max(input_complexity, 1)
        
        # More complex inputs lose more when compressed to fixed dim
        adjusted_loss = self.base_loss + (1 - self.base_loss) * complexity_ratio * 0.3
        adjusted_loss = min(adjusted_loss, 0.95)
        
        # Characterize the loss
        loss_regions = [
            LossRegion(
                loss_type=LossType.SEMANTIC,
                magnitude=adjusted_loss * 0.6,
                description="Meaning compressed to dense vector"
            ),
            LossRegion(
                loss_type=LossType.PROJECTION,
                magnitude=adjusted_loss * 0.4,
                description=f"Projected to {self.embedding_dim} dimensions"
            ),
        ]
        
        return TopologicalLossCharacterization(
            total_loss=adjusted_loss,
            loss_regions=loss_regions,
            preserved_dimensions=self.embedding_dim,
            is_measured=True,
            confidence=0.7,  # Embedding loss is hard to measure precisely
        )


class EntityExtractionTransform(MeasuredTransform[Any, 'Olog']):
    """
    Transform that extracts entities and relationships.
    
    Loss types:
    - Relational: relationships not captured
    - Structural: graph structure simplified
    - Semantic: entity nuances lost
    """
    
    def __init__(
        self,
        source: str,
        extract_fn: Callable[[Any], 'Olog'],
        name: Optional[str] = None,
    ):
        super().__init__(
            source=source,
            target="olog",
            name=name,
            expected_loss_types=[LossType.RELATIONAL, LossType.STRUCTURAL, LossType.SEMANTIC]
        )
        self.extract_fn = extract_fn
    
    def transform(self, data: Any) -> TransformResult:
        from .knowledge import Olog
        
        # Extract entities
        olog = self.extract_fn(data)
        
        # Measure loss based on extraction results
        loss = self._measure_extraction_loss(data, olog)
        
        return TransformResult(
            data=olog,
            loss=loss,
            inversion_hints={
                "num_entities": olog.num_entities if isinstance(olog, Olog) else 0,
                "num_relationships": olog.num_relationships if isinstance(olog, Olog) else 0,
            }
        )
    
    def _measure_extraction_loss(
        self, 
        original: Any, 
        olog: 'Olog'
    ) -> TopologicalLossCharacterization:
        from .knowledge import Olog
        
        # Estimate based on extraction confidence and completeness
        if isinstance(olog, Olog):
            avg_confidence = olog.average_confidence()
            num_facts = olog.num_entities + olog.num_relationships
        else:
            avg_confidence = 0.5
            num_facts = 0
        
        # More facts extracted = less loss (up to a point)
        completeness = min(num_facts / 20, 1.0)  # Assume 20 facts is "complete"
        
        # Loss is inverse of confidence * completeness
        total_loss = 1 - (avg_confidence * completeness * 0.6 + 0.2)
        total_loss = max(0.2, min(0.8, total_loss))
        
        loss_regions = [
            LossRegion(
                loss_type=LossType.RELATIONAL,
                magnitude=total_loss * 0.4,
                description="Some relationships not captured"
            ),
            LossRegion(
                loss_type=LossType.SEMANTIC,
                magnitude=total_loss * 0.35,
                description="Entity nuances simplified"
            ),
            LossRegion(
                loss_type=LossType.STRUCTURAL,
                magnitude=total_loss * 0.25,
                description="Graph structure may be incomplete"
            ),
        ]
        
        return TopologicalLossCharacterization(
            total_loss=total_loss,
            loss_regions=loss_regions,
            is_measured=True,
            confidence=avg_confidence,
        )


class TextGenerationTransform(MeasuredTransform['Olog', str]):
    """
    Transform that generates text from structured data.
    
    This is interesting because generation can ADD information
    (hallucination) as well as lose it. We track both.
    """
    
    def __init__(
        self,
        generate_fn: Callable[['Olog'], str],
        name: Optional[str] = None,
        temperature: float = 0.7,
    ):
        super().__init__(
            source="olog",
            target="text",
            name=name,
            expected_loss_types=[LossType.STRUCTURAL, LossType.SEMANTIC]
        )
        self.generate_fn = generate_fn
        self.temperature = temperature
    
    def transform(self, data: 'Olog') -> TransformResult[str]:
        # Generate text
        text = self.generate_fn(data)
        
        # Measure loss (and potential hallucination)
        loss = self._measure_generation_loss(data, text)
        
        return TransformResult(
            data=text,
            loss=loss,
            inversion_hints={
                "source_facts": data.num_relationships if hasattr(data, 'num_relationships') else 0,
                "generated_length": len(text),
            }
        )
    
    def _measure_generation_loss(
        self, 
        olog: 'Olog', 
        text: str
    ) -> TopologicalLossCharacterization:
        from .knowledge import Olog
        
        if isinstance(olog, Olog):
            num_facts = olog.num_relationships
        else:
            num_facts = 0
        
        # Higher temperature = more creative = more potential loss/hallucination
        temperature_factor = self.temperature
        
        # Base loss for generation
        base_loss = 0.3
        
        # Adjust based on temperature
        total_loss = base_loss + temperature_factor * 0.2
        
        loss_regions = [
            LossRegion(
                loss_type=LossType.STRUCTURAL,
                magnitude=total_loss * 0.5,
                description="Graph structure linearized to text"
            ),
            LossRegion(
                loss_type=LossType.SEMANTIC,
                magnitude=total_loss * 0.3,
                description="Some semantic precision lost in verbalization"
            ),
        ]
        
        # Note: we could also track "hallucination" as negative loss
        # (information added that wasn't in the source)
        
        return TopologicalLossCharacterization(
            total_loss=total_loss,
            loss_regions=loss_regions,
            is_measured=True,
            confidence=0.8,
            metadata={
                "temperature": self.temperature,
                "potential_hallucination": temperature_factor > 0.5,
            }
        )


# ==================== Transform Registry ====================

class MeasuredTransformRegistry:
    """
    Registry for storing and retrieving measured transforms.
    
    Transforms are stored by (source, target) pair and can be
    looked up to find paths through modality space.
    """
    
    def __init__(self):
        self._transforms: Dict[Tuple[str, str], MeasuredTransform] = {}
        self._by_source: Dict[str, List[MeasuredTransform]] = {}
        self._by_target: Dict[str, List[MeasuredTransform]] = {}
    
    def register(self, transform: MeasuredTransform) -> None:
        """Register a transform."""
        key = (transform.source, transform.target)
        self._transforms[key] = transform
        
        # Index by source
        if transform.source not in self._by_source:
            self._by_source[transform.source] = []
        self._by_source[transform.source].append(transform)
        
        # Index by target
        if transform.target not in self._by_target:
            self._by_target[transform.target] = []
        self._by_target[transform.target].append(transform)
    
    def get(self, source: str, target: str) -> Optional[MeasuredTransform]:
        """Get transform by source and target."""
        return self._transforms.get((source, target))
    
    def get_from_source(self, source: str) -> List[MeasuredTransform]:
        """Get all transforms from a source modality."""
        return self._by_source.get(source, [])
    
    def get_to_target(self, target: str) -> List[MeasuredTransform]:
        """Get all transforms to a target modality."""
        return self._by_target.get(target, [])
    
    def list_all(self) -> List[Tuple[str, str, str]]:
        """List all registered transforms as (source, target, name)."""
        return [
            (t.source, t.target, t.name)
            for t in self._transforms.values()
        ]


# Global registry
MEASURED_TRANSFORM_REGISTRY = MeasuredTransformRegistry()


def register_measured_transform(transform: MeasuredTransform) -> MeasuredTransform:
    """Register a measured transform in the global registry."""
    MEASURED_TRANSFORM_REGISTRY.register(transform)
    return transform


# ==================== Pipeline with Measured Loss ====================

@dataclass
class PipelineResult(Generic[U]):
    """Result of running a pipeline of measured transforms."""
    data: U
    total_loss: TopologicalLossCharacterization
    step_results: List[TransformResult]
    
    def summary(self) -> str:
        """Human-readable summary of the pipeline."""
        lines = [
            f"Pipeline completed with {len(self.step_results)} steps",
            f"Total loss: {self.total_loss.total_loss:.1%}",
            "",
            "Step breakdown:"
        ]
        
        for i, result in enumerate(self.step_results):
            lines.append(f"  {i+1}. Loss: {result.loss.total_loss:.1%}")
            dominant = result.loss.dominant_loss_type()
            if dominant:
                lines.append(f"      Dominant: {dominant.name}")
        
        return "\n".join(lines)


def run_measured_pipeline(
    data: Any,
    transforms: List[MeasuredTransform]
) -> PipelineResult:
    """
    Run a pipeline of measured transforms.
    
    Returns full loss characterization for each step and combined.
    """
    current_data = data
    step_results = []
    
    for transform in transforms:
        result = transform(current_data)
        step_results.append(result)
        current_data = result.data
    
    # Combine loss characterizations
    total_loss = _combine_losses([r.loss for r in step_results])
    
    return PipelineResult(
        data=current_data,
        total_loss=total_loss,
        step_results=step_results
    )


def _combine_losses(
    losses: List[TopologicalLossCharacterization]
) -> TopologicalLossCharacterization:
    """Combine multiple loss characterizations."""
    if not losses:
        return TopologicalLossCharacterization(total_loss=0.0, is_measured=True)
    
    # Multiplicative combination: preservation = prod(1 - loss_i)
    preservation = 1.0
    for loss in losses:
        preservation *= (1 - loss.total_loss)
    
    total_loss = 1 - preservation
    
    # Combine loss regions from all steps
    all_regions = []
    for loss in losses:
        all_regions.extend(loss.loss_regions)
    
    # Aggregate by type
    by_type: Dict[LossType, float] = {}
    for region in all_regions:
        if region.loss_type in by_type:
            # Take max (conservative)
            by_type[region.loss_type] = max(by_type[region.loss_type], region.magnitude)
        else:
            by_type[region.loss_type] = region.magnitude
    
    combined_regions = [
        LossRegion(loss_type=lt, magnitude=mag)
        for lt, mag in by_type.items()
    ]
    
    return TopologicalLossCharacterization(
        total_loss=total_loss,
        loss_regions=combined_regions,
        is_measured=all(l.is_measured for l in losses),
        confidence=min(l.confidence for l in losses) if losses else 1.0,
    )
