"""
Core data structures for ModalSheaf.

Defines Modality, Transformation, and their properties.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class TransformationType(Enum):
    """
    Classification of transformation types by information preservation.
    
    This corresponds to mathematical morphism types:
    - ISOMORPHISM: Bijective, fully invertible (no information loss)
    - EMBEDDING: Injective, invertible on image (information preserved but encoded)
    - PROJECTION: Surjective, dimension reduction (information reduced)
    - LOSSY: Neither injective nor surjective (information lost)
    """
    ISOMORPHISM = auto()   # Fully reversible, no info loss
    EMBEDDING = auto()     # One-way but info preserved (injective)
    PROJECTION = auto()    # Dimension reduction (surjective)
    LOSSY = auto()         # Information lost, not invertible
    UNKNOWN = auto()       # Not yet characterized


@dataclass
class Modality:
    """
    Represents a data modality (e.g., image, text, audio, embedding).
    
    In sheaf terms, this is a "stalk" - the data space associated with
    a point in the base topology.
    
    Attributes:
        name: Unique identifier for this modality
        shape: Expected shape of data (None for variable-length)
        dtype: Data type (numpy dtype string or 'str', 'object')
        description: Human-readable description
        metadata: Additional properties
    
    Examples:
        >>> image_mod = Modality("image", shape=(224, 224, 3), dtype="float32")
        >>> text_mod = Modality("text", shape=None, dtype="str")
        >>> embedding_mod = Modality("clip_embedding", shape=(768,), dtype="float32")
    """
    name: str
    shape: Optional[Tuple[int, ...]] = None
    dtype: str = "float32"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Modality):
            return self.name == other.name
        return self.name == other
    
    @property
    def dimensionality(self) -> Optional[int]:
        """Total number of dimensions (product of shape), or None if variable."""
        if self.shape is None:
            return None
        return int(np.prod(self.shape))
    
    def validate(self, data: Any) -> bool:
        """Check if data matches this modality's expected format."""
        if self.shape is None:
            return True  # Variable shape, accept anything
        
        if hasattr(data, 'shape'):
            return data.shape == self.shape
        
        return True  # Can't validate, assume ok


@dataclass
class Transformation:
    """
    Represents a transformation between modalities (a restriction map).
    
    In sheaf terms, this is a morphism between stalks that defines
    how data "restricts" from one modality to another.
    
    Attributes:
        source: Source modality name
        target: Target modality name
        forward: Forward transformation function
        inverse: Inverse transformation (None if not invertible)
        transform_type: Classification of information preservation
        info_loss_estimate: Estimated information loss (0.0 = none, 1.0 = total)
        name: Optional name for this transformation
        metadata: Additional properties
    
    Examples:
        >>> # CLIP image encoder (lossy projection)
        >>> img_to_emb = Transformation(
        ...     source="image",
        ...     target="embedding",
        ...     forward=clip_encode_image,
        ...     inverse=None,
        ...     transform_type=TransformationType.LOSSY,
        ...     info_loss_estimate=0.8
        ... )
        
        >>> # Lossless format conversion (isomorphism)
        >>> png_to_numpy = Transformation(
        ...     source="png_image",
        ...     target="numpy_image",
        ...     forward=png_to_array,
        ...     inverse=array_to_png,
        ...     transform_type=TransformationType.ISOMORPHISM,
        ...     info_loss_estimate=0.0
        ... )
    """
    source: str
    target: str
    forward: Callable[[Any], Any]
    inverse: Optional[Callable[[Any], Any]] = None
    transform_type: TransformationType = TransformationType.UNKNOWN
    info_loss_estimate: float = 0.5  # 0.0 = no loss, 1.0 = total loss
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    matrix: Optional[np.ndarray] = None  # Linear restriction map matrix R for sheaf Laplacian
    
    def __post_init__(self):
        if self.name is None:
            self.name = f"{self.source}_to_{self.target}"
        
        # Auto-detect transform type from inverse availability
        if self.transform_type == TransformationType.UNKNOWN:
            if self.inverse is not None:
                self.transform_type = TransformationType.ISOMORPHISM
            else:
                self.transform_type = TransformationType.LOSSY
    
    def __call__(self, data: Any) -> Any:
        """Apply the forward transformation."""
        return self.forward(data)
    
    def apply_inverse(self, data: Any) -> Any:
        """Apply the inverse transformation if available."""
        if self.inverse is None:
            raise ValueError(
                f"Transformation {self.name} is not invertible "
                f"(type: {self.transform_type.name})"
            )
        return self.inverse(data)
    
    @property
    def is_invertible(self) -> bool:
        """Check if this transformation has an inverse."""
        return self.inverse is not None
    
    @property
    def is_lossless(self) -> bool:
        """Check if this transformation preserves all information."""
        return self.transform_type in (
            TransformationType.ISOMORPHISM,
            TransformationType.EMBEDDING
        )


def compose_transformations(
    t1: Transformation, 
    t2: Transformation
) -> Transformation:
    """
    Compose two transformations: t1 then t2.
    
    The result maps from t1.source to t2.target.
    Information loss accumulates (conservatively estimated).
    
    Args:
        t1: First transformation (applied first)
        t2: Second transformation (applied second)
    
    Returns:
        New Transformation representing the composition
    
    Raises:
        ValueError: If t1.target != t2.source
    """
    if t1.target != t2.source:
        raise ValueError(
            f"Cannot compose: {t1.name} outputs to '{t1.target}' "
            f"but {t2.name} expects '{t2.source}'"
        )
    
    # Compose forward functions
    def composed_forward(data):
        return t2.forward(t1.forward(data))
    
    # Compose inverse functions (if both exist)
    composed_inverse = None
    if t1.inverse is not None and t2.inverse is not None:
        def composed_inverse(data):
            return t1.inverse(t2.inverse(data))
    
    # Determine composed type (conservative)
    if t1.transform_type == TransformationType.ISOMORPHISM and \
       t2.transform_type == TransformationType.ISOMORPHISM:
        composed_type = TransformationType.ISOMORPHISM
    elif t1.is_lossless and t2.is_lossless:
        composed_type = TransformationType.EMBEDDING
    else:
        composed_type = TransformationType.LOSSY
    
    # Estimate combined info loss (1 - (1-l1)*(1-l2))
    combined_loss = 1 - (1 - t1.info_loss_estimate) * (1 - t2.info_loss_estimate)
    
    return Transformation(
        source=t1.source,
        target=t2.target,
        forward=composed_forward,
        inverse=composed_inverse,
        transform_type=composed_type,
        info_loss_estimate=combined_loss,
        name=f"{t1.name}_then_{t2.name}",
        metadata={
            "composed_from": [t1.name, t2.name],
            "composition_order": "left_to_right"
        }
    )
