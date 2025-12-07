"""
Transform utilities and common transformation functions.

Provides helpers for:
- Registering transforms with decorators
- Composing transforms
- Inverting transforms
- Common ML transforms (with optional dependencies)
"""

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np

from .core import Transformation, TransformationType


# ==================== Decorator-based Registration ====================

_transform_registry: Dict[str, Transformation] = {}


def register_transform(
    source: str,
    target: str,
    transform_type: TransformationType = TransformationType.UNKNOWN,
    info_loss: float = 0.5,
    name: Optional[str] = None
):
    """
    Decorator to register a function as a transformation.
    
    Example:
        >>> @register_transform("image", "embedding", info_loss=0.8)
        ... def encode_image(img):
        ...     return model.encode(img)
    """
    def decorator(func: Callable) -> Callable:
        transform_name = name or func.__name__
        
        transform = Transformation(
            source=source,
            target=target,
            forward=func,
            inverse=None,
            transform_type=transform_type,
            info_loss_estimate=info_loss,
            name=transform_name
        )
        
        _transform_registry[transform_name] = transform
        
        # Attach metadata to function
        func._modalsheaf_transform = transform
        
        return func
    
    return decorator


def register_inverse(transform_name: str):
    """
    Decorator to register an inverse for an existing transform.
    
    Example:
        >>> @register_transform("text", "tokens")
        ... def tokenize(text):
        ...     return tokenizer.encode(text)
        
        >>> @register_inverse("tokenize")
        ... def detokenize(tokens):
        ...     return tokenizer.decode(tokens)
    """
    def decorator(func: Callable) -> Callable:
        if transform_name not in _transform_registry:
            raise ValueError(f"Transform '{transform_name}' not found in registry")
        
        transform = _transform_registry[transform_name]
        transform.inverse = func
        
        # Update type if it was unknown
        if transform.transform_type == TransformationType.UNKNOWN:
            transform.transform_type = TransformationType.ISOMORPHISM
            transform.info_loss_estimate = 0.0
        
        return func
    
    return decorator


def get_registered_transform(name: str) -> Optional[Transformation]:
    """Get a transform from the registry by name."""
    return _transform_registry.get(name)


def list_registered_transforms() -> List[str]:
    """List all registered transform names."""
    return list(_transform_registry.keys())


# ==================== Transform Composition ====================

def compose_transforms(*transforms: Transformation) -> Transformation:
    """
    Compose multiple transformations into one.
    
    Args:
        *transforms: Transformations to compose (applied left to right)
    
    Returns:
        Single composed transformation
    
    Example:
        >>> composed = compose_transforms(tokenize, embed, project)
    """
    if len(transforms) == 0:
        raise ValueError("Need at least one transform to compose")
    
    if len(transforms) == 1:
        return transforms[0]
    
    from .core import compose_transformations
    
    result = transforms[0]
    for t in transforms[1:]:
        result = compose_transformations(result, t)
    
    return result


def invert_transform(transform: Transformation) -> Transformation:
    """
    Create the inverse of a transformation.
    
    Args:
        transform: The transformation to invert
    
    Returns:
        Inverse transformation
    
    Raises:
        ValueError: If transform is not invertible
    """
    if not transform.is_invertible:
        raise ValueError(
            f"Transform '{transform.name}' is not invertible "
            f"(type: {transform.transform_type.name})"
        )
    
    return Transformation(
        source=transform.target,
        target=transform.source,
        forward=transform.inverse,
        inverse=transform.forward,
        transform_type=transform.transform_type,
        info_loss_estimate=transform.info_loss_estimate,
        name=f"{transform.name}_inverse"
    )


# ==================== Common Transforms ====================

def identity_transform(source: str, target: str) -> Transformation:
    """Create an identity transformation (useful for same-type conversions)."""
    return Transformation(
        source=source,
        target=target,
        forward=lambda x: x,
        inverse=lambda x: x,
        transform_type=TransformationType.ISOMORPHISM,
        info_loss_estimate=0.0,
        name=f"identity_{source}_to_{target}"
    )


def numpy_to_list_transform() -> Transformation:
    """Transform numpy arrays to Python lists."""
    return Transformation(
        source="numpy_array",
        target="python_list",
        forward=lambda x: x.tolist(),
        inverse=lambda x: np.array(x),
        transform_type=TransformationType.ISOMORPHISM,
        info_loss_estimate=0.0,
        name="numpy_to_list"
    )


def flatten_transform(source: str) -> Transformation:
    """Flatten array to 1D (lossy - loses shape info)."""
    return Transformation(
        source=source,
        target=f"{source}_flat",
        forward=lambda x: np.asarray(x).flatten(),
        inverse=None,  # Can't unflatten without knowing original shape
        transform_type=TransformationType.PROJECTION,
        info_loss_estimate=0.1,  # Low loss - just shape info
        name=f"flatten_{source}"
    )


def normalize_transform(source: str) -> Transformation:
    """L2 normalize vectors."""
    def normalize(x):
        x = np.asarray(x)
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x
    
    return Transformation(
        source=source,
        target=f"{source}_normalized",
        forward=normalize,
        inverse=None,  # Can't recover original magnitude
        transform_type=TransformationType.PROJECTION,
        info_loss_estimate=0.05,  # Only loses magnitude
        name=f"normalize_{source}"
    )


# ==================== ML Framework Transforms ====================

def create_torch_transform(
    source: str,
    target: str,
    model: Any,
    device: str = "cpu",
    info_loss: float = 0.5
) -> Transformation:
    """
    Create a transformation using a PyTorch model.
    
    Args:
        source: Source modality name
        target: Target modality name
        model: PyTorch model (nn.Module)
        device: Device to run on
        info_loss: Estimated information loss
    
    Returns:
        Transformation wrapping the model
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required. Install with: pip install torch")
    
    def forward(x):
        model.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)
            output = model(x)
            return output.cpu().numpy()
    
    return Transformation(
        source=source,
        target=target,
        forward=forward,
        inverse=None,
        transform_type=TransformationType.LOSSY,
        info_loss_estimate=info_loss,
        name=f"torch_{source}_to_{target}",
        metadata={"framework": "pytorch", "device": device}
    )


def create_huggingface_transform(
    source: str,
    target: str,
    model_name: str,
    task: str = "feature-extraction",
    info_loss: float = 0.5
) -> Transformation:
    """
    Create a transformation using a HuggingFace model.
    
    Args:
        source: Source modality name
        target: Target modality name
        model_name: HuggingFace model identifier
        task: Pipeline task type
        info_loss: Estimated information loss
    
    Returns:
        Transformation wrapping the HuggingFace pipeline
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "Transformers required. Install with: pip install transformers"
        )
    
    pipe = pipeline(task, model=model_name)
    
    def forward(x):
        result = pipe(x)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                return result[0].get("embedding", result)
            return result[0]
        return result
    
    return Transformation(
        source=source,
        target=target,
        forward=forward,
        inverse=None,
        transform_type=TransformationType.LOSSY,
        info_loss_estimate=info_loss,
        name=f"hf_{model_name.replace('/', '_')}",
        metadata={"framework": "huggingface", "model": model_name, "task": task}
    )


# ==================== Image Transforms ====================

def create_resize_transform(
    target_size: Tuple[int, int],
    source: str = "image",
    target: str = "image_resized"
) -> Transformation:
    """
    Create an image resize transformation.
    
    Args:
        target_size: (height, width) to resize to
        source: Source modality name
        target: Target modality name
    
    Returns:
        Resize transformation
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required. Install with: pip install pillow")
    
    def resize(img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((target_size[1], target_size[0]))  # PIL uses (w, h)
        return np.array(img)
    
    # Resize is lossy if downsampling
    return Transformation(
        source=source,
        target=target,
        forward=resize,
        inverse=None,
        transform_type=TransformationType.PROJECTION,
        info_loss_estimate=0.3,
        name=f"resize_to_{target_size[0]}x{target_size[1]}",
        metadata={"target_size": target_size}
    )


def create_grayscale_transform(
    source: str = "image",
    target: str = "image_gray"
) -> Transformation:
    """Create RGB to grayscale transformation."""
    def to_gray(img):
        img = np.asarray(img)
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Standard luminance weights
            return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        return img
    
    return Transformation(
        source=source,
        target=target,
        forward=to_gray,
        inverse=None,
        transform_type=TransformationType.PROJECTION,
        info_loss_estimate=0.67,  # Lose 2/3 of color info
        name="rgb_to_grayscale"
    )
