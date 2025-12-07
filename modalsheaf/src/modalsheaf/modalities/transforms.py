"""
Built-in transformation functions between modalities.

This module provides:
- Restriction maps (coarse → fine, or high-level → low-level)
- Extension maps (fine → coarse, or low-level → high-level)

In sheaf terms:
- Restriction: ρ_{U,V}: F(U) → F(V) where V ⊆ U (zoom in, extract detail)
- Extension: ε_{V,U}: F(V) → F(U) where V ⊆ U (zoom out, aggregate)

In ML terms:
- Restriction: decode, expand, upsample, generate
- Extension: encode, compress, downsample, summarize
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from ..core import Transformation, TransformationType


# ==================== Text Transformations ====================

def text_to_tokens(
    text: str,
    tokenizer: Optional[Any] = None,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Extension map: text → tokens
    
    Converts text string to token IDs.
    Uses simple whitespace tokenization if no tokenizer provided.
    """
    if tokenizer is not None:
        # Use provided tokenizer (e.g., HuggingFace)
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text)
            if max_length:
                tokens = tokens[:max_length]
            return np.array(tokens, dtype=np.int64)
    
    # Simple fallback: character-level
    tokens = [ord(c) for c in text]
    if max_length:
        tokens = tokens[:max_length]
    return np.array(tokens, dtype=np.int64)


def tokens_to_text(
    tokens: np.ndarray,
    tokenizer: Optional[Any] = None,
) -> str:
    """
    Restriction map: tokens → text
    
    Converts token IDs back to text string.
    """
    tokens = np.asarray(tokens).flatten().tolist()
    
    if tokenizer is not None:
        if hasattr(tokenizer, 'decode'):
            return tokenizer.decode(tokens)
    
    # Simple fallback: character-level
    return ''.join(chr(t) for t in tokens if 0 <= t < 0x110000)


def text_to_sentences(text: str) -> List[str]:
    """
    Restriction map: text → sentences
    
    Split text into sentences.
    """
    import re
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def sentences_to_text(sentences: List[str]) -> str:
    """
    Extension map: sentences → text
    
    Join sentences back into text.
    """
    return ' '.join(sentences)


def text_to_words(text: str) -> List[str]:
    """
    Restriction map: text → words
    """
    return text.split()


def words_to_text(words: List[str]) -> str:
    """
    Extension map: words → text
    """
    return ' '.join(words)


def text_to_chars(text: str) -> List[str]:
    """
    Restriction map: text → characters
    """
    return list(text)


def chars_to_text(chars: List[str]) -> str:
    """
    Extension map: characters → text
    """
    return ''.join(chars)


# ==================== Image Transformations ====================

def image_to_patches(
    image: np.ndarray,
    patch_size: int = 16,
) -> np.ndarray:
    """
    Restriction map: image → patches
    
    Split image into non-overlapping patches.
    Returns shape (num_patches, patch_size, patch_size, channels)
    """
    image = np.asarray(image)
    
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    
    H, W, C = image.shape
    
    # Crop to multiple of patch_size
    H_crop = (H // patch_size) * patch_size
    W_crop = (W // patch_size) * patch_size
    image = image[:H_crop, :W_crop, :]
    
    # Reshape into patches
    patches = image.reshape(
        H_crop // patch_size, patch_size,
        W_crop // patch_size, patch_size,
        C
    )
    patches = patches.transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(-1, patch_size, patch_size, C)
    
    return patches


def patches_to_image(
    patches: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Extension map: patches → image
    
    Reassemble patches into image.
    """
    patches = np.asarray(patches)
    num_patches, patch_size, _, C = patches.shape
    H, W = image_shape
    
    H_patches = H // patch_size
    W_patches = W // patch_size
    
    patches = patches.reshape(H_patches, W_patches, patch_size, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)
    image = patches.reshape(H_patches * patch_size, W_patches * patch_size, C)
    
    if C == 1:
        image = image.squeeze(-1)
    
    return image


def image_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Extension map: RGB image → grayscale
    
    Reduces color channels (lossy).
    """
    image = np.asarray(image)
    
    if image.ndim == 2:
        return image
    
    if image.shape[-1] == 3:
        # Standard luminance weights
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    
    return image


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Restriction map: grayscale → RGB
    
    Expands to 3 channels (no new information).
    """
    image = np.asarray(image)
    
    if image.ndim == 3 and image.shape[-1] == 3:
        return image
    
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    
    return image


def image_resize(
    image: np.ndarray,
    target_size: Tuple[int, int],
    method: str = "bilinear",
) -> np.ndarray:
    """
    Bidirectional map: image → resized image
    
    Can be extension (upsample) or restriction (downsample).
    """
    try:
        from PIL import Image
        
        img = Image.fromarray(image.astype(np.uint8))
        
        resample_methods = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        resample = resample_methods.get(method, Image.BILINEAR)
        
        img = img.resize((target_size[1], target_size[0]), resample=resample)
        return np.array(img)
    
    except ImportError:
        # Simple nearest-neighbor fallback
        image = np.asarray(image)
        H, W = image.shape[:2]
        target_H, target_W = target_size
        
        row_indices = (np.arange(target_H) * H / target_H).astype(int)
        col_indices = (np.arange(target_W) * W / target_W).astype(int)
        
        return image[row_indices][:, col_indices]


def image_normalize(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Extension map: raw image → normalized image
    
    Standard ImageNet normalization.
    """
    image = np.asarray(image, dtype=np.float32)
    
    if image.max() > 1.0:
        image = image / 255.0
    
    mean = np.array(mean)
    std = np.array(std)
    
    return (image - mean) / std


def image_denormalize(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Restriction map: normalized image → raw image
    """
    image = np.asarray(image, dtype=np.float32)
    
    mean = np.array(mean)
    std = np.array(std)
    
    image = image * std + mean
    return np.clip(image * 255, 0, 255).astype(np.uint8)


# ==================== Embedding Transformations ====================

def embedding_normalize(embedding: np.ndarray) -> np.ndarray:
    """
    Extension map: embedding → unit embedding
    
    L2 normalize to unit sphere.
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    
    if norm > 0:
        return embedding / norm
    return embedding


def embedding_reduce_dim(
    embedding: np.ndarray,
    target_dim: int,
    method: str = "pca",
) -> np.ndarray:
    """
    Extension map: high-dim embedding → low-dim embedding
    
    Dimensionality reduction (lossy).
    """
    embedding = np.asarray(embedding).reshape(1, -1)
    
    if method == "truncate":
        return embedding[0, :target_dim]
    
    if method == "random_projection":
        np.random.seed(42)  # Reproducible
        proj = np.random.randn(embedding.shape[1], target_dim)
        proj = proj / np.linalg.norm(proj, axis=0)
        return (embedding @ proj).flatten()
    
    # Default: truncate
    return embedding[0, :target_dim]


def embedding_expand_dim(
    embedding: np.ndarray,
    target_dim: int,
) -> np.ndarray:
    """
    Restriction map: low-dim embedding → high-dim embedding
    
    Zero-padding (no new information).
    """
    embedding = np.asarray(embedding).flatten()
    
    if len(embedding) >= target_dim:
        return embedding[:target_dim]
    
    padded = np.zeros(target_dim, dtype=embedding.dtype)
    padded[:len(embedding)] = embedding
    return padded


def embeddings_average(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Extension map: multiple embeddings → single embedding
    
    Mean pooling.
    """
    embeddings = [np.asarray(e) for e in embeddings]
    return np.mean(embeddings, axis=0)


def embeddings_max_pool(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Extension map: multiple embeddings → single embedding
    
    Max pooling.
    """
    embeddings = [np.asarray(e) for e in embeddings]
    return np.max(embeddings, axis=0)


# ==================== Code Transformations ====================

def code_to_ast(code: str, language: str = "python") -> Dict:
    """
    Extension map: code → AST
    
    Parse code into abstract syntax tree.
    """
    if language == "python":
        import ast
        try:
            tree = ast.parse(code)
            return _ast_to_dict(tree)
        except SyntaxError as e:
            return {"error": str(e), "type": "SyntaxError"}
    
    # For other languages, return a simple structure
    return {
        "type": "raw",
        "language": language,
        "content": code,
    }


def _ast_to_dict(node) -> Dict:
    """Convert Python AST node to dictionary."""
    import ast
    
    result = {"type": node.__class__.__name__}
    
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            result[field] = [
                _ast_to_dict(item) if isinstance(item, ast.AST) else item
                for item in value
            ]
        elif isinstance(value, ast.AST):
            result[field] = _ast_to_dict(value)
        else:
            result[field] = value
    
    return result


def ast_to_code(ast_dict: Dict, language: str = "python") -> str:
    """
    Restriction map: AST → code
    
    Unparse AST back to code (may lose formatting).
    """
    if language == "python":
        import ast
        
        if ast_dict.get("type") == "raw":
            return ast_dict.get("content", "")
        
        # For full AST reconstruction, would need ast.unparse (Python 3.9+)
        # This is a simplified version
        return f"# AST with root type: {ast_dict.get('type', 'unknown')}"
    
    return ast_dict.get("content", "")


def code_to_functions(code: str, language: str = "python") -> List[Dict]:
    """
    Restriction map: code → function definitions
    
    Extract function signatures and bodies.
    """
    functions = []
    
    if language == "python":
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "lineno": node.lineno,
                        "docstring": ast.get_docstring(node),
                    })
        except SyntaxError:
            pass
    
    return functions


def code_to_imports(code: str, language: str = "python") -> List[str]:
    """
    Restriction map: code → import statements
    """
    imports = []
    
    if language == "python":
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
        except SyntaxError:
            pass
    
    return imports


def code_remove_comments(code: str, language: str = "python") -> str:
    """
    Extension map: code → code without comments
    
    Strip comments (lossy).
    """
    import re
    
    if language == "python":
        # Remove # comments (but not in strings - simplified)
        lines = code.split('\n')
        result = []
        for line in lines:
            # Simple: remove everything after # not in quotes
            if '#' in line:
                # Very simplified - doesn't handle strings properly
                idx = line.find('#')
                line = line[:idx].rstrip()
            if line.strip():
                result.append(line)
        return '\n'.join(result)
    
    return code


# ==================== JSON Transformations ====================

def json_to_text(data: Union[Dict, List]) -> str:
    """
    Restriction map: JSON → text
    
    Serialize JSON to string.
    """
    import json
    return json.dumps(data, indent=2, ensure_ascii=False)


def text_to_json(text: str) -> Union[Dict, List]:
    """
    Extension map: text → JSON
    
    Parse JSON from string.
    """
    import json
    return json.loads(text)


def json_flatten(data: Dict, separator: str = ".") -> Dict[str, Any]:
    """
    Restriction map: nested JSON → flat JSON
    
    Flatten nested structure.
    """
    result = {}
    
    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                _flatten(value, new_key)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                new_key = f"{prefix}{separator}{i}" if prefix else str(i)
                _flatten(value, new_key)
        else:
            result[prefix] = obj
    
    _flatten(data)
    return result


def json_unflatten(data: Dict[str, Any], separator: str = ".") -> Dict:
    """
    Extension map: flat JSON → nested JSON
    
    Reconstruct nested structure.
    """
    result = {}
    
    for key, value in data.items():
        parts = key.split(separator)
        current = result
        
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                # Check if next part is numeric (list) or not (dict)
                next_part = parts[i + 1]
                if next_part.isdigit():
                    current[part] = []
                else:
                    current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


def json_extract_keys(data: Dict, keys: List[str]) -> Dict:
    """
    Restriction map: JSON → subset of JSON
    
    Extract only specified keys.
    """
    return {k: data[k] for k in keys if k in data}


def json_get_schema(data: Union[Dict, List]) -> Dict:
    """
    Extension map: JSON data → JSON schema
    
    Infer schema from data (lossy - loses actual values).
    """
    def _get_type(value):
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            if value:
                return {"type": "array", "items": _get_type(value[0])}
            return {"type": "array"}
        if isinstance(value, dict):
            return {
                "type": "object",
                "properties": {k: _get_type(v) for k, v in value.items()}
            }
        return "unknown"
    
    return _get_type(data)


# ==================== Transformation Registry ====================

@dataclass
class TransformSpec:
    """Specification for a built-in transformation."""
    name: str
    source: str
    target: str
    func: Callable
    inverse_func: Optional[Callable] = None
    transform_type: TransformationType = TransformationType.LOSSY
    info_loss: float = 0.5
    description: str = ""


# Registry of built-in transforms
BUILTIN_TRANSFORMS: Dict[str, TransformSpec] = {}


def register_builtin(spec: TransformSpec) -> TransformSpec:
    """Register a built-in transformation."""
    BUILTIN_TRANSFORMS[spec.name] = spec
    return spec


# Register text transforms
register_builtin(TransformSpec(
    name="text_to_tokens",
    source="text", target="tokens",
    func=text_to_tokens,
    inverse_func=tokens_to_text,
    transform_type=TransformationType.EMBEDDING,
    info_loss=0.1,
    description="Tokenize text to token IDs"
))

register_builtin(TransformSpec(
    name="text_to_sentences",
    source="text", target="sentences",
    func=text_to_sentences,
    inverse_func=sentences_to_text,
    transform_type=TransformationType.ISOMORPHISM,
    info_loss=0.0,
    description="Split text into sentences"
))

register_builtin(TransformSpec(
    name="text_to_words",
    source="text", target="words",
    func=text_to_words,
    inverse_func=words_to_text,
    transform_type=TransformationType.EMBEDDING,
    info_loss=0.05,
    description="Split text into words"
))

# Register image transforms
register_builtin(TransformSpec(
    name="image_to_patches",
    source="image", target="patches",
    func=image_to_patches,
    inverse_func=None,  # Needs image_shape
    transform_type=TransformationType.EMBEDDING,
    info_loss=0.0,
    description="Split image into patches"
))

register_builtin(TransformSpec(
    name="image_to_grayscale",
    source="image", target="image_gray",
    func=image_to_grayscale,
    inverse_func=grayscale_to_rgb,
    transform_type=TransformationType.PROJECTION,
    info_loss=0.67,
    description="Convert RGB to grayscale"
))

register_builtin(TransformSpec(
    name="image_normalize",
    source="image", target="image_normalized",
    func=image_normalize,
    inverse_func=image_denormalize,
    transform_type=TransformationType.ISOMORPHISM,
    info_loss=0.0,
    description="ImageNet normalization"
))

# Register embedding transforms
register_builtin(TransformSpec(
    name="embedding_normalize",
    source="embedding", target="embedding_unit",
    func=embedding_normalize,
    inverse_func=None,
    transform_type=TransformationType.PROJECTION,
    info_loss=0.1,
    description="L2 normalize embedding"
))

# Register code transforms
register_builtin(TransformSpec(
    name="code_to_ast",
    source="code", target="ast",
    func=code_to_ast,
    inverse_func=ast_to_code,
    transform_type=TransformationType.EMBEDDING,
    info_loss=0.1,
    description="Parse code to AST"
))

register_builtin(TransformSpec(
    name="code_to_functions",
    source="code", target="functions",
    func=code_to_functions,
    inverse_func=None,
    transform_type=TransformationType.PROJECTION,
    info_loss=0.7,
    description="Extract function definitions"
))

register_builtin(TransformSpec(
    name="code_remove_comments",
    source="code", target="code_clean",
    func=code_remove_comments,
    inverse_func=None,
    transform_type=TransformationType.PROJECTION,
    info_loss=0.2,
    description="Remove comments from code"
))

# Register JSON transforms
register_builtin(TransformSpec(
    name="json_to_text",
    source="json_data", target="text",
    func=json_to_text,
    inverse_func=text_to_json,
    transform_type=TransformationType.ISOMORPHISM,
    info_loss=0.0,
    description="Serialize JSON to text"
))

register_builtin(TransformSpec(
    name="json_flatten",
    source="json_data", target="json_flat",
    func=json_flatten,
    inverse_func=json_unflatten,
    transform_type=TransformationType.ISOMORPHISM,
    info_loss=0.0,
    description="Flatten nested JSON"
))

register_builtin(TransformSpec(
    name="json_get_schema",
    source="json_data", target="json_schema",
    func=json_get_schema,
    inverse_func=None,
    transform_type=TransformationType.PROJECTION,
    info_loss=0.9,
    description="Extract schema from JSON"
))


def get_builtin_transform(name: str) -> Optional[TransformSpec]:
    """Get a built-in transformation by name."""
    return BUILTIN_TRANSFORMS.get(name)


def list_builtin_transforms(
    source: Optional[str] = None,
    target: Optional[str] = None,
) -> List[str]:
    """List built-in transforms, optionally filtered by source/target."""
    result = []
    for name, spec in BUILTIN_TRANSFORMS.items():
        if source and spec.source != source:
            continue
        if target and spec.target != target:
            continue
        result.append(name)
    return result


def create_transformation_from_builtin(name: str) -> Transformation:
    """Create a Transformation object from a built-in spec."""
    spec = get_builtin_transform(name)
    if spec is None:
        raise ValueError(f"Unknown built-in transform: {name}")
    
    return Transformation(
        source=spec.source,
        target=spec.target,
        forward=spec.func,
        inverse=spec.inverse_func,
        transform_type=spec.transform_type,
        info_loss_estimate=spec.info_loss,
        name=spec.name,
        metadata={"builtin": True, "description": spec.description}
    )
