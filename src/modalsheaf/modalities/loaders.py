"""
Unified loading and saving interface for all modalities.

Provides a single entry point for loading/saving any supported data type.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import (
    detect_modality_from_extension,
    detect_modality_from_data,
    get_modality,
)
from .handlers import get_handler


def detect_modality(
    data: Optional[Any] = None,
    path: Optional[Union[str, Path]] = None,
    mime_type: Optional[str] = None,
) -> Optional[str]:
    """
    Detect the modality of data or a file.
    
    Args:
        data: The data to analyze
        path: File path (for extension-based detection)
        mime_type: MIME type (for type-based detection)
    
    Returns:
        Modality name or None if undetected
    
    Example:
        >>> detect_modality(path="image.png")
        "image_file"
        >>> detect_modality(data="Hello world")
        "text"
        >>> detect_modality(data=np.random.randn(768))
        "embedding"
    """
    # Try path-based detection first
    if path is not None:
        path_str = str(path)
        mod = detect_modality_from_extension(path_str)
        if mod:
            return mod
    
    # Try MIME type
    if mime_type is not None:
        from .base import detect_modality_from_mime
        mod = detect_modality_from_mime(mime_type)
        if mod:
            return mod
    
    # Try data-based detection
    if data is not None:
        return detect_modality_from_data(data)
    
    return None


def load_file(
    path: Union[str, Path],
    modality: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Load data from a file.
    
    Args:
        path: Path to the file
        modality: Override modality detection (optional)
        **kwargs: Additional arguments for the handler
    
    Returns:
        Loaded data
    
    Example:
        >>> image = load_file("photo.jpg")
        >>> code = load_file("main.py")
        >>> data = load_file("config.json")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Detect modality if not provided
    if modality is None:
        modality = detect_modality(path=path)
        
        # Map file modalities to their content modalities
        file_to_content = {
            "text_file": "text",
            "image_file": "image",
            "code_file": "code",
            "json_file": "json_data",
        }
        modality = file_to_content.get(modality, modality)
    
    if modality is None:
        raise ValueError(f"Could not detect modality for: {path}")
    
    # Get handler and load
    handler = get_handler(modality)
    if handler is None:
        raise ValueError(f"No handler for modality: {modality}")
    
    return handler.load(path, **kwargs)


def save_file(
    data: Any,
    path: Union[str, Path],
    modality: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save data to a file.
    
    Args:
        data: Data to save
        path: Destination path
        modality: Override modality detection (optional)
        **kwargs: Additional arguments for the handler
    
    Example:
        >>> save_file(image_array, "output.png")
        >>> save_file({"key": "value"}, "config.json")
    """
    path = Path(path)
    
    # Detect modality if not provided
    if modality is None:
        # Try from data first
        modality = detect_modality(data=data)
        
        # Fall back to extension
        if modality is None:
            modality = detect_modality(path=path)
            file_to_content = {
                "text_file": "text",
                "image_file": "image",
                "code_file": "code",
                "json_file": "json_data",
            }
            modality = file_to_content.get(modality, modality)
    
    if modality is None:
        raise ValueError(f"Could not detect modality for saving to: {path}")
    
    # Get handler and save
    handler = get_handler(modality)
    if handler is None:
        raise ValueError(f"No handler for modality: {modality}")
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    handler.save(data, path, **kwargs)


def load_url(
    url: str,
    target_modality: Optional[str] = None,
) -> Any:
    """
    Load data from a URL.
    
    Args:
        url: URL to fetch
        target_modality: Expected modality of content (optional)
    
    Returns:
        Loaded data
    
    Example:
        >>> html = load_url("https://example.com")
        >>> image = load_url("https://example.com/image.png", target_modality="image")
    """
    handler = get_handler("url")
    if handler is None:
        raise ValueError("URL handler not available")
    
    result = handler.load(url)
    
    if "error" in result:
        raise RuntimeError(f"Failed to load URL: {result['error']}")
    
    content = result.get("content")
    content_type = result.get("content_type", "")
    
    # Try to convert to target modality
    if target_modality:
        target_handler = get_handler(target_modality)
        if target_handler:
            return target_handler.normalize(content)
    
    # Return based on content type
    if "json" in content_type:
        import json
        return json.loads(content) if isinstance(content, str) else content
    
    if "image" in content_type:
        # Would need to decode image bytes
        return content
    
    return content


def get_metadata(
    data: Any,
    modality: Optional[str] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get metadata about data.
    
    Args:
        data: The data to analyze
        modality: Override modality detection (optional)
        path: Original file path (for additional context)
    
    Returns:
        Dictionary of metadata
    
    Example:
        >>> get_metadata(image_array)
        {"modality": "image", "shape": (224, 224, 3), ...}
    """
    if modality is None:
        modality = detect_modality(data=data, path=path)
    
    if modality is None:
        return {"modality": "unknown"}
    
    handler = get_handler(modality)
    if handler is None:
        return {"modality": modality}
    
    metadata = handler.get_metadata(data)
    
    if path:
        metadata["source_path"] = str(path)
    
    return metadata


# ==================== Batch Operations ====================

def load_directory(
    path: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True,
    modality: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load all matching files from a directory.
    
    Args:
        path: Directory path
        pattern: Glob pattern for files
        recursive: Whether to search recursively
        modality: Filter by modality (optional)
    
    Returns:
        Dictionary mapping relative paths to loaded data
    
    Example:
        >>> images = load_directory("./images", pattern="*.png")
        >>> code = load_directory("./src", pattern="*.py")
    """
    path = Path(path)
    
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    
    glob_method = path.rglob if recursive else path.glob
    
    results = {}
    for file_path in glob_method(pattern):
        if not file_path.is_file():
            continue
        
        # Check modality filter
        if modality:
            detected = detect_modality(path=file_path)
            if detected != modality and not detected.endswith(f"_{modality}"):
                continue
        
        try:
            rel_path = file_path.relative_to(path)
            results[str(rel_path)] = load_file(file_path)
        except Exception as e:
            # Skip files that can't be loaded
            pass
    
    return results


def save_directory(
    data: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """
    Save multiple files to a directory.
    
    Args:
        data: Dictionary mapping relative paths to data
        path: Destination directory
    
    Example:
        >>> save_directory({"a.txt": "hello", "b.json": {"x": 1}}, "./output")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    for rel_path, content in data.items():
        file_path = path / rel_path
        save_file(content, file_path)
