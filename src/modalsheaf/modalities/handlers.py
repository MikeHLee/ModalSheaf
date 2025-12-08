"""
Modality handlers for loading, saving, and validating data.

Each handler knows how to work with a specific modality type.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
import numpy as np


# ==================== Base Handler ====================

class ModalityHandler(ABC):
    """
    Abstract base class for modality handlers.
    
    Handlers know how to:
    - Validate data for their modality
    - Load data from files/sources
    - Save data to files
    - Provide metadata about the data
    """
    
    modality_name: str = "base"
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Check if data is valid for this modality."""
        pass
    
    @abstractmethod
    def load(self, source: Any) -> Any:
        """Load data from a source (file path, URL, etc.)."""
        pass
    
    @abstractmethod
    def save(self, data: Any, destination: Any) -> None:
        """Save data to a destination."""
        pass
    
    def get_metadata(self, data: Any) -> Dict[str, Any]:
        """Extract metadata from data."""
        return {"modality": self.modality_name}
    
    def normalize(self, data: Any) -> Any:
        """Normalize data to canonical form."""
        return data


# ==================== Handler Registry ====================

_HANDLER_REGISTRY: Dict[str, ModalityHandler] = {}


def register_handler(handler: ModalityHandler) -> ModalityHandler:
    """Register a handler for a modality."""
    _HANDLER_REGISTRY[handler.modality_name] = handler
    return handler


def get_handler(modality_name: str) -> Optional[ModalityHandler]:
    """Get the handler for a modality."""
    return _HANDLER_REGISTRY.get(modality_name)


def list_handlers() -> List[str]:
    """List all registered handlers."""
    return list(_HANDLER_REGISTRY.keys())


# ==================== Text Handler ====================

class TextHandler(ModalityHandler):
    """Handler for text modality."""
    
    modality_name = "text"
    
    def validate(self, data: Any) -> bool:
        return isinstance(data, str)
    
    def load(self, source: Union[str, Path]) -> str:
        """Load text from file path."""
        path = Path(source)
        if path.exists():
            return path.read_text(encoding='utf-8')
        # If not a path, assume it's already text
        return str(source)
    
    def save(self, data: str, destination: Union[str, Path]) -> None:
        """Save text to file."""
        Path(destination).write_text(data, encoding='utf-8')
    
    def get_metadata(self, data: str) -> Dict[str, Any]:
        return {
            "modality": self.modality_name,
            "length": len(data),
            "num_lines": data.count('\n') + 1,
            "num_words": len(data.split()),
        }


register_handler(TextHandler())


# ==================== Image Handler ====================

class ImageHandler(ModalityHandler):
    """Handler for image modality."""
    
    modality_name = "image"
    
    def validate(self, data: Any) -> bool:
        if isinstance(data, np.ndarray):
            return data.ndim in (2, 3)  # Grayscale or RGB
        # Check for PIL Image
        try:
            from PIL import Image
            return isinstance(data, Image.Image)
        except ImportError:
            pass
        return False
    
    def load(self, source: Union[str, Path]) -> np.ndarray:
        """Load image from file path."""
        try:
            from PIL import Image
            img = Image.open(source)
            return np.array(img)
        except ImportError:
            raise ImportError("Pillow required for image loading. Install with: pip install pillow")
    
    def save(self, data: np.ndarray, destination: Union[str, Path]) -> None:
        """Save image to file."""
        try:
            from PIL import Image
            img = Image.fromarray(data.astype(np.uint8))
            img.save(destination)
        except ImportError:
            raise ImportError("Pillow required for image saving. Install with: pip install pillow")
    
    def get_metadata(self, data: np.ndarray) -> Dict[str, Any]:
        data = np.asarray(data)
        return {
            "modality": self.modality_name,
            "shape": data.shape,
            "height": data.shape[0],
            "width": data.shape[1],
            "channels": data.shape[2] if data.ndim == 3 else 1,
            "dtype": str(data.dtype),
            "min": float(data.min()),
            "max": float(data.max()),
        }
    
    def normalize(self, data: Any) -> np.ndarray:
        """Normalize to float32 array in [0, 1]."""
        data = np.asarray(data)
        if data.dtype == np.uint8:
            return data.astype(np.float32) / 255.0
        return data.astype(np.float32)


register_handler(ImageHandler())


# ==================== Embedding Handler ====================

class EmbeddingHandler(ModalityHandler):
    """Handler for embedding modality."""
    
    modality_name = "embedding"
    
    def validate(self, data: Any) -> bool:
        if isinstance(data, (list, tuple)):
            return all(isinstance(x, (int, float)) for x in data)
        if isinstance(data, np.ndarray):
            return data.ndim == 1
        return False
    
    def load(self, source: Union[str, Path]) -> np.ndarray:
        """Load embedding from .npy file."""
        return np.load(source)
    
    def save(self, data: np.ndarray, destination: Union[str, Path]) -> None:
        """Save embedding to .npy file."""
        np.save(destination, np.asarray(data))
    
    def get_metadata(self, data: Any) -> Dict[str, Any]:
        data = np.asarray(data)
        return {
            "modality": self.modality_name,
            "dimension": len(data),
            "dtype": str(data.dtype),
            "norm": float(np.linalg.norm(data)),
        }
    
    def normalize(self, data: Any) -> np.ndarray:
        """Convert to float32 numpy array."""
        return np.asarray(data, dtype=np.float32)


register_handler(EmbeddingHandler())


# ==================== Code Handler ====================

class CodeHandler(ModalityHandler):
    """Handler for source code modality."""
    
    modality_name = "code"
    
    # Language detection by extension
    LANG_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sql': 'sql',
        '.sh': 'bash',
        '.bash': 'bash',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.json': 'json',
        '.xml': 'xml',
        '.md': 'markdown',
    }
    
    def validate(self, data: Any) -> bool:
        return isinstance(data, str)
    
    def load(self, source: Union[str, Path]) -> str:
        """Load code from file."""
        path = Path(source)
        return path.read_text(encoding='utf-8')
    
    def save(self, data: str, destination: Union[str, Path]) -> None:
        """Save code to file."""
        Path(destination).write_text(data, encoding='utf-8')
    
    def get_metadata(self, data: str, path: Optional[str] = None) -> Dict[str, Any]:
        metadata = {
            "modality": self.modality_name,
            "length": len(data),
            "num_lines": data.count('\n') + 1,
        }
        
        if path:
            ext = Path(path).suffix.lower()
            metadata["language"] = self.LANG_EXTENSIONS.get(ext, "unknown")
            metadata["extension"] = ext
        
        return metadata
    
    def detect_language(self, path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(path).suffix.lower()
        return self.LANG_EXTENSIONS.get(ext, "unknown")


register_handler(CodeHandler())


# ==================== JSON Handler ====================

class JSONHandler(ModalityHandler):
    """Handler for JSON data modality."""
    
    modality_name = "json_data"
    
    def validate(self, data: Any) -> bool:
        return isinstance(data, (dict, list))
    
    def load(self, source: Union[str, Path]) -> Any:
        """Load JSON from file or string."""
        import json
        
        path = Path(source)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Try parsing as JSON string
        if isinstance(source, str):
            return json.loads(source)
        
        return source
    
    def save(self, data: Any, destination: Union[str, Path]) -> None:
        """Save JSON to file."""
        import json
        
        with open(destination, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_metadata(self, data: Any) -> Dict[str, Any]:
        import json
        
        return {
            "modality": self.modality_name,
            "type": type(data).__name__,
            "size": len(json.dumps(data)),
            "num_keys": len(data) if isinstance(data, dict) else None,
            "num_items": len(data) if isinstance(data, list) else None,
        }


register_handler(JSONHandler())


# ==================== URL Handler ====================

class URLHandler(ModalityHandler):
    """Handler for URL modality."""
    
    modality_name = "url"
    
    def validate(self, data: Any) -> bool:
        if not isinstance(data, str):
            return False
        return data.startswith(('http://', 'https://'))
    
    def load(self, source: str) -> Dict[str, Any]:
        """Fetch URL and return content with metadata."""
        try:
            import urllib.request
            
            with urllib.request.urlopen(source, timeout=30) as response:
                content = response.read()
                content_type = response.headers.get('Content-Type', '')
                
                # Try to decode as text
                try:
                    text = content.decode('utf-8')
                except:
                    text = None
                
                return {
                    "url": source,
                    "content": text or content,
                    "content_type": content_type,
                    "status": response.status,
                }
        except Exception as e:
            return {
                "url": source,
                "error": str(e),
            }
    
    def save(self, data: Any, destination: Any) -> None:
        """URLs can't be saved in the traditional sense."""
        raise NotImplementedError("URLs cannot be saved")
    
    def get_metadata(self, data: str) -> Dict[str, Any]:
        from urllib.parse import urlparse
        
        parsed = urlparse(data)
        return {
            "modality": self.modality_name,
            "scheme": parsed.scheme,
            "domain": parsed.netloc,
            "path": parsed.path,
        }


register_handler(URLHandler())


# ==================== Composite Handlers (Stubs) ====================

class CodebaseHandler(ModalityHandler):
    """Handler for codebase (directory of code) modality."""
    
    modality_name = "codebase"
    
    def validate(self, data: Any) -> bool:
        if isinstance(data, str):
            return Path(data).is_dir()
        return isinstance(data, dict)
    
    def load(self, source: Union[str, Path]) -> Dict[str, str]:
        """Load all code files from directory."""
        path = Path(source)
        code_handler = CodeHandler()
        
        files = {}
        for ext in code_handler.LANG_EXTENSIONS.keys():
            for file_path in path.rglob(f"*{ext}"):
                # Skip common non-source directories
                if any(part.startswith('.') or part in ('node_modules', 'venv', '__pycache__', 'dist', 'build') 
                       for part in file_path.parts):
                    continue
                
                rel_path = file_path.relative_to(path)
                try:
                    files[str(rel_path)] = file_path.read_text(encoding='utf-8')
                except:
                    pass  # Skip files that can't be read
        
        return files
    
    def save(self, data: Dict[str, str], destination: Union[str, Path]) -> None:
        """Save codebase to directory."""
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        
        for rel_path, content in data.items():
            file_path = dest / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
    
    def get_metadata(self, data: Dict[str, str]) -> Dict[str, Any]:
        code_handler = CodeHandler()
        
        languages = {}
        total_lines = 0
        
        for path, content in data.items():
            lang = code_handler.detect_language(path)
            languages[lang] = languages.get(lang, 0) + 1
            total_lines += content.count('\n') + 1
        
        return {
            "modality": self.modality_name,
            "num_files": len(data),
            "total_lines": total_lines,
            "languages": languages,
        }


register_handler(CodebaseHandler())
