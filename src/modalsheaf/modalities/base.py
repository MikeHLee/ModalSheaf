"""
Base modality definitions.

Defines the core modalities and the registry system for adding new ones.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import numpy as np

from ..core import Modality


# ==================== Modality Categories ====================

class ModalityCategory(Enum):
    """Categories of modalities for organization."""
    PRIMITIVE = auto()      # Basic types (text, numbers)
    TENSOR = auto()         # Array-like (images, embeddings)
    STRUCTURED = auto()     # Structured data (JSON, XML)
    FILE = auto()           # File-based
    CODE = auto()           # Source code
    MEDIA = auto()          # Audio/video
    COMPOSITE = auto()      # Combinations
    WEB = auto()            # Web-related
    SYSTEM = auto()         # System/API


# ==================== Modality Specifications ====================

@dataclass
class ModalitySpec:
    """
    Full specification for a modality type.
    
    Extends the basic Modality with additional metadata for
    handling, validation, and transformation hints.
    """
    modality: Modality
    category: ModalityCategory
    file_extensions: List[str] = field(default_factory=list)
    mime_types: List[str] = field(default_factory=list)
    python_types: List[Type] = field(default_factory=list)
    parent: Optional[str] = None  # Parent modality (for hierarchy)
    children: List[str] = field(default_factory=list)
    typical_transforms_to: List[str] = field(default_factory=list)
    typical_transforms_from: List[str] = field(default_factory=list)
    
    @property
    def name(self) -> str:
        return self.modality.name


# ==================== Modality Registry ====================

MODALITY_REGISTRY: Dict[str, ModalitySpec] = {}


def register_modality(spec: ModalitySpec) -> ModalitySpec:
    """Register a modality specification."""
    MODALITY_REGISTRY[spec.name] = spec
    return spec


def get_modality(name: str) -> Optional[ModalitySpec]:
    """Get a modality specification by name."""
    return MODALITY_REGISTRY.get(name)


def list_modalities(category: Optional[ModalityCategory] = None) -> List[str]:
    """List all registered modalities, optionally filtered by category."""
    if category is None:
        return list(MODALITY_REGISTRY.keys())
    return [
        name for name, spec in MODALITY_REGISTRY.items()
        if spec.category == category
    ]


# ==================== Core Modalities (Tier 1) ====================

# Text
TEXT = register_modality(ModalitySpec(
    modality=Modality(
        name="text",
        shape=None,  # Variable length
        dtype="str",
        description="Raw text string"
    ),
    category=ModalityCategory.PRIMITIVE,
    python_types=[str],
    mime_types=["text/plain"],
    typical_transforms_to=["tokens", "embedding", "text_file"],
    typical_transforms_from=["text_file", "tokens"],
))

# Tokens (tokenized text)
TOKENS = register_modality(ModalitySpec(
    modality=Modality(
        name="tokens",
        shape=None,  # Variable length sequence
        dtype="int64",
        description="Token IDs from tokenizer"
    ),
    category=ModalityCategory.TENSOR,
    python_types=[list, np.ndarray],
    parent="text",
    typical_transforms_to=["embedding", "text"],
    typical_transforms_from=["text"],
))

# Sentences (text split into sentences)
SENTENCES = register_modality(ModalitySpec(
    modality=Modality(
        name="sentences",
        shape=None,
        dtype="object",
        description="List of sentences"
    ),
    category=ModalityCategory.PRIMITIVE,
    python_types=[list],
    parent="text",
    typical_transforms_to=["text", "embedding"],
    typical_transforms_from=["text"],
))

# Words (text split into words)
WORDS = register_modality(ModalitySpec(
    modality=Modality(
        name="words",
        shape=None,
        dtype="object",
        description="List of words"
    ),
    category=ModalityCategory.PRIMITIVE,
    python_types=[list],
    parent="text",
    typical_transforms_to=["text", "tokens"],
    typical_transforms_from=["text"],
))

# Characters
CHARS = register_modality(ModalitySpec(
    modality=Modality(
        name="chars",
        shape=None,
        dtype="object",
        description="List of characters"
    ),
    category=ModalityCategory.PRIMITIVE,
    python_types=[list],
    parent="text",
    typical_transforms_to=["text"],
    typical_transforms_from=["text"],
))

# Image
IMAGE = register_modality(ModalitySpec(
    modality=Modality(
        name="image",
        shape=None,  # Variable H, W, C
        dtype="float32",
        description="Image as array (H, W, C) or (H, W)"
    ),
    category=ModalityCategory.TENSOR,
    file_extensions=[".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"],
    mime_types=["image/png", "image/jpeg", "image/webp", "image/gif"],
    python_types=[np.ndarray],
    typical_transforms_to=["embedding", "image_file", "patches", "image_gray", "image_normalized"],
    typical_transforms_from=["image_file", "patches"],
))

# Image patches
PATCHES = register_modality(ModalitySpec(
    modality=Modality(
        name="patches",
        shape=None,  # (num_patches, patch_h, patch_w, channels)
        dtype="float32",
        description="Image split into patches"
    ),
    category=ModalityCategory.TENSOR,
    python_types=[np.ndarray],
    parent="image",
    typical_transforms_to=["image", "embedding"],
    typical_transforms_from=["image"],
))

# Grayscale image
IMAGE_GRAY = register_modality(ModalitySpec(
    modality=Modality(
        name="image_gray",
        shape=None,  # (H, W)
        dtype="float32",
        description="Grayscale image"
    ),
    category=ModalityCategory.TENSOR,
    python_types=[np.ndarray],
    parent="image",
    typical_transforms_to=["image", "embedding"],
    typical_transforms_from=["image"],
))

# Normalized image
IMAGE_NORMALIZED = register_modality(ModalitySpec(
    modality=Modality(
        name="image_normalized",
        shape=None,
        dtype="float32",
        description="Normalized image (ImageNet stats)"
    ),
    category=ModalityCategory.TENSOR,
    python_types=[np.ndarray],
    parent="image",
    typical_transforms_to=["embedding", "image"],
    typical_transforms_from=["image"],
))

# Embedding (dense vector)
EMBEDDING = register_modality(ModalitySpec(
    modality=Modality(
        name="embedding",
        shape=None,  # Variable dimension
        dtype="float32",
        description="Dense embedding vector"
    ),
    category=ModalityCategory.TENSOR,
    python_types=[np.ndarray, list],
    typical_transforms_to=["embedding_unit", "embedding_reduced"],
    typical_transforms_from=["text", "image", "tokens", "code"],
))

# Unit-normalized embedding
EMBEDDING_UNIT = register_modality(ModalitySpec(
    modality=Modality(
        name="embedding_unit",
        shape=None,
        dtype="float32",
        description="L2-normalized embedding (unit sphere)"
    ),
    category=ModalityCategory.TENSOR,
    python_types=[np.ndarray],
    parent="embedding",
    typical_transforms_to=[],
    typical_transforms_from=["embedding"],
))

# Reduced-dimension embedding
EMBEDDING_REDUCED = register_modality(ModalitySpec(
    modality=Modality(
        name="embedding_reduced",
        shape=None,
        dtype="float32",
        description="Dimensionality-reduced embedding"
    ),
    category=ModalityCategory.TENSOR,
    python_types=[np.ndarray],
    parent="embedding",
    typical_transforms_to=[],
    typical_transforms_from=["embedding"],
))

# JSON data
JSON_DATA = register_modality(ModalitySpec(
    modality=Modality(
        name="json_data",
        shape=None,
        dtype="object",
        description="Structured JSON data (dict or list)"
    ),
    category=ModalityCategory.STRUCTURED,
    file_extensions=[".json"],
    mime_types=["application/json"],
    python_types=[dict, list],
    typical_transforms_to=["text", "embedding", "json_flat", "json_schema"],
    typical_transforms_from=["json_file", "text"],
))

# Flattened JSON
JSON_FLAT = register_modality(ModalitySpec(
    modality=Modality(
        name="json_flat",
        shape=None,
        dtype="object",
        description="Flattened JSON (dot-notation keys)"
    ),
    category=ModalityCategory.STRUCTURED,
    python_types=[dict],
    parent="json_data",
    typical_transforms_to=["json_data", "text"],
    typical_transforms_from=["json_data"],
))

# JSON Schema
JSON_SCHEMA = register_modality(ModalitySpec(
    modality=Modality(
        name="json_schema",
        shape=None,
        dtype="object",
        description="JSON Schema (structure without values)"
    ),
    category=ModalityCategory.STRUCTURED,
    python_types=[dict],
    parent="json_data",
    typical_transforms_to=["text"],
    typical_transforms_from=["json_data"],
))


# ==================== File Modalities (Tier 2) ====================

TEXT_FILE = register_modality(ModalitySpec(
    modality=Modality(
        name="text_file",
        shape=None,
        dtype="path",
        description="Text file (.txt, .md, .rst)"
    ),
    category=ModalityCategory.FILE,
    file_extensions=[".txt", ".md", ".rst", ".text"],
    mime_types=["text/plain", "text/markdown"],
    python_types=[str],  # Path as string
    typical_transforms_to=["text"],
    typical_transforms_from=["text"],
))

IMAGE_FILE = register_modality(ModalitySpec(
    modality=Modality(
        name="image_file",
        shape=None,
        dtype="path",
        description="Image file (.png, .jpg, etc.)"
    ),
    category=ModalityCategory.FILE,
    file_extensions=[".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"],
    mime_types=["image/png", "image/jpeg", "image/webp"],
    python_types=[str],
    typical_transforms_to=["image"],
    typical_transforms_from=["image"],
))

CODE_FILE = register_modality(ModalitySpec(
    modality=Modality(
        name="code_file",
        shape=None,
        dtype="path",
        description="Source code file"
    ),
    category=ModalityCategory.CODE,
    file_extensions=[
        ".py", ".js", ".ts", ".jsx", ".tsx",  # Python, JavaScript
        ".java", ".kt", ".scala",              # JVM
        ".c", ".cpp", ".h", ".hpp",            # C/C++
        ".go", ".rs", ".swift",                # Modern systems
        ".rb", ".php", ".pl",                  # Scripting
        ".sql", ".sh", ".bash",                # Query/shell
        ".html", ".css", ".scss",              # Web
        ".yaml", ".yml", ".toml",              # Config
    ],
    mime_types=["text/x-python", "application/javascript", "text/x-java"],
    python_types=[str],
    typical_transforms_to=["code", "text"],
    typical_transforms_from=["code"],
))

JSON_FILE = register_modality(ModalitySpec(
    modality=Modality(
        name="json_file",
        shape=None,
        dtype="path",
        description="JSON file"
    ),
    category=ModalityCategory.FILE,
    file_extensions=[".json", ".jsonl"],
    mime_types=["application/json"],
    python_types=[str],
    typical_transforms_to=["json_data"],
    typical_transforms_from=["json_data"],
))


# ==================== Code Modalities ====================

CODE = register_modality(ModalitySpec(
    modality=Modality(
        name="code",
        shape=None,
        dtype="str",
        description="Source code as string (with language metadata)"
    ),
    category=ModalityCategory.CODE,
    python_types=[str],
    typical_transforms_to=["tokens", "embedding", "ast", "code_file", "functions", "code_clean"],
    typical_transforms_from=["code_file"],
))

AST = register_modality(ModalitySpec(
    modality=Modality(
        name="ast",
        shape=None,
        dtype="object",
        description="Abstract Syntax Tree"
    ),
    category=ModalityCategory.CODE,
    python_types=[dict, object],
    parent="code",
    typical_transforms_to=["embedding", "code"],
    typical_transforms_from=["code"],
))

FUNCTIONS = register_modality(ModalitySpec(
    modality=Modality(
        name="functions",
        shape=None,
        dtype="object",
        description="List of function definitions extracted from code"
    ),
    category=ModalityCategory.CODE,
    python_types=[list],
    parent="code",
    typical_transforms_to=["embedding"],
    typical_transforms_from=["code"],
))

CODE_CLEAN = register_modality(ModalitySpec(
    modality=Modality(
        name="code_clean",
        shape=None,
        dtype="str",
        description="Code with comments removed"
    ),
    category=ModalityCategory.CODE,
    python_types=[str],
    parent="code",
    typical_transforms_to=["tokens", "embedding"],
    typical_transforms_from=["code"],
))


# ==================== Web Modalities (Tier 3 - Stubs) ====================

URL = register_modality(ModalitySpec(
    modality=Modality(
        name="url",
        shape=None,
        dtype="str",
        description="URL string"
    ),
    category=ModalityCategory.WEB,
    python_types=[str],
    typical_transforms_to=["webpage", "text", "json_data"],
))

WEBPAGE = register_modality(ModalitySpec(
    modality=Modality(
        name="webpage",
        shape=None,
        dtype="object",
        description="Webpage content (HTML + metadata)"
    ),
    category=ModalityCategory.WEB,
    file_extensions=[".html", ".htm"],
    mime_types=["text/html"],
    python_types=[dict],
    typical_transforms_to=["text", "image", "embedding"],
    typical_transforms_from=["url"],
))


# ==================== Composite Modalities (Tier 3 - Stubs) ====================

DOCUMENT = register_modality(ModalitySpec(
    modality=Modality(
        name="document",
        shape=None,
        dtype="object",
        description="Multi-page document (PDF, DOCX, etc.)"
    ),
    category=ModalityCategory.COMPOSITE,
    file_extensions=[".pdf", ".docx", ".doc", ".odt"],
    mime_types=["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    python_types=[dict],
    typical_transforms_to=["text", "image"],  # Extract text or render pages
))

CODEBASE = register_modality(ModalitySpec(
    modality=Modality(
        name="codebase",
        shape=None,
        dtype="object",
        description="Directory of source code files"
    ),
    category=ModalityCategory.COMPOSITE,
    python_types=[dict],
    typical_transforms_to=["code", "embedding"],
    children=["code_file"],
))

DATASET = register_modality(ModalitySpec(
    modality=Modality(
        name="dataset",
        shape=None,
        dtype="object",
        description="Collection of data samples"
    ),
    category=ModalityCategory.COMPOSITE,
    file_extensions=[".csv", ".parquet", ".arrow"],
    mime_types=["text/csv", "application/vnd.apache.parquet"],
    python_types=[dict],
))


# ==================== Knowledge Graph Modalities ====================

OLOG = register_modality(ModalitySpec(
    modality=Modality(
        name="olog",
        shape=None,
        dtype="object",
        description="Ontology log - categorical knowledge graph (entities + relationships)"
    ),
    category=ModalityCategory.STRUCTURED,
    python_types=[dict, object],
    typical_transforms_to=["text", "triples", "embedding", "json_data"],
    typical_transforms_from=["text", "image", "embedding"],
))

TRIPLES = register_modality(ModalitySpec(
    modality=Modality(
        name="triples",
        shape=None,
        dtype="object",
        description="List of (subject, predicate, object) triples"
    ),
    category=ModalityCategory.STRUCTURED,
    python_types=[list],
    parent="olog",
    typical_transforms_to=["text", "olog", "json_data"],
    typical_transforms_from=["olog", "text"],
))

ENTITIES = register_modality(ModalitySpec(
    modality=Modality(
        name="entities",
        shape=None,
        dtype="object",
        description="List of extracted entities with attributes"
    ),
    category=ModalityCategory.STRUCTURED,
    python_types=[list],
    parent="olog",
    typical_transforms_to=["olog", "embedding"],
    typical_transforms_from=["text", "image"],
))

RELATIONSHIPS = register_modality(ModalitySpec(
    modality=Modality(
        name="relationships",
        shape=None,
        dtype="object",
        description="List of relationships between entities"
    ),
    category=ModalityCategory.STRUCTURED,
    python_types=[list],
    parent="olog",
    typical_transforms_to=["olog", "triples"],
    typical_transforms_from=["text", "image"],
))


# ==================== Media Modalities (Tier 4 - Stubs) ====================

AUDIO = register_modality(ModalitySpec(
    modality=Modality(
        name="audio",
        shape=None,  # (samples,) or (samples, channels)
        dtype="float32",
        description="Audio waveform"
    ),
    category=ModalityCategory.MEDIA,
    file_extensions=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
    mime_types=["audio/wav", "audio/mpeg", "audio/flac"],
    python_types=[np.ndarray],
    typical_transforms_to=["spectrogram", "embedding"],
))

VIDEO = register_modality(ModalitySpec(
    modality=Modality(
        name="video",
        shape=None,  # (frames, H, W, C)
        dtype="float32",
        description="Video as frame sequence"
    ),
    category=ModalityCategory.MEDIA,
    file_extensions=[".mp4", ".avi", ".mov", ".webm", ".mkv"],
    mime_types=["video/mp4", "video/x-msvideo", "video/webm"],
    python_types=[np.ndarray],
    typical_transforms_to=["image", "embedding"],  # Extract frames or encode
))

SPECTROGRAM = register_modality(ModalitySpec(
    modality=Modality(
        name="spectrogram",
        shape=None,  # (freq_bins, time_steps)
        dtype="float32",
        description="Audio spectrogram (mel or STFT)"
    ),
    category=ModalityCategory.MEDIA,
    python_types=[np.ndarray],
    parent="audio",
    typical_transforms_to=["image", "embedding"],
    typical_transforms_from=["audio"],
))


# ==================== Utility Functions ====================

def detect_modality_from_extension(path: str) -> Optional[str]:
    """Detect modality from file extension."""
    import os
    ext = os.path.splitext(path)[1].lower()
    
    for name, spec in MODALITY_REGISTRY.items():
        if ext in spec.file_extensions:
            return name
    
    return None


def detect_modality_from_mime(mime_type: str) -> Optional[str]:
    """Detect modality from MIME type."""
    for name, spec in MODALITY_REGISTRY.items():
        if mime_type in spec.mime_types:
            return name
    
    return None


def detect_modality_from_data(data: Any) -> Optional[str]:
    """Attempt to detect modality from data type and content."""
    # String types
    if isinstance(data, str):
        # Check if it's a path
        import os
        if os.path.exists(data):
            return detect_modality_from_extension(data)
        
        # Check if it's a URL
        if data.startswith(('http://', 'https://')):
            return "url"
        
        # Check if it looks like JSON
        if data.strip().startswith(('{', '[')):
            try:
                import json
                json.loads(data)
                return "json_data"
            except:
                pass
        
        # Default to text
        return "text"
    
    # Dict/list -> JSON
    if isinstance(data, (dict, list)):
        return "json_data"
    
    # Numpy array
    if isinstance(data, np.ndarray):
        ndim = data.ndim
        if ndim == 1:
            return "embedding"
        elif ndim == 2:
            # Could be grayscale image or 2D embedding
            if data.shape[0] > 100 and data.shape[1] > 100:
                return "image"
            return "embedding"
        elif ndim == 3:
            return "image"
        elif ndim == 4:
            return "video"
    
    return None


def get_modality_hierarchy() -> Dict[str, List[str]]:
    """Get the parent-child hierarchy of modalities."""
    hierarchy = {}
    
    for name, spec in MODALITY_REGISTRY.items():
        if spec.parent:
            if spec.parent not in hierarchy:
                hierarchy[spec.parent] = []
            hierarchy[spec.parent].append(name)
    
    return hierarchy
