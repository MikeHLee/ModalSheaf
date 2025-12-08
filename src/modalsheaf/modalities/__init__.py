"""
Built-in modality definitions and handlers.

This module provides pre-defined modalities for common data types.
Start with a core set, expand as needed.

Modality Hierarchy:
    
    Base Types (Tier 1 - Implemented)
    ├── text          # Raw text strings
    ├── image         # Image arrays (numpy, PIL)
    ├── embedding     # Dense vectors
    └── json_data     # Structured JSON
    
    File Types (Tier 2 - Implemented)
    ├── text_file     # .txt, .md, .rst
    ├── image_file    # .png, .jpg, .webp
    ├── code_file     # .py, .js, .ts, etc.
    └── json_file     # .json
    
    Composite Types (Tier 3 - Planned)
    ├── document      # Multi-page document
    ├── webpage       # HTML + assets
    ├── codebase      # Directory of code
    └── dataset       # Collection of samples
    
    Audio/Video (Tier 4 - Planned)
    ├── audio         # Waveforms
    ├── video         # Frame sequences
    └── spectrogram   # Audio visualization
    
    Advanced (Tier 5 - Future)
    ├── api_endpoint  # REST/GraphQL
    ├── database      # SQL/NoSQL
    └── web_app       # Full application
"""

from .base import (
    # Core modality definitions
    TEXT,
    IMAGE,
    EMBEDDING,
    JSON_DATA,
    TOKENS,
    # File-based modalities
    TEXT_FILE,
    IMAGE_FILE,
    CODE_FILE,
    JSON_FILE,
    # Knowledge graph modalities
    OLOG,
    TRIPLES,
    ENTITIES,
    RELATIONSHIPS,
    # Modality registry
    MODALITY_REGISTRY,
    get_modality,
    register_modality,
    list_modalities,
)

from .handlers import (
    # Data handlers
    TextHandler,
    ImageHandler,
    EmbeddingHandler,
    CodeHandler,
    # Handler registry
    get_handler,
    register_handler,
)

from .loaders import (
    # File loaders
    load_file,
    save_file,
    detect_modality,
)

from .transforms import (
    # Text transforms
    text_to_tokens,
    tokens_to_text,
    text_to_sentences,
    sentences_to_text,
    text_to_words,
    words_to_text,
    # Image transforms
    image_to_patches,
    patches_to_image,
    image_to_grayscale,
    grayscale_to_rgb,
    image_resize,
    image_normalize,
    image_denormalize,
    # Embedding transforms
    embedding_normalize,
    embedding_reduce_dim,
    embedding_expand_dim,
    embeddings_average,
    # Code transforms
    code_to_ast,
    ast_to_code,
    code_to_functions,
    code_to_imports,
    code_remove_comments,
    # JSON transforms
    json_to_text,
    text_to_json,
    json_flatten,
    json_unflatten,
    json_get_schema,
    # Registry
    BUILTIN_TRANSFORMS,
    get_builtin_transform,
    list_builtin_transforms,
    create_transformation_from_builtin,
)

__all__ = [
    # Modality constants
    "TEXT",
    "IMAGE", 
    "EMBEDDING",
    "JSON_DATA",
    "TOKENS",
    "TEXT_FILE",
    "IMAGE_FILE",
    "CODE_FILE",
    "JSON_FILE",
    # Knowledge graph modalities
    "OLOG",
    "TRIPLES",
    "ENTITIES",
    "RELATIONSHIPS",
    # Registry functions
    "MODALITY_REGISTRY",
    "get_modality",
    "register_modality",
    "list_modalities",
    # Handlers
    "TextHandler",
    "ImageHandler",
    "EmbeddingHandler",
    "CodeHandler",
    "get_handler",
    "register_handler",
    # Loaders
    "load_file",
    "save_file",
    "detect_modality",
    # Text transforms
    "text_to_tokens",
    "tokens_to_text",
    "text_to_sentences",
    "sentences_to_text",
    "text_to_words",
    "words_to_text",
    # Image transforms
    "image_to_patches",
    "patches_to_image",
    "image_to_grayscale",
    "grayscale_to_rgb",
    "image_resize",
    "image_normalize",
    "image_denormalize",
    # Embedding transforms
    "embedding_normalize",
    "embedding_reduce_dim",
    "embedding_expand_dim",
    "embeddings_average",
    # Code transforms
    "code_to_ast",
    "ast_to_code",
    "code_to_functions",
    "code_to_imports",
    "code_remove_comments",
    # JSON transforms
    "json_to_text",
    "text_to_json",
    "json_flatten",
    "json_unflatten",
    "json_get_schema",
    # Transform registry
    "BUILTIN_TRANSFORMS",
    "get_builtin_transform",
    "list_builtin_transforms",
    "create_transformation_from_builtin",
]
