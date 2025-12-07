"""
ModalSheaf - Practical sheaf-theoretic tools for multimodal ML data transformations.

This library provides intuitive APIs for:
- Defining modalities and transformations between them
- Tracking information loss in transformations
- Measuring consistency across multimodal data
- Computing cohomology for data fusion diagnostics
- Loading/saving various data modalities (text, images, code, etc.)
"""

from .core import Modality, Transformation, TransformationType
from .graph import ModalityGraph
from .consistency import ConsistencyChecker, CohomologyResult
from .transforms import (
    register_transform,
    compose_transforms,
    invert_transform,
)

# Gluing operations
from .gluing import (
    LocalSection,
    Overlap,
    GluingResult,
    GluingProtocol,
    PanoramaGluing,
    CoordinateGluing,
    HierarchicalGluing,
    DocumentGluing,
    CodebaseGluing,
    glue_with_protocol,
)

# Diagnostics
from .diagnostics import (
    ContributorScore,
    ClusterAnalysis,
    DiagnosticReport,
    DiagnosticAnalyzer,
    TemporalAnalyzer,
    ConsensusBuilder,
    diagnose_gluing_problem,
    find_consensus,
)

# Modality system
from .modalities import (
    # Modality constants
    TEXT, IMAGE, EMBEDDING, JSON_DATA, TOKENS,
    TEXT_FILE, IMAGE_FILE, CODE_FILE, JSON_FILE,
    # Registry
    get_modality, register_modality, list_modalities,
    # Handlers
    get_handler, register_handler,
    # Loaders
    load_file, save_file, detect_modality,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Modality",
    "Transformation",
    "TransformationType",
    "ModalityGraph",
    # Consistency
    "ConsistencyChecker",
    "CohomologyResult",
    # Transform utilities
    "register_transform",
    "compose_transforms",
    "invert_transform",
    # Gluing
    "LocalSection",
    "Overlap",
    "GluingResult",
    "GluingProtocol",
    "PanoramaGluing",
    "CoordinateGluing",
    "HierarchicalGluing",
    "DocumentGluing",
    "CodebaseGluing",
    "glue_with_protocol",
    # Diagnostics
    "ContributorScore",
    "ClusterAnalysis",
    "DiagnosticReport",
    "DiagnosticAnalyzer",
    "TemporalAnalyzer",
    "ConsensusBuilder",
    "diagnose_gluing_problem",
    "find_consensus",
    # Modality constants
    "TEXT", "IMAGE", "EMBEDDING", "JSON_DATA", "TOKENS",
    "TEXT_FILE", "IMAGE_FILE", "CODE_FILE", "JSON_FILE",
    # Modality registry
    "get_modality", "register_modality", "list_modalities",
    # Handlers
    "get_handler", "register_handler",
    # Loaders
    "load_file", "save_file", "detect_modality",
]
