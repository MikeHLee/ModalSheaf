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
    # Knowledge graph modalities
    OLOG, TRIPLES, ENTITIES, RELATIONSHIPS,
    # Registry
    get_modality, register_modality, list_modalities,
    # Handlers
    get_handler, register_handler,
    # Loaders
    load_file, save_file, detect_modality,
)

# Knowledge graph / Olog support
from .knowledge import (
    # Data structures
    Entity,
    Relationship,
    Olog,
    # Information loss tracking
    LossWarningLevel,
    InfoLossReport,
    estimate_info_loss,
    # LLM integration
    LLMPipelineConfig,
    KnowledgeExtractor,
    StructuredGenerator,
    # Analysis
    analyze_round_trip,
)

# Measured transforms with topological loss characterization
from .measured_transforms import (
    # Loss characterization
    LossType,
    LossRegion,
    TopologicalLossCharacterization,
    TransformResult,
    # Transform base classes
    MeasuredTransform,
    EmbeddingTransform,
    EntityExtractionTransform,
    TextGenerationTransform,
    # Registry
    MeasuredTransformRegistry,
    MEASURED_TRANSFORM_REGISTRY,
    register_measured_transform,
    # Pipeline
    PipelineResult,
    run_measured_pipeline,
)

# Advanced: Rigorous Čech cohomology
from .cech import (
    CechCochain,
    CechComplex,
    CohomologyGroup,
    CechCohomology,
    CohomologyResult,
    compute_cech_cohomology,
)

# Advanced: Persistent cohomology for noisy data
from .persistence import (
    PersistenceInterval,
    PersistenceDiagram,
    PersistentCohomologyResult,
    PersistentCohomology,
    compute_persistent_cohomology,
    persistence_based_consistency,
)

# Advanced: Cocycle conditions for gluing
from .cocycles import (
    TransitionFunction,
    CocycleViolation,
    CocycleCheckResult,
    CocycleChecker,
    CocycleRepairer,
    check_cocycle,
    repair_cocycle,
)

# Applications: Domain-specific sheaf implementations
from .applications import (
    BrainSheaf,
    BrainRegion,
    DissonanceResult,
    PersistentCycleResult,
    load_fmri_data,
    load_connectivity_matrix,
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
    # Knowledge graph modalities
    "OLOG", "TRIPLES", "ENTITIES", "RELATIONSHIPS",
    # Modality registry
    "get_modality", "register_modality", "list_modalities",
    # Handlers
    "get_handler", "register_handler",
    # Loaders
    "load_file", "save_file", "detect_modality",
    # Knowledge graph / Olog support
    "Entity",
    "Relationship", 
    "Olog",
    "LossWarningLevel",
    "InfoLossReport",
    "estimate_info_loss",
    "LLMPipelineConfig",
    "KnowledgeExtractor",
    "StructuredGenerator",
    "analyze_round_trip",
    # Measured transforms
    "LossType",
    "LossRegion",
    "TopologicalLossCharacterization",
    "TransformResult",
    "MeasuredTransform",
    "EmbeddingTransform",
    "EntityExtractionTransform",
    "TextGenerationTransform",
    "MeasuredTransformRegistry",
    "MEASURED_TRANSFORM_REGISTRY",
    "register_measured_transform",
    "PipelineResult",
    "run_measured_pipeline",
    # Advanced: Čech cohomology
    "CechCochain",
    "CechComplex",
    "CohomologyGroup",
    "CechCohomology",
    "CohomologyResult",
    "compute_cech_cohomology",
    # Advanced: Persistent cohomology
    "PersistenceInterval",
    "PersistenceDiagram",
    "PersistentCohomologyResult",
    "PersistentCohomology",
    "compute_persistent_cohomology",
    "persistence_based_consistency",
    # Advanced: Cocycle conditions
    "TransitionFunction",
    "CocycleViolation",
    "CocycleCheckResult",
    "CocycleChecker",
    "CocycleRepairer",
    "check_cocycle",
    "repair_cocycle",
    # Applications: Neuro-topological
    "BrainSheaf",
    "BrainRegion",
    "DissonanceResult",
    "PersistentCycleResult",
    "load_fmri_data",
    "load_connectivity_matrix",
]
