"""
Knowledge graph and olog (ontology log) support for ModalSheaf.

This module provides:
- Entity and Relationship extraction from text/images
- Olog (ontology log) construction as a categorical structure
- Structured language generation from graphs
- Round-trip fidelity estimation
- LLM integration with information loss warnings

An olog is a category where:
- Objects are "types" (e.g., "a person", "a city")
- Morphisms are "aspects" (e.g., "lives in", "was born in")
- Composition represents transitive relationships

References:
- Spivak, D.I. (2014). Category Theory for the Sciences. MIT Press.
- Spivak, D.I. & Kent, R.E. (2012). Ologs: A Categorical Framework for Knowledge Representation.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import warnings
import numpy as np

from .core import Modality, Transformation, TransformationType


# ==================== Data Structures ====================

@dataclass
class Entity:
    """
    An entity in a knowledge graph (object in an olog).
    
    In olog terms, this is a "type" - a collection of things sharing a property.
    The label should read as "a thing which..." or "an X".
    
    Examples:
        Entity("person", label="a person")
        Entity("city", label="a city", attributes={"population": int})
    """
    id: str
    label: str  # Human-readable, should start with "a" or "an"
    entity_type: str = "thing"
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Extraction confidence
    source_span: Optional[Tuple[int, int]] = None  # Character offsets in source
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False


@dataclass 
class Relationship:
    """
    A relationship between entities (morphism in an olog).
    
    In olog terms, this is an "aspect" - a functional relationship.
    The label should read as a verb phrase: "lives in", "wrote", "is parent of".
    
    Examples:
        Relationship("r1", source="person_1", target="city_1", label="lives in")
    """
    id: str
    source: str  # Source entity ID
    target: str  # Target entity ID
    label: str   # Verb phrase
    relationship_type: str = "related_to"
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Olog:
    """
    An ontology log - a categorical knowledge representation.
    
    An olog is a labeled graph where:
    - Nodes are types (entities)
    - Edges are aspects (functional relationships)
    - Paths compose (if A->B->C, then A->C exists implicitly)
    
    This structure is ideal for:
    - Representing extracted knowledge
    - Checking consistency (commutative diagrams)
    - Generating structured descriptions
    """
    entities: Dict[str, Entity] = field(default_factory=dict)
    relationships: Dict[str, Relationship] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the olog."""
        self.entities[entity.id] = entity
    
    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship to the olog."""
        if rel.source not in self.entities:
            raise ValueError(f"Source entity '{rel.source}' not found")
        if rel.target not in self.entities:
            raise ValueError(f"Target entity '{rel.target}' not found")
        self.relationships[rel.id] = rel
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)
    
    def get_relationships_from(self, entity_id: str) -> List[Relationship]:
        """Get all relationships originating from an entity."""
        return [r for r in self.relationships.values() if r.source == entity_id]
    
    def get_relationships_to(self, entity_id: str) -> List[Relationship]:
        """Get all relationships pointing to an entity."""
        return [r for r in self.relationships.values() if r.target == entity_id]
    
    def to_triples(self) -> List[Tuple[str, str, str]]:
        """Convert to (subject, predicate, object) triples."""
        triples = []
        for rel in self.relationships.values():
            subj = self.entities[rel.source].label
            pred = rel.label
            obj = self.entities[rel.target].label
            triples.append((subj, pred, obj))
        return triples
    
    def average_confidence(self) -> float:
        """Compute average extraction confidence."""
        confidences = [e.confidence for e in self.entities.values()]
        confidences += [r.confidence for r in self.relationships.values()]
        return np.mean(confidences) if confidences else 0.0
    
    @property
    def num_entities(self) -> int:
        return len(self.entities)
    
    @property
    def num_relationships(self) -> int:
        return len(self.relationships)
    
    def __repr__(self) -> str:
        return f"Olog(entities={self.num_entities}, relationships={self.num_relationships})"


# ==================== Information Loss Tracking ====================

class LossWarningLevel(Enum):
    """Warning levels for information loss."""
    NONE = auto()      # No warning (< 10% loss)
    LOW = auto()       # Informational (10-30% loss)
    MEDIUM = auto()    # Warning (30-60% loss)
    HIGH = auto()      # Strong warning (60-80% loss)
    CRITICAL = auto()  # Error-level (> 80% loss)


@dataclass
class InfoLossReport:
    """
    Report on information loss through a transformation pipeline.
    
    Provides detailed breakdown of where information is lost
    and recommendations for the user.
    """
    total_loss: float  # 0.0 to 1.0
    preservation_rate: float  # 1.0 - total_loss
    warning_level: LossWarningLevel
    steps: List[Dict[str, Any]]  # Per-step breakdown
    recommendations: List[str]
    
    @property
    def round_trip_success_probability(self) -> float:
        """
        Estimate probability of successful round-trip reconstruction.
        
        This is a rough estimate based on information preservation.
        Actual success depends on the specific data and models used.
        """
        return self.preservation_rate ** 2  # Square for round-trip
    
    def format_warning(self) -> str:
        """Format a human-readable warning message."""
        if self.warning_level == LossWarningLevel.NONE:
            return ""
        
        level_emoji = {
            LossWarningLevel.LOW: "ℹ️",
            LossWarningLevel.MEDIUM: "⚠️",
            LossWarningLevel.HIGH: "🔶",
            LossWarningLevel.CRITICAL: "🚨",
        }
        
        emoji = level_emoji.get(self.warning_level, "")
        pct = int(self.total_loss * 100)
        
        msg = f"{emoji} Information Loss Warning: ~{pct}% of original detail may be lost.\n"
        msg += f"   Round-trip reconstruction probability: ~{int(self.round_trip_success_probability * 100)}%\n"
        
        if self.recommendations:
            msg += "   Recommendations:\n"
            for rec in self.recommendations[:3]:  # Top 3
                msg += f"   - {rec}\n"
        
        return msg


def estimate_info_loss(
    transformations: List[Transformation],
    warn: bool = True
) -> InfoLossReport:
    """
    Estimate information loss through a transformation pipeline.
    
    Args:
        transformations: List of transformations to analyze
        warn: If True, emit warnings for significant loss
    
    Returns:
        InfoLossReport with detailed breakdown
    """
    if not transformations:
        return InfoLossReport(
            total_loss=0.0,
            preservation_rate=1.0,
            warning_level=LossWarningLevel.NONE,
            steps=[],
            recommendations=[]
        )
    
    # Calculate cumulative loss
    preservation = 1.0
    steps = []
    
    for t in transformations:
        step_preservation = 1.0 - t.info_loss_estimate
        preservation *= step_preservation
        
        steps.append({
            "name": t.name,
            "source": t.source,
            "target": t.target,
            "loss": t.info_loss_estimate,
            "type": t.transform_type.name,
            "cumulative_preservation": preservation,
        })
    
    total_loss = 1.0 - preservation
    
    # Determine warning level
    if total_loss < 0.1:
        level = LossWarningLevel.NONE
    elif total_loss < 0.3:
        level = LossWarningLevel.LOW
    elif total_loss < 0.6:
        level = LossWarningLevel.MEDIUM
    elif total_loss < 0.8:
        level = LossWarningLevel.HIGH
    else:
        level = LossWarningLevel.CRITICAL
    
    # Generate recommendations
    recommendations = []
    
    # Find the lossiest step
    if steps:
        worst_step = max(steps, key=lambda s: s["loss"])
        if worst_step["loss"] > 0.3:
            recommendations.append(
                f"Consider alternatives to '{worst_step['name']}' "
                f"({int(worst_step['loss']*100)}% loss)"
            )
    
    # Check for lossy embeddings
    lossy_embeddings = [s for s in steps if "embedding" in s["target"].lower() and s["loss"] > 0.5]
    if lossy_embeddings:
        recommendations.append(
            "Embedding step loses significant detail. "
            "Consider preserving original data alongside embeddings."
        )
    
    # Check for LLM-bound data
    llm_targets = ["tokens", "prompt", "context"]
    llm_steps = [s for s in steps if any(t in s["target"].lower() for t in llm_targets)]
    if llm_steps and total_loss > 0.3:
        recommendations.append(
            "Data being sent to LLM has lost detail. "
            "Consider including original source or structured summary."
        )
    
    if total_loss > 0.7:
        recommendations.append(
            "High information loss detected. "
            "Round-trip reconstruction will likely fail."
        )
    
    report = InfoLossReport(
        total_loss=total_loss,
        preservation_rate=preservation,
        warning_level=level,
        steps=steps,
        recommendations=recommendations
    )
    
    # Emit warning if requested
    if warn and level.value >= LossWarningLevel.MEDIUM.value:
        warning_msg = report.format_warning()
        if warning_msg:
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
    
    return report


# ==================== LLM Integration ====================

@dataclass
class LLMPipelineConfig:
    """
    Configuration for LLM-based extraction/generation pipelines.
    
    Tracks information loss at each stage and provides warnings.
    """
    # Extraction settings
    extraction_model: str = "gpt-4"
    extraction_temperature: float = 0.0
    
    # Generation settings  
    generation_model: str = "gpt-4"
    generation_temperature: float = 0.7
    
    # Loss estimates (can be calibrated)
    text_to_embedding_loss: float = 0.7  # High - semantic compression
    image_to_embedding_loss: float = 0.8  # Very high - visual detail lost
    embedding_to_entities_loss: float = 0.4  # Medium - structure extraction
    entities_to_text_loss: float = 0.3  # Low-medium - generation is creative
    
    # Warning settings
    warn_on_loss: bool = True
    loss_threshold_warn: float = 0.3
    loss_threshold_error: float = 0.8


class KnowledgeExtractor:
    """
    Extract knowledge graphs/ologs from text and images.
    
    This is a base class - actual extraction requires LLM integration.
    Subclass and implement _extract_from_text/_extract_from_image.
    
    Example:
        >>> extractor = KnowledgeExtractor(config)
        >>> olog, report = extractor.extract(text="Einstein was born in Ulm.")
        >>> print(olog.to_triples())
        [('a person named Einstein', 'was born in', 'a city named Ulm')]
    """
    
    def __init__(self, config: Optional[LLMPipelineConfig] = None):
        self.config = config or LLMPipelineConfig()
        self._extraction_history: List[InfoLossReport] = []
    
    def extract(
        self,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> Tuple[Olog, InfoLossReport]:
        """
        Extract an olog from input data.
        
        Args:
            text: Raw text to extract from
            image: Image array to extract from
            embedding: Pre-computed embedding
        
        Returns:
            Tuple of (extracted Olog, InfoLossReport)
        """
        transformations = []
        
        # Track transformations based on input type
        if image is not None:
            transformations.append(Transformation(
                source="image",
                target="embedding",
                forward=lambda x: x,  # Placeholder
                info_loss_estimate=self.config.image_to_embedding_loss,
                transform_type=TransformationType.LOSSY,
                name="image_to_embedding"
            ))
        
        if text is not None:
            transformations.append(Transformation(
                source="text", 
                target="embedding",
                forward=lambda x: x,
                info_loss_estimate=self.config.text_to_embedding_loss,
                transform_type=TransformationType.LOSSY,
                name="text_to_embedding"
            ))
        
        # Embedding to entities
        transformations.append(Transformation(
            source="embedding",
            target="olog",
            forward=lambda x: x,
            info_loss_estimate=self.config.embedding_to_entities_loss,
            transform_type=TransformationType.LOSSY,
            name="embedding_to_olog"
        ))
        
        # Estimate loss
        report = estimate_info_loss(
            transformations, 
            warn=self.config.warn_on_loss
        )
        self._extraction_history.append(report)
        
        # Perform extraction (subclass implements)
        if text is not None:
            olog = self._extract_from_text(text)
        elif image is not None:
            olog = self._extract_from_image(image)
        elif embedding is not None:
            olog = self._extract_from_embedding(embedding)
        else:
            raise ValueError("Must provide text, image, or embedding")
        
        return olog, report
    
    def _extract_from_text(self, text: str) -> Olog:
        """
        Extract olog from text. Override in subclass.
        
        Default implementation uses simple heuristics.
        For production, integrate with an LLM or NER model.
        """
        # Placeholder - returns empty olog
        # Real implementation would call LLM
        return Olog(metadata={"source": "text", "raw_length": len(text)})
    
    def _extract_from_image(self, image: Any) -> Olog:
        """Extract olog from image. Override in subclass."""
        return Olog(metadata={"source": "image"})
    
    def _extract_from_embedding(self, embedding: np.ndarray) -> Olog:
        """Extract olog from embedding. Override in subclass."""
        return Olog(metadata={"source": "embedding", "dim": embedding.shape})


class StructuredGenerator:
    """
    Generate structured language from knowledge graphs/ologs.
    
    Provides probability estimates for generation fidelity.
    
    Example:
        >>> generator = StructuredGenerator(config)
        >>> text, report = generator.generate(olog, style="narrative")
        >>> print(f"Generated with {report.preservation_rate:.0%} fidelity")
    """
    
    def __init__(self, config: Optional[LLMPipelineConfig] = None):
        self.config = config or LLMPipelineConfig()
    
    def generate(
        self,
        olog: Olog,
        style: str = "factual",
        include_confidence: bool = False
    ) -> Tuple[str, InfoLossReport]:
        """
        Generate text from an olog.
        
        Args:
            olog: The knowledge graph to verbalize
            style: Generation style ("factual", "narrative", "technical")
            include_confidence: Include confidence scores in output
        
        Returns:
            Tuple of (generated text, InfoLossReport)
        """
        # Track generation transformation
        transformations = [
            Transformation(
                source="olog",
                target="text",
                forward=lambda x: x,
                info_loss_estimate=self.config.entities_to_text_loss,
                transform_type=TransformationType.LOSSY,
                name="olog_to_text"
            )
        ]
        
        report = estimate_info_loss(
            transformations,
            warn=self.config.warn_on_loss
        )
        
        # Generate text (subclass implements)
        text = self._generate_text(olog, style, include_confidence)
        
        return text, report
    
    def _generate_text(
        self, 
        olog: Olog, 
        style: str,
        include_confidence: bool
    ) -> str:
        """
        Generate text from olog. Override in subclass.
        
        Default implementation creates simple sentences from triples.
        """
        if not olog.relationships:
            return ""
        
        sentences = []
        for subj, pred, obj in olog.to_triples():
            sentence = f"{subj.capitalize()} {pred} {obj}."
            sentences.append(sentence)
        
        text = " ".join(sentences)
        
        if include_confidence:
            conf = olog.average_confidence()
            text += f" [Confidence: {conf:.0%}]"
        
        return text


# ==================== Round-Trip Analysis ====================

def analyze_round_trip(
    source_modality: str,
    target_modality: str,
    transformations: List[Transformation],
    inverse_transformations: Optional[List[Transformation]] = None
) -> Dict[str, Any]:
    """
    Analyze the fidelity of a round-trip transformation.
    
    Args:
        source_modality: Starting modality (e.g., "text")
        target_modality: Intermediate modality (e.g., "olog")
        transformations: Forward transformations
        inverse_transformations: Return transformations (if different)
    
    Returns:
        Analysis dict with success probability and recommendations
    """
    # Forward pass
    forward_report = estimate_info_loss(transformations, warn=False)
    
    # Backward pass
    if inverse_transformations:
        backward_report = estimate_info_loss(inverse_transformations, warn=False)
    else:
        # Assume symmetric loss
        backward_report = forward_report
    
    # Combined round-trip
    round_trip_preservation = (
        forward_report.preservation_rate * 
        backward_report.preservation_rate
    )
    
    return {
        "source": source_modality,
        "target": target_modality,
        "forward_loss": forward_report.total_loss,
        "backward_loss": backward_report.total_loss,
        "round_trip_preservation": round_trip_preservation,
        "success_probability": round_trip_preservation,
        "forward_steps": forward_report.steps,
        "backward_steps": backward_report.steps,
        "recommendations": (
            forward_report.recommendations + 
            backward_report.recommendations
        ),
    }


# ==================== Modality Registration ====================

# Register olog as a modality (to be added to base.py)
OLOG_MODALITY = Modality(
    name="olog",
    shape=None,
    dtype="object",
    description="Ontology log - categorical knowledge graph"
)

ENTITY_MODALITY = Modality(
    name="entity",
    shape=None,
    dtype="object", 
    description="Single entity with attributes"
)

RELATIONSHIP_MODALITY = Modality(
    name="relationship",
    shape=None,
    dtype="object",
    description="Relationship between entities"
)

TRIPLES_MODALITY = Modality(
    name="triples",
    shape=None,
    dtype="object",
    description="List of (subject, predicate, object) triples"
)
