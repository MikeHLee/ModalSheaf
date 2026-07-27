"""
Example 6: Measured Transforms with Topological Loss Characterization

Demonstrates:
- Transforms that return BOTH data AND loss characterization
- Topological characterization of information loss (not just a scalar)
- Data-dependent loss measurement
- Pipeline execution with full loss tracking
- Registry for storing and retrieving transforms

Key insight: Information loss has STRUCTURE. It's not just "70% lost" but:
- WHAT kind of information was lost (spatial, semantic, relational)
- WHERE in the data space the loss occurred  
- HOW the remaining information is shaped
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from modalsheaf import (
    # Measured transforms
    LossType,
    LossRegion,
    TopologicalLossCharacterization,
    TransformResult,
    MeasuredTransform,
    EmbeddingTransform,
    EntityExtractionTransform,
    TextGenerationTransform,
    # Registry and pipeline
    MEASURED_TRANSFORM_REGISTRY,
    register_measured_transform,
    run_measured_pipeline,
    # Knowledge structures
    Entity,
    Relationship,
    Olog,
)


# ==================== Custom Measured Transform ====================

class ImageToEmbeddingTransform(MeasuredTransform[np.ndarray, np.ndarray]):
    """
    Custom measured transform for image embedding.
    
    Demonstrates how to create a transform that:
    1. Applies a transformation function
    2. Measures the loss based on actual input/output
    3. Returns topological characterization
    """
    
    def __init__(self, embed_fn, embedding_dim: int = 768):
        super().__init__(
            source="image",
            target="embedding",
            name="image_to_embedding",
            expected_loss_types=[LossType.SPATIAL, LossType.SEMANTIC, LossType.PROJECTION]
        )
        self.embed_fn = embed_fn
        self.embedding_dim = embedding_dim
    
    def transform(self, image: np.ndarray) -> TransformResult[np.ndarray]:
        # Apply the embedding function
        embedding = self.embed_fn(image)
        
        # Measure loss based on actual image properties
        loss = self._measure_loss(image, embedding)
        
        return TransformResult(
            data=embedding,
            loss=loss,
            inversion_hints={
                "original_shape": image.shape,
                "original_dtype": str(image.dtype),
                "embedding_norm": float(np.linalg.norm(embedding)),
            }
        )
    
    def _measure_loss(
        self, 
        image: np.ndarray, 
        embedding: np.ndarray
    ) -> TopologicalLossCharacterization:
        """
        Measure loss based on actual image properties.
        
        This is where the magic happens - we analyze what was lost.
        """
        # Image properties
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        total_pixels = h * w * channels
        
        # Compression ratio
        compression_ratio = self.embedding_dim / total_pixels
        
        # Spatial loss: higher resolution = more spatial detail lost
        spatial_loss = 1 - compression_ratio
        spatial_loss = min(0.95, max(0.5, spatial_loss))
        
        # Semantic loss: depends on image complexity
        # (In practice, you'd use actual image analysis here)
        image_variance = np.var(image)
        complexity_factor = min(image_variance / 0.1, 1.0)  # Normalize
        semantic_loss = 0.6 + complexity_factor * 0.2
        
        # Total loss (weighted combination)
        total_loss = 0.4 * spatial_loss + 0.4 * semantic_loss + 0.2 * 0.8
        
        loss_regions = [
            LossRegion(
                loss_type=LossType.SPATIAL,
                magnitude=spatial_loss,
                affected_dimensions=[0, 1],  # Height and width
                description=f"Spatial detail lost: {h}x{w} -> {self.embedding_dim}D"
            ),
            LossRegion(
                loss_type=LossType.SEMANTIC,
                magnitude=semantic_loss,
                description="Visual semantics compressed"
            ),
            LossRegion(
                loss_type=LossType.PROJECTION,
                magnitude=0.8,
                affected_dimensions=list(range(self.embedding_dim)),
                description=f"Projected to {self.embedding_dim} dimensions"
            ),
        ]
        
        return TopologicalLossCharacterization(
            total_loss=total_loss,
            loss_regions=loss_regions,
            preserved_dimensions=self.embedding_dim,
            is_measured=True,
            confidence=0.75,
            metadata={
                "original_pixels": total_pixels,
                "compression_ratio": compression_ratio,
            }
        )


# ==================== Example Usage ====================

def example_basic_measured_transform():
    """Basic usage of measured transforms."""
    print("=" * 60)
    print("Example 1: Basic Measured Transform")
    print("=" * 60)
    
    # Create a simple embedding function
    def mock_text_embed(text: str) -> np.ndarray:
        # In practice, this would be CLIP, sentence-transformers, etc.
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(768).astype(np.float32)
    
    # Create measured transform
    text_encoder = EmbeddingTransform(
        source="text",
        embed_fn=mock_text_embed,
        embedding_dim=768,
        name="text_encoder",
        base_loss=0.7
    )
    
    # Transform some text
    short_text = "Hello world"
    long_text = """
    The theory of general relativity, developed by Albert Einstein between 
    1907 and 1915, describes gravity as a geometric property of space and time.
    It provides a unified description of gravity as a geometric property of 
    space and time, or four-dimensional spacetime.
    """
    
    short_result = text_encoder(short_text)
    long_result = text_encoder(long_text)
    
    print(f"\nShort text: '{short_text}'")
    print(f"  Embedding shape: {short_result.data.shape}")
    print(f"  Total loss: {short_result.loss.total_loss:.1%}")
    
    print(f"\nLong text: {len(long_text)} chars")
    print(f"  Embedding shape: {long_result.data.shape}")
    print(f"  Total loss: {long_result.loss.total_loss:.1%}")
    
    print(f"\nLoss characterization for long text:")
    print(long_result.loss.summary())
    
    # Check if loss is data-dependent
    print(f"\nIs loss data-dependent? {text_encoder.is_data_dependent}")
    print(f"Average loss from history: {text_encoder.average_loss():.1%}")


def example_image_transform():
    """Image embedding with spatial loss tracking."""
    print("\n" + "=" * 60)
    print("Example 2: Image Transform with Spatial Loss")
    print("=" * 60)
    
    # Mock image embedding function
    def mock_image_embed(image: np.ndarray) -> np.ndarray:
        return np.random.randn(768).astype(np.float32)
    
    # Create transform
    image_encoder = ImageToEmbeddingTransform(
        embed_fn=mock_image_embed,
        embedding_dim=768
    )
    
    # Test with different image sizes
    small_image = np.random.rand(64, 64, 3)
    medium_image = np.random.rand(224, 224, 3)
    large_image = np.random.rand(1024, 1024, 3)
    
    for name, image in [("64x64", small_image), ("224x224", medium_image), ("1024x1024", large_image)]:
        result = image_encoder(image)
        print(f"\n{name} image:")
        print(f"  Total loss: {result.loss.total_loss:.1%}")
        print(f"  Spatial loss: {result.loss.loss_by_type().get(LossType.SPATIAL, 0):.1%}")
        print(f"  Compression: {result.loss.metadata.get('compression_ratio', 0):.6f}")


def example_pipeline():
    """Run a full pipeline with measured loss at each step."""
    print("\n" + "=" * 60)
    print("Example 3: Full Pipeline with Loss Tracking")
    print("=" * 60)
    
    # Create transforms
    def mock_embed(text):
        return np.random.randn(768)
    
    def mock_extract(embedding):
        olog = Olog()
        olog.add_entity(Entity("e1", "a concept", confidence=0.8))
        olog.add_entity(Entity("e2", "another concept", confidence=0.7))
        olog.add_relationship(Relationship("r1", "e1", "e2", "relates to", confidence=0.75))
        return olog
    
    def mock_generate(olog):
        return " ".join(f"{s} {p} {o}." for s, p, o in olog.to_triples())
    
    text_encoder = EmbeddingTransform(
        source="text",
        embed_fn=mock_embed,
        embedding_dim=768,
        name="text_encoder"
    )
    
    entity_extractor = EntityExtractionTransform(
        source="embedding",
        extract_fn=mock_extract,
        name="entity_extractor"
    )
    
    text_generator = TextGenerationTransform(
        generate_fn=mock_generate,
        name="text_generator",
        temperature=0.7
    )
    
    # Run pipeline
    input_text = "Einstein developed the theory of relativity in 1905."
    
    print(f"\nInput: '{input_text}'")
    print("\nRunning pipeline: text -> embedding -> olog -> text")
    
    pipeline_result = run_measured_pipeline(
        input_text,
        [text_encoder, entity_extractor, text_generator]
    )
    
    print(f"\nOutput: '{pipeline_result.data}'")
    print(f"\n{pipeline_result.summary()}")
    
    print("\nDetailed loss by type across pipeline:")
    for loss_type, magnitude in pipeline_result.total_loss.loss_by_type().items():
        print(f"  {loss_type.name}: {magnitude:.1%}")


def example_loss_comparison():
    """Compare loss characteristics across different modalities."""
    print("\n" + "=" * 60)
    print("Example 4: Loss Comparison Across Modalities")
    print("=" * 60)
    
    # Different source modalities have different loss profiles
    modality_configs = {
        "text": {
            "base_loss": 0.7,
            "dominant_loss": LossType.SEMANTIC,
            "description": "Meaning compressed, word order lost"
        },
        "image": {
            "base_loss": 0.85,
            "dominant_loss": LossType.SPATIAL,
            "description": "Spatial detail lost, texture compressed"
        },
        "audio": {
            "base_loss": 0.75,
            "dominant_loss": LossType.TEMPORAL,
            "description": "Temporal dynamics compressed"
        },
        "structured_data": {
            "base_loss": 0.3,
            "dominant_loss": LossType.STRUCTURAL,
            "description": "Schema preserved, some relations lost"
        },
    }
    
    print("\nLoss profiles by source modality:")
    print("-" * 50)
    
    for modality, config in modality_configs.items():
        print(f"\n{modality.upper()}:")
        print(f"  Base loss: {config['base_loss']:.0%}")
        print(f"  Dominant loss type: {config['dominant_loss'].name}")
        print(f"  Description: {config['description']}")
        
        # Estimate round-trip
        forward_preservation = 1 - config['base_loss']
        backward_preservation = 0.7  # Assume generation is ~30% lossy
        round_trip = forward_preservation * backward_preservation
        print(f"  Round-trip preservation: {round_trip:.1%}")


def example_registry():
    """Using the transform registry."""
    print("\n" + "=" * 60)
    print("Example 5: Transform Registry")
    print("=" * 60)
    
    # Create and register transforms
    def mock_fn(x):
        return np.random.randn(768)
    
    transforms = [
        EmbeddingTransform("text", mock_fn, 768, "text_to_embedding"),
        EmbeddingTransform("image", mock_fn, 768, "image_to_embedding"),
        EmbeddingTransform("audio", mock_fn, 768, "audio_to_embedding"),
    ]
    
    for t in transforms:
        register_measured_transform(t)
    
    print("\nRegistered transforms:")
    for source, target, name in MEASURED_TRANSFORM_REGISTRY.list_all():
        print(f"  {source} -> {target}: {name}")
    
    # Look up a transform
    text_transform = MEASURED_TRANSFORM_REGISTRY.get("text", "embedding")
    if text_transform:
        print(f"\nFound transform: {text_transform.name}")
        print(f"  Expected loss types: {[lt.name for lt in text_transform.expected_loss_types]}")


def example_topological_characterization():
    """Deep dive into topological loss characterization."""
    print("\n" + "=" * 60)
    print("Example 6: Topological Loss Characterization")
    print("=" * 60)
    
    # Create a detailed loss characterization
    loss = TopologicalLossCharacterization(
        total_loss=0.75,
        loss_regions=[
            LossRegion(
                loss_type=LossType.SPATIAL,
                magnitude=0.8,
                affected_dimensions=[0, 1],
                betti_numbers=(1, 0, 0),  # Connected, no holes
                description="2D spatial structure collapsed"
            ),
            LossRegion(
                loss_type=LossType.SEMANTIC,
                magnitude=0.6,
                description="Fine-grained meaning lost"
            ),
            LossRegion(
                loss_type=LossType.RELATIONAL,
                magnitude=0.4,
                betti_numbers=(3, 2, 0),  # 3 components, 2 holes
                description="Some entity relationships not captured"
            ),
        ],
        preserved_dimensions=768,
        preserved_betti=(1, 0, 0),  # Single connected component preserved
        loss_entropy=0.7,  # Fairly uniform loss
        is_measured=True,
        confidence=0.85,
    )
    
    print("\nDetailed loss characterization:")
    print(loss.summary())
    
    print(f"\nTopological invariants:")
    print(f"  Preserved Betti numbers: {loss.preserved_betti}")
    print(f"  (b0=connected components, b1=holes, b2=voids)")
    
    print(f"\nLoss distribution:")
    print(f"  Entropy: {loss.loss_entropy:.2f} (1.0 = uniform, 0.0 = concentrated)")
    print(f"  Dominant loss type: {loss.dominant_loss_type().name}")
    
    print(f"\nLoss by type:")
    for lt, mag in sorted(loss.loss_by_type().items(), key=lambda x: -x[1]):
        print(f"  {lt.name}: {mag:.1%}")


# ==================== Main ====================

if __name__ == "__main__":
    example_basic_measured_transform()
    example_image_transform()
    example_pipeline()
    example_loss_comparison()
    example_registry()
    example_topological_characterization()
    
    print("\n" + "=" * 60)
    print("Summary: Measured Transforms")
    print("=" * 60)
    print("""
Key concepts:

1. MEASURED TRANSFORMS return TransformResult containing:
   - data: The transformed output
   - loss: TopologicalLossCharacterization
   - inversion_hints: Info that might help reverse the transform

2. TOPOLOGICAL LOSS CHARACTERIZATION includes:
   - total_loss: Scalar for backward compatibility
   - loss_regions: List of LossRegion with type, magnitude, affected dims
   - preserved_dimensions: Effective dimensionality of output
   - betti_numbers: Topological invariants (components, holes, voids)
   - is_measured: Whether computed from actual data or estimated

3. LOSS TYPES categorize what kind of information was lost:
   - SPATIAL: Position/location detail
   - TEMPORAL: Sequence/time detail
   - SEMANTIC: Meaning/conceptual detail
   - RELATIONAL: Relationships between elements
   - STRUCTURAL: Graph/topology structure
   - PROJECTION: Dimensionality reduction

4. PIPELINES track loss at each step:
   - Cumulative loss computed multiplicatively
   - Loss types aggregated across steps
   - Full audit trail of what was lost where

5. REGISTRY stores transforms for lookup:
   - Find transforms by (source, target) pair
   - Build pipelines dynamically
   - Track historical loss for calibration
""")
