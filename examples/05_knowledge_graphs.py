"""
Example 5: Knowledge Graphs and Ologs

Demonstrates:
- Extracting entity-relationship graphs (ologs) from text/images
- Generating structured language from graphs
- Tracking information loss through the pipeline
- Round-trip fidelity analysis
- LLM integration with loss warnings

An olog (ontology log) is a categorical representation of knowledge:
- Objects are types ("a person", "a city")
- Morphisms are aspects ("lives in", "was born in")
- Composition gives transitive relationships
"""

from modalsheaf import (
    # Core
    ModalityGraph,
    Transformation,
    TransformationType,
    # Knowledge graph
    Entity,
    Relationship,
    Olog,
    KnowledgeExtractor,
    StructuredGenerator,
    LLMPipelineConfig,
    # Analysis
    estimate_info_loss,
    analyze_round_trip,
    InfoLossReport,
)


# ==================== Basic Olog Construction ====================

def example_manual_olog():
    """Manually construct an olog from known facts."""
    print("=" * 60)
    print("Example 1: Manual Olog Construction")
    print("=" * 60)
    
    # Create an olog about Einstein
    olog = Olog()
    
    # Add entities (types in categorical terms)
    olog.add_entity(Entity(
        id="einstein",
        label="a physicist named Albert Einstein",
        entity_type="person",
        attributes={"birth_year": 1879, "death_year": 1955}
    ))
    
    olog.add_entity(Entity(
        id="ulm",
        label="a city named Ulm",
        entity_type="location",
        attributes={"country": "Germany"}
    ))
    
    olog.add_entity(Entity(
        id="relativity",
        label="the theory of relativity",
        entity_type="theory",
        attributes={"year": 1905}
    ))
    
    olog.add_entity(Entity(
        id="nobel",
        label="the Nobel Prize in Physics",
        entity_type="award",
        attributes={"year": 1921}
    ))
    
    # Add relationships (aspects/morphisms)
    olog.add_relationship(Relationship(
        id="r1",
        source="einstein",
        target="ulm",
        label="was born in",
        relationship_type="birthplace"
    ))
    
    olog.add_relationship(Relationship(
        id="r2",
        source="einstein",
        target="relativity",
        label="developed",
        relationship_type="created"
    ))
    
    olog.add_relationship(Relationship(
        id="r3",
        source="einstein",
        target="nobel",
        label="received",
        relationship_type="awarded"
    ))
    
    print(f"Created: {olog}")
    print(f"\nTriples:")
    for subj, pred, obj in olog.to_triples():
        print(f"  ({subj}, {pred}, {obj})")
    
    return olog


# ==================== Information Loss Tracking ====================

def example_info_loss_tracking():
    """Track information loss through a transformation pipeline."""
    print("\n" + "=" * 60)
    print("Example 2: Information Loss Tracking")
    print("=" * 60)
    
    # Define a typical text -> olog -> text pipeline
    transformations = [
        Transformation(
            source="text",
            target="embedding",
            forward=lambda x: x,  # Placeholder
            info_loss_estimate=0.7,  # High loss - semantic compression
            transform_type=TransformationType.LOSSY,
            name="text_to_embedding"
        ),
        Transformation(
            source="embedding",
            target="entities",
            forward=lambda x: x,
            info_loss_estimate=0.4,  # Medium loss - structure extraction
            transform_type=TransformationType.LOSSY,
            name="embedding_to_entities"
        ),
        Transformation(
            source="entities",
            target="olog",
            forward=lambda x: x,
            info_loss_estimate=0.1,  # Low loss - just organizing
            transform_type=TransformationType.LOSSY,
            name="entities_to_olog"
        ),
    ]
    
    # Estimate loss (will emit warning if significant)
    report = estimate_info_loss(transformations, warn=True)
    
    print(f"\nPipeline Analysis:")
    print(f"  Total information loss: {report.total_loss:.1%}")
    print(f"  Preservation rate: {report.preservation_rate:.1%}")
    print(f"  Warning level: {report.warning_level.name}")
    print(f"  Round-trip success probability: {report.round_trip_success_probability:.1%}")
    
    print(f"\nPer-step breakdown:")
    for step in report.steps:
        print(f"  {step['name']}: {step['loss']:.0%} loss "
              f"(cumulative preservation: {step['cumulative_preservation']:.1%})")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    return report


# ==================== Round-Trip Analysis ====================

def example_round_trip():
    """Analyze round-trip fidelity: text -> olog -> text."""
    print("\n" + "=" * 60)
    print("Example 3: Round-Trip Analysis")
    print("=" * 60)
    
    # Forward: text -> olog
    forward_transforms = [
        Transformation(
            source="text",
            target="embedding",
            forward=lambda x: x,
            info_loss_estimate=0.7,
            name="encode_text"
        ),
        Transformation(
            source="embedding",
            target="olog",
            forward=lambda x: x,
            info_loss_estimate=0.4,
            name="extract_olog"
        ),
    ]
    
    # Backward: olog -> text
    backward_transforms = [
        Transformation(
            source="olog",
            target="text",
            forward=lambda x: x,
            info_loss_estimate=0.3,  # Generation is creative
            name="generate_text"
        ),
    ]
    
    analysis = analyze_round_trip(
        source_modality="text",
        target_modality="olog",
        transformations=forward_transforms,
        inverse_transformations=backward_transforms
    )
    
    print(f"\nRound-trip: text -> olog -> text")
    print(f"  Forward loss: {analysis['forward_loss']:.1%}")
    print(f"  Backward loss: {analysis['backward_loss']:.1%}")
    print(f"  Round-trip preservation: {analysis['round_trip_preservation']:.1%}")
    print(f"  Success probability: {analysis['success_probability']:.1%}")
    
    # Compare different modalities
    print("\n--- Comparison of different source modalities ---")
    
    modality_losses = {
        "text": 0.7,
        "image": 0.85,
        "audio": 0.75,
        "structured_data": 0.2,
    }
    
    for modality, embed_loss in modality_losses.items():
        forward = [
            Transformation(
                source=modality, target="embedding",
                forward=lambda x: x, info_loss_estimate=embed_loss,
                name=f"{modality}_to_embedding"
            ),
            Transformation(
                source="embedding", target="olog",
                forward=lambda x: x, info_loss_estimate=0.4,
                name="embedding_to_olog"
            ),
        ]
        backward = [
            Transformation(
                source="olog", target=modality,
                forward=lambda x: x, info_loss_estimate=0.3,
                name=f"olog_to_{modality}"
            ),
        ]
        
        result = analyze_round_trip(modality, "olog", forward, backward)
        print(f"  {modality:20s} -> olog -> {modality:20s}: "
              f"{result['success_probability']:.1%} success")


# ==================== LLM Pipeline with Warnings ====================

def example_llm_pipeline():
    """Demonstrate LLM pipeline with information loss warnings."""
    print("\n" + "=" * 60)
    print("Example 4: LLM Pipeline with Loss Warnings")
    print("=" * 60)
    
    # Configure the pipeline
    config = LLMPipelineConfig(
        extraction_model="gpt-4",
        generation_model="gpt-4",
        text_to_embedding_loss=0.7,
        image_to_embedding_loss=0.85,
        embedding_to_entities_loss=0.4,
        entities_to_text_loss=0.3,
        warn_on_loss=True,
        loss_threshold_warn=0.3,
    )
    
    print(f"Pipeline configuration:")
    print(f"  Text -> Embedding loss: {config.text_to_embedding_loss:.0%}")
    print(f"  Image -> Embedding loss: {config.image_to_embedding_loss:.0%}")
    print(f"  Embedding -> Entities loss: {config.embedding_to_entities_loss:.0%}")
    print(f"  Entities -> Text loss: {config.entities_to_text_loss:.0%}")
    
    # Create extractor and generator
    extractor = KnowledgeExtractor(config)
    generator = StructuredGenerator(config)
    
    # Simulate extraction from text
    print("\n--- Extracting from text ---")
    sample_text = "Albert Einstein was born in Ulm, Germany in 1879."
    
    # This will trigger a warning due to high loss
    olog, extract_report = extractor.extract(text=sample_text)
    
    print(f"Extraction report:")
    print(f"  Total loss: {extract_report.total_loss:.1%}")
    print(f"  Warning level: {extract_report.warning_level.name}")
    
    # Simulate extraction from image (even higher loss)
    print("\n--- Extracting from image ---")
    import numpy as np
    sample_image = np.zeros((224, 224, 3))  # Placeholder
    
    olog_img, img_report = extractor.extract(image=sample_image)
    
    print(f"Image extraction report:")
    print(f"  Total loss: {img_report.total_loss:.1%}")
    print(f"  Warning level: {img_report.warning_level.name}")
    
    # Generate text from olog
    print("\n--- Generating from olog ---")
    manual_olog = example_manual_olog()
    
    generated_text, gen_report = generator.generate(
        manual_olog, 
        style="factual",
        include_confidence=True
    )
    
    print(f"\nGenerated text: {generated_text}")
    print(f"Generation loss: {gen_report.total_loss:.1%}")


# ==================== Integration with ModalityGraph ====================

def example_modality_graph_integration():
    """Show how ologs integrate with the ModalityGraph system."""
    print("\n" + "=" * 60)
    print("Example 5: ModalityGraph Integration")
    print("=" * 60)
    
    # Create a graph with olog support
    graph = ModalityGraph("knowledge_extraction")
    
    # Add modalities
    graph.add_modality("text", dtype="str", description="Raw text")
    graph.add_modality("image", shape=(224, 224, 3), description="Image array")
    graph.add_modality("embedding", shape=(768,), description="Dense vector")
    graph.add_modality("olog", dtype="object", description="Knowledge graph")
    graph.add_modality("triples", dtype="object", description="RDF-style triples")
    
    # Add transformations with loss estimates
    graph.add_transformation(
        "text", "embedding",
        forward=lambda x: x,  # Placeholder for real encoder
        info_loss="high",
        name="text_encoder"
    )
    
    graph.add_transformation(
        "image", "embedding",
        forward=lambda x: x,
        info_loss=0.85,  # Even higher for images
        name="image_encoder"
    )
    
    graph.add_transformation(
        "embedding", "olog",
        forward=lambda x: Olog(),  # Placeholder
        info_loss="medium",
        name="entity_extractor"
    )
    
    graph.add_transformation(
        "olog", "triples",
        forward=lambda o: o.to_triples() if isinstance(o, Olog) else [],
        inverse=lambda t: Olog(),  # Simplified inverse
        info_loss="low",
        name="olog_to_triples"
    )
    
    graph.add_transformation(
        "olog", "text",
        forward=lambda o: " ".join(f"{s} {p} {obj}." for s, p, obj in o.to_triples()) if isinstance(o, Olog) else "",
        info_loss="medium",
        name="text_generator"
    )
    
    print(f"Graph: {graph}")
    print(f"Modalities: {graph.modalities}")
    
    # Find paths
    print(f"\nPath from text to triples: {graph.find_path('text', 'triples')}")
    print(f"Path from image to text: {graph.find_path('image', 'text')}")
    
    # Estimate path loss
    text_to_triples_loss = graph.estimate_path_info_loss("text", "triples")
    image_to_text_loss = graph.estimate_path_info_loss("image", "text")
    
    print(f"\nEstimated losses:")
    print(f"  text -> triples: {text_to_triples_loss:.1%}")
    print(f"  image -> text: {image_to_text_loss:.1%}")


# ==================== Main ====================

if __name__ == "__main__":
    # Run all examples
    example_manual_olog()
    example_info_loss_tracking()
    example_round_trip()
    example_llm_pipeline()
    example_modality_graph_integration()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key takeaways:

1. OLOGS provide categorical structure for knowledge representation
   - Entities are types (objects in a category)
   - Relationships are aspects (morphisms)
   - Composition gives transitive relationships

2. INFORMATION LOSS is tracked through the pipeline
   - Text -> Embedding: ~70% loss (semantic compression)
   - Image -> Embedding: ~85% loss (visual detail lost)
   - Embedding -> Olog: ~40% loss (structure extraction)
   - Olog -> Text: ~30% loss (generation is creative)

3. ROUND-TRIP SUCCESS depends on cumulative preservation
   - text -> olog -> text: ~13% success probability
   - image -> olog -> text: ~6% success probability
   - structured_data -> olog -> text: ~34% success probability

4. WARNINGS alert users when feeding lossy data to LLMs
   - Helps users understand what detail is lost
   - Suggests preserving original alongside embeddings
   - Recommends including structured summaries

5. INTEGRATION with ModalityGraph enables:
   - Automatic path finding through modality space
   - Cumulative loss estimation along paths
   - Consistency checking across modalities
""")
