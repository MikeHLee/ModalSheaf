#!/usr/bin/env python3
"""
Basic usage example for ModalSheaf.

This example shows how to:
1. Create a modality graph
2. Add modalities and transformations
3. Transform data between modalities
4. Check consistency of multimodal data
"""

import numpy as np
from modalsheaf import (
    ModalityGraph,
    Modality,
    Transformation,
    TransformationType,
    ConsistencyChecker,
)


def main():
    # ==================== Setup ====================
    print("=" * 60)
    print("ModalSheaf Basic Usage Example")
    print("=" * 60)
    
    # Create a modality graph
    graph = ModalityGraph(name="example")
    
    # Add modalities
    graph.add_modality("image", shape=(64, 64, 3), dtype="float32",
                       description="RGB image")
    graph.add_modality("embedding", shape=(128,), dtype="float32",
                       description="Dense embedding vector")
    graph.add_modality("text", dtype="str",
                       description="Text description")
    
    print(f"\nCreated graph: {graph}")
    print(f"Modalities: {graph.modalities}")
    
    # ==================== Add Transformations ====================
    
    # Simple image encoder (mock)
    def encode_image(img):
        """Mock image encoder - just flatten and project."""
        flat = np.asarray(img).flatten()
        # Simple projection to 128 dims
        np.random.seed(42)  # Reproducible
        proj = np.random.randn(128, len(flat)) * 0.01
        return proj @ flat
    
    # Simple text encoder (mock)
    def encode_text(text):
        """Mock text encoder - hash-based embedding."""
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(128)
    
    # Add transformations
    graph.add_transformation(
        "image", "embedding",
        forward=encode_image,
        info_loss="high",  # Lossy compression
        name="image_encoder"
    )
    
    graph.add_transformation(
        "text", "embedding",
        forward=encode_text,
        info_loss="high",
        name="text_encoder"
    )
    
    print(f"\nAdded transformations:")
    print(f"  - image -> embedding (info_loss: high)")
    print(f"  - text -> embedding (info_loss: high)")
    
    # ==================== Transform Data ====================
    
    # Create sample data
    sample_image = np.random.rand(64, 64, 3)
    sample_text = "a random image"
    
    # Transform to embeddings
    image_embedding = graph.transform("image", "embedding", sample_image)
    text_embedding = graph.transform("text", "embedding", sample_text)
    
    print(f"\nTransformed data:")
    print(f"  Image shape: {sample_image.shape} -> Embedding shape: {image_embedding.shape}")
    print(f"  Text: '{sample_text}' -> Embedding shape: {text_embedding.shape}")
    
    # ==================== Check Consistency ====================
    
    print("\n" + "=" * 60)
    print("Consistency Analysis")
    print("=" * 60)
    
    # Create consistency checker
    checker = ConsistencyChecker(graph, common_modality="embedding")
    
    # Check consistency of image and text
    result = checker.check({
        "image": sample_image,
        "text": sample_text
    })
    
    print(f"\nResult: {result}")
    print(f"  H⁰ dimension: {result.h0_dim} (global consensus)")
    print(f"  H¹ dimension: {result.h1_dim} (inconsistencies)")
    print(f"  Consistency score: {result.consistency_score:.3f}")
    print(f"  Diagnosis: {result.diagnosis}")
    
    # Get detailed diagnosis
    diagnosis = checker.diagnose_inconsistency(result)
    print(f"\nDetailed diagnosis:")
    print(f"  Status: {diagnosis['status']}")
    if diagnosis.get('recommendations'):
        print(f"  Recommendations:")
        for rec in diagnosis['recommendations']:
            print(f"    - {rec}")
    
    # ==================== Path Analysis ====================
    
    print("\n" + "=" * 60)
    print("Path Analysis")
    print("=" * 60)
    
    # Find path between modalities
    path = graph.find_path("image", "embedding")
    print(f"\nPath from image to embedding: {' -> '.join(path)}")
    
    # Estimate information loss
    loss = graph.estimate_path_info_loss("image", "embedding")
    print(f"Estimated information loss: {loss:.1%}")
    
    # ==================== Transformation Types ====================
    
    print("\n" + "=" * 60)
    print("Understanding Transformation Types")
    print("=" * 60)
    
    print("""
    TransformationType explains what happens to information:
    
    ISOMORPHISM  - Fully reversible, no info loss (e.g., lossless format conversion)
    EMBEDDING    - One-way but info preserved (e.g., text -> high-dim embedding)
    PROJECTION   - Dimension reduction (e.g., PCA, pooling)
    LOSSY        - Information lost (e.g., JPEG compression, most neural encoders)
    
    In mathematical terms:
    - Isomorphism: f has inverse f⁻¹ such that f⁻¹(f(x)) = x
    - Embedding (monomorphism): f is injective, no two inputs map to same output
    - Projection (epimorphism): f is surjective, every output is reachable
    - Lossy: Neither injective nor surjective
    """)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
