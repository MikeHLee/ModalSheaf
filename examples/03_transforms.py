#!/usr/bin/env python3
"""
Transformation Functions Example for ModalSheaf.

This example demonstrates:
1. Restriction maps (zoom in, extract detail)
2. Extension maps (zoom out, aggregate)
3. Built-in transforms for text, images, code, and JSON
4. Information loss tracking
5. Building transformation pipelines
"""

import numpy as np
from modalsheaf import ModalityGraph, TransformationType
from modalsheaf.modalities import (
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
    image_normalize,
    image_denormalize,
    # Embedding transforms
    embedding_normalize,
    embedding_reduce_dim,
    embeddings_average,
    # Code transforms
    code_to_ast,
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
    list_builtin_transforms,
    get_builtin_transform,
    create_transformation_from_builtin,
)


def demo_text_transforms():
    """Demonstrate text transformation hierarchy."""
    print("\n" + "=" * 60)
    print("TEXT TRANSFORMATIONS")
    print("=" * 60)
    
    text = """
    Machine learning is transforming how we process data.
    Neural networks can learn complex patterns. Deep learning
    enables breakthroughs in vision and language understanding.
    """
    
    print(f"\nOriginal text ({len(text)} chars):")
    print(f"  '{text[:60]}...'")
    
    # Text → Sentences (restriction: zoom into structure)
    sentences = text_to_sentences(text)
    print(f"\n→ Sentences ({len(sentences)} sentences):")
    for i, s in enumerate(sentences[:3]):
        print(f"    [{i}] {s[:50]}...")
    
    # Sentences → Text (extension: aggregate back)
    reconstructed = sentences_to_text(sentences)
    print(f"\n← Reconstructed text: {len(reconstructed)} chars")
    
    # Text → Words (restriction: zoom into tokens)
    words = text_to_words(text)
    print(f"\n→ Words ({len(words)} words):")
    print(f"    {words[:10]}...")
    
    # Text → Tokens (restriction: numerical representation)
    tokens = text_to_tokens(text)
    print(f"\n→ Tokens ({len(tokens)} tokens):")
    print(f"    {tokens[:20]}...")
    
    # Tokens → Text (extension: decode)
    decoded = tokens_to_text(tokens)
    print(f"\n← Decoded: '{decoded[:50]}...'")
    
    print("\n  Hierarchy: text → sentences → words → chars → tokens")
    print("  Direction: restriction (→) zooms in, extension (←) aggregates")


def demo_image_transforms():
    """Demonstrate image transformation hierarchy."""
    print("\n" + "=" * 60)
    print("IMAGE TRANSFORMATIONS")
    print("=" * 60)
    
    # Create sample image
    image = np.random.rand(64, 64, 3).astype(np.float32)
    print(f"\nOriginal image: shape={image.shape}, range=[{image.min():.2f}, {image.max():.2f}]")
    
    # Image → Patches (restriction: zoom into local regions)
    patches = image_to_patches(image, patch_size=16)
    print(f"\n→ Patches: shape={patches.shape}")
    print(f"    {patches.shape[0]} patches of {patches.shape[1]}x{patches.shape[2]}")
    
    # Patches → Image (extension: reassemble)
    reconstructed = patches_to_image(patches, image_shape=(64, 64))
    print(f"\n← Reconstructed: shape={reconstructed.shape}")
    print(f"    Reconstruction error: {np.abs(image - reconstructed).max():.6f}")
    
    # Image → Grayscale (extension: reduce channels, LOSSY)
    gray = image_to_grayscale(image)
    print(f"\n→ Grayscale: shape={gray.shape}")
    print(f"    ⚠️ LOSSY: Color information lost!")
    
    # Grayscale → RGB (restriction: expand channels, no new info)
    rgb = grayscale_to_rgb(gray)
    print(f"\n← RGB from gray: shape={rgb.shape}")
    print(f"    Note: No color recovered, just 3 identical channels")
    
    # Image → Normalized (extension: standardize)
    normalized = image_normalize(image)
    print(f"\n→ Normalized: range=[{normalized.min():.2f}, {normalized.max():.2f}]")
    
    # Normalized → Image (restriction: denormalize)
    denormalized = image_denormalize(normalized)
    print(f"\n← Denormalized: range=[{denormalized.min()}, {denormalized.max()}]")
    
    print("\n  Hierarchy: image → patches → pixels")
    print("  Branches: image → grayscale (lossy), image → normalized (lossless)")


def demo_embedding_transforms():
    """Demonstrate embedding transformations."""
    print("\n" + "=" * 60)
    print("EMBEDDING TRANSFORMATIONS")
    print("=" * 60)
    
    # Create sample embeddings
    emb1 = np.random.randn(768).astype(np.float32)
    emb2 = np.random.randn(768).astype(np.float32)
    emb3 = np.random.randn(768).astype(np.float32)
    
    print(f"\nOriginal embedding: dim={len(emb1)}, norm={np.linalg.norm(emb1):.2f}")
    
    # Normalize (extension: project to unit sphere)
    unit_emb = embedding_normalize(emb1)
    print(f"\n→ Normalized: norm={np.linalg.norm(unit_emb):.6f}")
    print(f"    ⚠️ LOSSY: Magnitude information lost!")
    
    # Reduce dimension (extension: compress)
    reduced = embedding_reduce_dim(emb1, target_dim=128)
    print(f"\n→ Reduced: dim={len(reduced)}")
    print(f"    ⚠️ LOSSY: {768-128} dimensions lost!")
    
    # Average multiple embeddings (extension: aggregate)
    avg_emb = embeddings_average([emb1, emb2, emb3])
    print(f"\n→ Averaged 3 embeddings: dim={len(avg_emb)}")
    print(f"    ⚠️ LOSSY: Individual embedding info lost!")
    
    print("\n  All embedding transforms are extensions (aggregation/compression)")
    print("  There's no way to 'expand' an embedding without external info")


def demo_code_transforms():
    """Demonstrate code transformation hierarchy."""
    print("\n" + "=" * 60)
    print("CODE TRANSFORMATIONS")
    print("=" * 60)
    
    code = '''
import numpy as np
from typing import List

def calculate_mean(values: List[float]) -> float:
    """Calculate the arithmetic mean of a list of values."""
    # Check for empty list
    if not values:
        return 0.0
    return sum(values) / len(values)

def calculate_std(values: List[float]) -> float:
    """Calculate standard deviation."""
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return np.sqrt(variance)

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    print(f"Mean: {calculate_mean(data)}")
'''
    
    print(f"\nOriginal code: {len(code)} chars, {code.count(chr(10))} lines")
    
    # Code → AST (extension: parse structure)
    ast = code_to_ast(code, language="python")
    print(f"\n→ AST: root type = {ast.get('type', 'unknown')}")
    if 'body' in ast:
        print(f"    {len(ast['body'])} top-level nodes")
    
    # Code → Functions (restriction: extract specific elements)
    functions = code_to_functions(code, language="python")
    print(f"\n→ Functions ({len(functions)} found):")
    for func in functions:
        print(f"    - {func['name']}({', '.join(func['args'])})")
        if func.get('docstring'):
            print(f"      \"{func['docstring'][:40]}...\"")
    
    # Code → Imports (restriction: extract dependencies)
    imports = code_to_imports(code, language="python")
    print(f"\n→ Imports ({len(imports)} found):")
    for imp in imports:
        print(f"    - {imp}")
    
    # Code → Clean code (extension: remove comments)
    clean = code_remove_comments(code, language="python")
    print(f"\n→ Clean code (no comments): {len(clean)} chars")
    print(f"    Removed {len(code) - len(clean)} chars of comments")
    
    print("\n  Hierarchy: code → AST → functions/imports")
    print("  code → clean_code is lossy (comments lost)")


def demo_json_transforms():
    """Demonstrate JSON transformation hierarchy."""
    print("\n" + "=" * 60)
    print("JSON TRANSFORMATIONS")
    print("=" * 60)
    
    data = {
        "name": "ModalSheaf",
        "version": "0.1.0",
        "config": {
            "modalities": ["text", "image", "code"],
            "transforms": {
                "text": ["tokenize", "embed"],
                "image": ["resize", "normalize"]
            }
        },
        "metadata": {
            "author": "Michael",
            "created": "2024"
        }
    }
    
    print(f"\nOriginal JSON: {len(str(data))} chars")
    print(f"  Keys: {list(data.keys())}")
    
    # JSON → Text (restriction: serialize)
    text = json_to_text(data)
    print(f"\n→ Text: {len(text)} chars")
    print(f"    '{text[:50]}...'")
    
    # Text → JSON (extension: parse)
    parsed = text_to_json(text)
    print(f"\n← Parsed: {type(parsed).__name__}")
    print(f"    Roundtrip OK: {parsed == data}")
    
    # JSON → Flat (restriction: flatten hierarchy)
    flat = json_flatten(data)
    print(f"\n→ Flattened: {len(flat)} keys")
    for key in list(flat.keys())[:5]:
        print(f"    '{key}': {flat[key]}")
    print(f"    ...")
    
    # Flat → JSON (extension: unflatten)
    unflat = json_unflatten(flat)
    print(f"\n← Unflattened: {len(unflat)} top-level keys")
    
    # JSON → Schema (extension: extract structure, LOSSY)
    schema = json_get_schema(data)
    print(f"\n→ Schema:")
    print(f"    type: {schema.get('type')}")
    print(f"    properties: {list(schema.get('properties', {}).keys())}")
    print(f"    ⚠️ LOSSY: All values lost, only structure remains!")
    
    print("\n  Hierarchy: json → flat_json → key-value pairs")
    print("  json → schema is lossy (values lost)")


def demo_builtin_registry():
    """Show the built-in transform registry."""
    print("\n" + "=" * 60)
    print("BUILT-IN TRANSFORM REGISTRY")
    print("=" * 60)
    
    all_transforms = list_builtin_transforms()
    print(f"\n{len(all_transforms)} built-in transforms registered:")
    
    # Group by source modality
    by_source = {}
    for name in all_transforms:
        spec = get_builtin_transform(name)
        source = spec.source
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(spec)
    
    for source, specs in sorted(by_source.items()):
        print(f"\n  From '{source}':")
        for spec in specs:
            arrow = "↔" if spec.inverse_func else "→"
            loss = f"loss={spec.info_loss:.0%}" if spec.info_loss > 0 else "lossless"
            print(f"    {arrow} {spec.target}: {spec.description} ({loss})")
    
    # Create a Transformation object from registry
    print("\n\nCreating Transformation from registry:")
    t = create_transformation_from_builtin("text_to_sentences")
    print(f"  {t.name}: {t.source} → {t.target}")
    print(f"  Type: {t.transform_type.name}")
    print(f"  Invertible: {t.is_invertible}")


def demo_transformation_pipeline():
    """Build a transformation pipeline using the graph."""
    print("\n" + "=" * 60)
    print("TRANSFORMATION PIPELINE")
    print("=" * 60)
    
    # Create a graph with transforms
    graph = ModalityGraph(name="pipeline_demo")
    
    # Add modalities
    graph.add_modality("text")
    graph.add_modality("sentences")
    graph.add_modality("words")
    graph.add_modality("tokens")
    graph.add_modality("embedding", shape=(768,))
    
    # Add transforms
    graph.add_transformation(
        "text", "sentences",
        forward=text_to_sentences,
        inverse=sentences_to_text,
        info_loss="none"
    )
    
    graph.add_transformation(
        "text", "words",
        forward=text_to_words,
        inverse=words_to_text,
        info_loss="low"
    )
    
    graph.add_transformation(
        "text", "tokens",
        forward=text_to_tokens,
        inverse=tokens_to_text,
        info_loss="low"
    )
    
    # Mock embedding function
    def mock_embed(tokens):
        np.random.seed(sum(tokens[:10]) % 1000)
        return np.random.randn(768).astype(np.float32)
    
    graph.add_transformation(
        "tokens", "embedding",
        forward=mock_embed,
        info_loss="high"
    )
    
    print(f"\nGraph: {graph}")
    print(f"Modalities: {graph.modalities}")
    
    # Transform through pipeline
    text = "Hello world. This is a test."
    
    print(f"\nInput text: '{text}'")
    
    # Direct transforms
    sentences = graph.transform("text", "sentences", text)
    print(f"→ sentences: {sentences}")
    
    words = graph.transform("text", "words", text)
    print(f"→ words: {words}")
    
    tokens = graph.transform("text", "tokens", text)
    print(f"→ tokens: {tokens[:20]}...")
    
    # Multi-hop transform
    embedding = graph.transform("text", "embedding", text)
    print(f"→ embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.2f}")
    
    # Check path and info loss
    path = graph.find_path("text", "embedding")
    loss = graph.estimate_path_info_loss("text", "embedding")
    print(f"\nPath: {' → '.join(path)}")
    print(f"Total info loss: {loss:.1%}")


def main():
    print("=" * 60)
    print("ModalSheaf Transformation Functions")
    print("=" * 60)
    print("""
    Sheaf Theory Perspective:
    
    RESTRICTION MAPS (ρ): Zoom in, extract detail
    - text → sentences → words → chars
    - image → patches → pixels
    - code → AST → functions
    - json → flat_json → key-value
    
    EXTENSION MAPS (ε): Zoom out, aggregate
    - chars → words → sentences → text
    - pixels → patches → image
    - functions → AST → code
    - key-value → flat_json → json
    
    Information Flow:
    - Restriction: May lose global context
    - Extension: May lose local detail
    - Some transforms are isomorphisms (lossless both ways)
    - Most ML encoders are lossy extensions
    """)
    
    demo_text_transforms()
    demo_image_transforms()
    demo_embedding_transforms()
    demo_code_transforms()
    demo_json_transforms()
    demo_builtin_registry()
    demo_transformation_pipeline()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
