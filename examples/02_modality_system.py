#!/usr/bin/env python3
"""
Modality System Example for ModalSheaf.

This example demonstrates:
1. Built-in modality types
2. Loading/saving different file types
3. Detecting modalities automatically
4. Working with code, text, images, and structured data
5. The modality hierarchy
"""

import numpy as np
from pathlib import Path
import tempfile
import json

from modalsheaf import (
    # Modality constants
    TEXT, IMAGE, EMBEDDING, JSON_DATA, CODE_FILE,
    # Functions
    list_modalities,
    get_modality,
    detect_modality,
    load_file,
    save_file,
    get_handler,
)
from modalsheaf.modalities.base import ModalityCategory, get_modality_hierarchy


def main():
    print("=" * 70)
    print("ModalSheaf Modality System")
    print("=" * 70)
    
    # ==================== List Available Modalities ====================
    
    print("\n## Available Modalities\n")
    
    all_modalities = list_modalities()
    print(f"Total registered: {len(all_modalities)}")
    
    # Group by category
    for category in ModalityCategory:
        mods = list_modalities(category)
        if mods:
            print(f"\n### {category.name}")
            for mod_name in mods:
                spec = get_modality(mod_name)
                print(f"  - {mod_name}: {spec.modality.description}")
    
    # ==================== Modality Detection ====================
    
    print("\n" + "=" * 70)
    print("## Automatic Modality Detection")
    print("=" * 70)
    
    # Detect from data
    test_cases = [
        ("Hello, world!", "Simple string"),
        ("https://example.com", "URL string"),
        ('{"key": "value"}', "JSON string"),
        (np.random.randn(768), "1D numpy array"),
        (np.random.randn(224, 224, 3), "3D numpy array (image-like)"),
        ({"name": "test", "value": 42}, "Python dict"),
    ]
    
    print("\nDetecting modality from data:")
    for data, description in test_cases:
        detected = detect_modality(data=data)
        print(f"  {description:30} -> {detected}")
    
    # Detect from file extensions
    print("\nDetecting modality from file paths:")
    test_paths = [
        "document.txt",
        "image.png",
        "script.py",
        "config.json",
        "styles.css",
        "data.csv",
        "video.mp4",
    ]
    
    for path in test_paths:
        detected = detect_modality(path=path)
        print(f"  {path:20} -> {detected}")
    
    # ==================== Working with Handlers ====================
    
    print("\n" + "=" * 70)
    print("## Modality Handlers")
    print("=" * 70)
    
    # Text handler
    text_handler = get_handler("text")
    sample_text = "This is a sample text.\nIt has multiple lines.\nAnd some words."
    
    print("\nText Handler:")
    print(f"  Valid: {text_handler.validate(sample_text)}")
    metadata = text_handler.get_metadata(sample_text)
    print(f"  Metadata: {metadata}")
    
    # Embedding handler
    emb_handler = get_handler("embedding")
    sample_embedding = np.random.randn(768).astype(np.float32)
    
    print("\nEmbedding Handler:")
    print(f"  Valid: {emb_handler.validate(sample_embedding)}")
    metadata = emb_handler.get_metadata(sample_embedding)
    print(f"  Metadata: {metadata}")
    
    # Code handler
    code_handler = get_handler("code")
    sample_code = '''
def hello_world():
    """A simple function."""
    print("Hello, World!")
    return 42

if __name__ == "__main__":
    hello_world()
'''
    
    print("\nCode Handler:")
    print(f"  Valid: {code_handler.validate(sample_code)}")
    metadata = code_handler.get_metadata(sample_code, path="example.py")
    print(f"  Metadata: {metadata}")
    print(f"  Detected language for .py: {code_handler.detect_language('test.py')}")
    print(f"  Detected language for .ts: {code_handler.detect_language('test.ts')}")
    print(f"  Detected language for .rs: {code_handler.detect_language('test.rs')}")
    
    # ==================== File Operations ====================
    
    print("\n" + "=" * 70)
    print("## File Operations")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save and load text
        text_path = tmpdir / "sample.txt"
        save_file(sample_text, text_path)
        loaded_text = load_file(text_path)
        print(f"\nText file roundtrip: {'✓' if loaded_text == sample_text else '✗'}")
        
        # Save and load JSON
        json_data = {"name": "test", "values": [1, 2, 3], "nested": {"a": 1}}
        json_path = tmpdir / "data.json"
        save_file(json_data, json_path)
        loaded_json = load_file(json_path)
        print(f"JSON file roundtrip: {'✓' if loaded_json == json_data else '✗'}")
        
        # Save and load code
        code_path = tmpdir / "script.py"
        save_file(sample_code, code_path, modality="code")
        loaded_code = load_file(code_path)
        print(f"Code file roundtrip: {'✓' if loaded_code == sample_code else '✗'}")
        
        # Save and load embedding
        emb_path = tmpdir / "embedding.npy"
        save_file(sample_embedding, emb_path)
        loaded_emb = load_file(emb_path, modality="embedding")
        print(f"Embedding file roundtrip: {'✓' if np.allclose(loaded_emb, sample_embedding) else '✗'}")
    
    # ==================== Modality Hierarchy ====================
    
    print("\n" + "=" * 70)
    print("## Modality Hierarchy")
    print("=" * 70)
    
    hierarchy = get_modality_hierarchy()
    print("\nParent -> Children relationships:")
    for parent, children in hierarchy.items():
        print(f"  {parent} -> {children}")
    
    # ==================== Typical Transformations ====================
    
    print("\n" + "=" * 70)
    print("## Typical Transformation Paths")
    print("=" * 70)
    
    print("\nCommon transformation paths:")
    
    key_modalities = ["text", "image", "code", "json_data", "embedding"]
    for mod_name in key_modalities:
        spec = get_modality(mod_name)
        if spec:
            print(f"\n  {mod_name}:")
            if spec.typical_transforms_to:
                print(f"    -> can transform to: {spec.typical_transforms_to}")
            if spec.typical_transforms_from:
                print(f"    <- can transform from: {spec.typical_transforms_from}")
    
    # ==================== Supported File Extensions ====================
    
    print("\n" + "=" * 70)
    print("## Supported File Extensions")
    print("=" * 70)
    
    print("\nCode files:")
    code_spec = get_modality("code_file")
    if code_spec:
        # Group by language type
        extensions = code_spec.file_extensions
        print(f"  {len(extensions)} extensions supported:")
        print(f"  {', '.join(extensions[:10])}...")
    
    print("\nImage files:")
    image_spec = get_modality("image_file")
    if image_spec:
        print(f"  {', '.join(image_spec.file_extensions)}")
    
    print("\nAudio files:")
    audio_spec = get_modality("audio")
    if audio_spec:
        print(f"  {', '.join(audio_spec.file_extensions)}")
    
    print("\nVideo files:")
    video_spec = get_modality("video")
    if video_spec:
        print(f"  {', '.join(video_spec.file_extensions)}")
    
    # ==================== Summary ====================
    
    print("\n" + "=" * 70)
    print("## Summary")
    print("=" * 70)
    
    print("""
    ModalSheaf's modality system provides:
    
    ✓ Automatic detection of data types
    ✓ Unified load/save interface for all modalities
    ✓ Metadata extraction for any data type
    ✓ Extensible registry for custom modalities
    ✓ Hierarchical organization of modality types
    ✓ Transformation path suggestions
    
    Implemented Modalities (Tier 1-2):
    - text, tokens, embedding (primitives)
    - image (tensor)
    - json_data (structured)
    - code, ast (code)
    - text_file, image_file, code_file, json_file (files)
    - url, webpage (web)
    
    Planned Modalities (Tier 3-5):
    - document, codebase, dataset (composite)
    - audio, video, spectrogram (media)
    - api_endpoint, database, web_app (advanced)
    """)
    
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
