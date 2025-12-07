# ModalSheaf

**Practical sheaf-theoretic tools for multimodal ML data transformations**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is this?

ModalSheaf provides a **practical, intuitive API** for moving data between ML modalities (text, images, audio, embeddings, etc.) while tracking:

- **Information loss** during transformations
- **Reversibility** of transformations (isomorphisms vs lossy maps)
- **Consistency** when fusing multiple data sources
- **Hierarchical structure** (pixels → patches → images → videos)

Built on sheaf theory, but you don't need to know any math to use it.

## Quick Example

```python
from modalsheaf import ModalityGraph, Modality, Transformation

# Define your modalities
graph = ModalityGraph()
graph.add_modality("image", shape=(224, 224, 3))
graph.add_modality("embedding", shape=(768,))
graph.add_modality("text", shape=None)  # variable length

# Register transformations (restriction maps)
graph.add_transformation(
    source="image",
    target="embedding",
    func=clip_image_encoder,
    inverse=None,  # Not invertible!
    info_loss="high"  # Lossy transformation
)

graph.add_transformation(
    source="text",
    target="embedding", 
    func=clip_text_encoder,
    inverse=None,
    info_loss="high"
)

# Check consistency between modalities
image_emb = graph.transform("image", "embedding", my_image)
text_emb = graph.transform("text", "embedding", my_caption)

consistency = graph.measure_consistency(
    {"image": my_image, "text": my_caption}
)
# Returns: {"score": 0.87, "H1": 0.13, "diagnosis": "minor inconsistency"}
```

## Installation

```bash
pip install modalsheaf
```

## Core Concepts (No Math Required!)

### 1. Modalities as Places

Think of each data type (image, text, audio) as a **place** where data can live.

### 2. Transformations as Roads

Transformations (encoders, decoders) are **roads** connecting places. Some roads are:
- **Two-way** (invertible/isomorphism): You can go back and forth without losing anything
- **One-way** (lossy): Information is lost, you can't fully recover the original

### 3. Consistency as Agreement

When you have data from multiple sources about the same thing, do they **agree**?
- Image shows a cat, caption says "a dog" → **Inconsistent**
- Image shows a cat, caption says "a cat" → **Consistent**

### 4. The H⁰, H¹ Numbers (Cohomology Made Simple)

See [INTUITIVE_COHOMOLOGY.md](docs/INTUITIVE_COHOMOLOGY.md) for a full explanation, but briefly:

- **H⁰ = "What everyone agrees on"** — The global consensus
- **H¹ = "Where disagreements hide"** — Inconsistencies that can't be resolved

If H¹ = 0, your data is perfectly consistent. If H¹ ≠ 0, there's a conflict somewhere.

## Features

### Modality Management
- Define custom modalities with shapes and dtypes
- Build modality graphs with transformations
- Automatic path finding between modalities

### Transformation Tracking
- Register forward and inverse transforms
- Track information loss (isomorphism, embedding, projection, lossy)
- Compose transformations automatically

### Consistency Analysis
- Measure consistency across modality graph
- Compute cohomology (H⁰, H¹) for data fusion
- Diagnose where inconsistencies occur

### Built-in Modalities
- Images (PIL, numpy, torch tensors)
- Text (strings, token IDs, embeddings)
- Audio (waveforms, spectrograms, embeddings)
- Video (frame sequences, temporal embeddings)
- Structured data (JSON, dataframes)

### ML Framework Integration
- PyTorch transforms
- HuggingFace encoders
- OpenAI/Anthropic embeddings
- Custom encoders

## Documentation

- [Intuitive Guide to Cohomology](docs/INTUITIVE_COHOMOLOGY.md)
- [API Reference](docs/API.md)
- [Examples](examples/)
- [Theory Background](docs/THEORY.md)

## Comparison with pysheaf

| Feature | pysheaf | modalsheaf |
|---------|---------|------------|
| **Focus** | General sheaf theory | ML modality transformations |
| **API** | Mathematical (cells, cofaces) | Practical (modalities, transforms) |
| **Target users** | Mathematicians | ML practitioners |
| **Built-in modalities** | None | Images, text, audio, video |
| **ML integration** | None | PyTorch, HuggingFace, etc. |

## License

MIT

## Citation

```bibtex
@software{modalsheaf,
  title = {ModalSheaf: Practical Sheaf-Theoretic Tools for Multimodal ML},
  author = {Lee, Michael Harrison},
  year = {2024},
  url = {https://github.com/MikeHLee/modalsheaf}
}
```
