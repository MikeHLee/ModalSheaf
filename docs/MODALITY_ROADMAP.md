# Modality Roadmap

This document outlines the supported and planned modalities for ModalSheaf.

## Modality Tiers

### Tier 1: Core Primitives ✅ Implemented

| Modality | Description | Python Types | Status |
|----------|-------------|--------------|--------|
| `text` | Raw text string | `str` | ✅ |
| `tokens` | Token IDs | `list[int]`, `np.ndarray` | ✅ |
| `embedding` | Dense vector | `np.ndarray` | ✅ |
| `json_data` | Structured JSON | `dict`, `list` | ✅ |

### Tier 2: Files ✅ Implemented

| Modality | Extensions | Description | Status |
|----------|------------|-------------|--------|
| `text_file` | .txt, .md, .rst | Text documents | ✅ |
| `image_file` | .png, .jpg, .webp | Image files | ✅ |
| `code_file` | .py, .js, .ts, ... | Source code | ✅ |
| `json_file` | .json, .jsonl | JSON files | ✅ |

### Tier 3: Tensors & Structured ✅ Implemented

| Modality | Shape | Description | Status |
|----------|-------|-------------|--------|
| `image` | (H, W, C) | Image array | ✅ |
| `code` | - | Code string with language | ✅ |
| `ast` | - | Abstract Syntax Tree | ✅ (stub) |
| `url` | - | URL string | ✅ |
| `webpage` | - | HTML + metadata | ✅ (stub) |

### Tier 4: Composite 🔄 In Progress

| Modality | Description | Status |
|----------|-------------|--------|
| `document` | Multi-page doc (PDF, DOCX) | 🔄 Stub |
| `codebase` | Directory of code | ✅ Basic |
| `dataset` | Collection of samples | 🔄 Stub |
| `notebook` | Jupyter notebook | 📋 Planned |

### Tier 5: Media 📋 Planned

| Modality | Extensions | Description | Status |
|----------|------------|-------------|--------|
| `audio` | .wav, .mp3, .flac | Audio waveform | 📋 Stub |
| `video` | .mp4, .avi, .webm | Video frames | 📋 Stub |
| `spectrogram` | - | Audio visualization | 📋 Stub |

### Tier 6: Advanced 🔮 Future

| Modality | Description | Status |
|----------|-------------|--------|
| `api_endpoint` | REST/GraphQL API | 🔮 Future |
| `database` | SQL/NoSQL connection | 🔮 Future |
| `web_app` | Full web application | 🔮 Future |
| `ml_model` | Trained model | 🔮 Future |

---

## Modality Hierarchy

```
                        ┌─────────────┐
                        │   WORLD     │
                        └──────┬──────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼────┐           ┌─────▼─────┐          ┌────▼────┐
   │  TEXT   │           │  VISUAL   │          │  AUDIO  │
   └────┬────┘           └─────┬─────┘          └────┬────┘
        │                      │                     │
   ┌────┼────┐            ┌────┼────┐          ┌────┼────┐
   │    │    │            │    │    │          │    │    │
tokens code json       image video webpage   waveform spectrogram
   │    │    │            │    │    │          │    │
   └────┼────┘            └────┼────┘          └────┼────┘
        │                      │                     │
        └──────────────────────┼─────────────────────┘
                               │
                        ┌──────▼──────┐
                        │  EMBEDDING  │
                        └─────────────┘
```

---

## Transformation Types Between Modalities

### Lossless (Isomorphism)
- `text` ↔ `text_file` (save/load)
- `json_data` ↔ `json_file` (serialize/deserialize)
- `image` ↔ `image_file` (PNG, lossless formats)

### Lossy but Invertible (Embedding)
- `text` → `tokens` → `text` (tokenize/detokenize, may lose whitespace)
- `code` → `ast` → `code` (parse/unparse, may lose formatting)

### Lossy (Projection)
- `image` → `embedding` (encode, can't reconstruct)
- `text` → `embedding` (encode, can't reconstruct)
- `video` → `image` (extract frame, lose temporal info)
- `audio` → `spectrogram` (STFT, lose phase)
- `codebase` → `embedding` (summarize, lose details)

### Information Flow

```
High Information                              Low Information
(Raw Data)                                    (Abstract)

image (H×W×3)  ──────────────────────────────►  embedding (768)
   ~150K dims                                      768 dims
   
text (N chars) ──► tokens (M) ──────────────►  embedding (768)
   variable         ~N/4                           768 dims
   
audio (T samples) ──► spectrogram ──────────►  embedding (768)
   ~16K/sec            (F×T')                      768 dims
```

---

## Adding Custom Modalities

```python
from modalsheaf.modalities import (
    ModalitySpec, 
    ModalityCategory,
    register_modality,
    ModalityHandler,
    register_handler,
)
from modalsheaf import Modality

# 1. Define the modality specification
MY_MODALITY = register_modality(ModalitySpec(
    modality=Modality(
        name="my_custom_type",
        shape=None,
        dtype="object",
        description="My custom data type"
    ),
    category=ModalityCategory.STRUCTURED,
    file_extensions=[".myext"],
    mime_types=["application/x-mytype"],
    python_types=[MyClass],
    typical_transforms_to=["embedding", "text"],
    typical_transforms_from=["json_data"],
))

# 2. Create a handler
class MyHandler(ModalityHandler):
    modality_name = "my_custom_type"
    
    def validate(self, data):
        return isinstance(data, MyClass)
    
    def load(self, source):
        # Load from file
        ...
    
    def save(self, data, destination):
        # Save to file
        ...
    
    def get_metadata(self, data):
        return {
            "modality": self.modality_name,
            # ... custom metadata
        }

register_handler(MyHandler())
```

---

## Integration Points

### With ML Frameworks

```python
# PyTorch integration
from modalsheaf.transforms import create_torch_transform

img_encoder = create_torch_transform(
    source="image",
    target="embedding",
    model=clip_model.visual,
    device="cuda"
)

# HuggingFace integration
from modalsheaf.transforms import create_huggingface_transform

text_encoder = create_huggingface_transform(
    source="text",
    target="embedding",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### With File Systems

```python
from modalsheaf import load_file, save_file, detect_modality

# Automatic detection and loading
data = load_file("path/to/file.any")
modality = detect_modality(data=data)

# Batch operations
from modalsheaf.modalities.loaders import load_directory
all_code = load_directory("./src", pattern="*.py")
```

### With Web Resources

```python
from modalsheaf.modalities.loaders import load_url

# Fetch and parse
webpage = load_url("https://example.com")
api_response = load_url("https://api.example.com/data", target_modality="json_data")
```

---

## Future Directions

### Short-term (v0.2)
- [ ] Complete audio/video handlers
- [ ] PDF document extraction
- [ ] Jupyter notebook support
- [ ] Better codebase analysis (dependency graphs)

### Medium-term (v0.3)
- [ ] Database connections (SQLAlchemy)
- [ ] API endpoint modality
- [ ] Streaming data support
- [ ] Compression/quantization tracking

### Long-term (v1.0)
- [ ] Full web app analysis
- [ ] ML model as modality
- [ ] Distributed data sources
- [ ] Real-time transformation pipelines
