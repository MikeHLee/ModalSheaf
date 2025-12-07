# Restriction and Extension Maps

> *"Restriction zooms in; extension zooms out. Neither is free."*

This chapter explores the fundamental operations for moving data between scales
and modalities.

---

## The Two Directions

When we have nested regions V ⊆ U, we can move data in two directions:

```
        U (larger region)
       ╱ ╲
      ╱   ╲
     ↓     ↑
restriction  extension
     ↓     ↑
      ╲   ╱
       ╲ ╱
        V (smaller region)
```

| Direction | Name | What it does | Information |
|-----------|------|--------------|-------------|
| U → V | Restriction | Extract local from global | May lose context |
| V → U | Extension | Aggregate local to global | May lose detail |

---

## Restriction Maps

A **restriction map** ρ_{U,V}: F(U) → F(V) extracts the data on V from data on U.

### Properties

1. **Always exists** (for sheaves)
2. **Preserves structure**: ρ is a homomorphism
3. **Composes**: ρ_{U,V} ∘ ρ_{V,W} = ρ_{U,W}

### Examples in ML

| From | To | Restriction | Info Loss |
|------|-----|-------------|-----------|
| Image | Crop | Extract region | Context |
| Document | Paragraph | Extract section | Document structure |
| Video | Frame | Extract time slice | Temporal context |
| 3D scene | 2D projection | Camera projection | Depth |
| Full model | Pruned model | Remove weights | Capacity |

### Code Example

```python
from modalsheaf.modalities import image_to_patches, text_to_sentences

# Image → Patches (restriction to local regions)
patches = image_to_patches(image, patch_size=16)

# Text → Sentences (restriction to local units)
sentences = text_to_sentences(document)
```

---

## Extension Maps

An **extension map** ε_{V,U}: F(V) → F(U) aggregates local data to global.

### Key Difference from Restriction

Extension is **not always possible** or **not unique**:
- Given a patch, you can't reconstruct the full image
- Given a sentence, you can't reconstruct the document

### When Extension Works

1. **Trivial extension**: Pad with zeros/defaults
2. **Learned extension**: Train a decoder
3. **Gluing**: Combine multiple local pieces (next chapter)

### Examples in ML

| From | To | Extension | Info Added |
|------|-----|-----------|------------|
| Embedding | Text | Decoder | Language model prior |
| Patches | Image | Reassemble | Spatial structure |
| Features | Prediction | Classifier head | Task knowledge |
| Samples | Distribution | Density estimation | Smoothness prior |

### Code Example

```python
from modalsheaf.modalities import patches_to_image, sentences_to_text

# Patches → Image (extension with known layout)
image = patches_to_image(patches, image_shape=(224, 224))

# Sentences → Text (extension by concatenation)
document = sentences_to_text(sentences)
```

---

## Information Loss

Every transformation has an **information profile**:

```
Source ──────────────────────────────────────→ Target
       ↑                                    ↑
       │                                    │
   Original                            Transformed
   Information                         Information
       │                                    │
       └──────────── Lost ─────────────────┘
```

### Types of Transformations

| Type | Definition | Invertible? | Example |
|------|------------|-------------|---------|
| **Isomorphism** | Bijective, both directions continuous | Yes, perfectly | PNG ↔ numpy |
| **Embedding** | Injective (1-to-1) | Yes, on image | Text → high-dim |
| **Projection** | Surjective (onto) | No | RGB → grayscale |
| **Lossy** | Neither | No | JPEG compression |

### Tracking in ModalSheaf

```python
from modalsheaf import Transformation, TransformationType

t = Transformation(
    source="image",
    target="embedding",
    forward=clip_encoder,
    transform_type=TransformationType.LOSSY,
    info_loss_estimate=0.8,  # 80% of pixel info lost
)
```

---

## The Adjunction Perspective

In category theory, restriction and extension are often **adjoint functors**:

```
Extension ⊣ Restriction
```

This means:
- Extension is the "best" way to go from local to global
- Restriction is the "best" way to go from global to local
- They're related by a universal property

### Intuition

If you restrict then extend, you get something "larger" than you started with
(you've lost the specific global context).

If you extend then restrict, you get back what you started with
(extension followed by restriction is identity on the local part).

---

## Sheaves vs Cosheaves

The direction of maps determines whether we have a sheaf or cosheaf:

| Structure | Maps | Direction | ML Example |
|-----------|------|-----------|------------|
| Sheaf | Restriction | Contravariant (U→V when V⊆U) | Encoders |
| Cosheaf | Extension | Covariant (V→U when V⊆U) | Decoders |

### When to Use Which

- **Sheaf**: When you're extracting/encoding/compressing
- **Cosheaf**: When you're generating/decoding/expanding

### Combined View

A full multimodal system has both:

```
Image ──encoder──→ Embedding ──decoder──→ Image'
  │                    │                    │
  └── Sheaf ──────────┴────── Cosheaf ─────┘
```

---

## Hierarchical Structures

Many data types have natural hierarchies:

### Text Hierarchy

```
Document
    ↓ restriction
Sections
    ↓ restriction
Paragraphs
    ↓ restriction
Sentences
    ↓ restriction
Words
    ↓ restriction
Characters
```

Each level is a restriction of the level above.

### Image Hierarchy

```
Image (H×W×C)
    ↓ restriction
Patches (N×P×P×C)
    ↓ restriction
Pixels (H×W×C individual)
```

### Code Hierarchy

```
Codebase
    ↓ restriction
Modules
    ↓ restriction
Files
    ↓ restriction
Functions
    ↓ restriction
Lines
    ↓ restriction
Tokens
```

---

## Composition of Maps

Maps compose in the expected way:

```python
# Composing restrictions
text_to_words = compose(text_to_sentences, sentences_to_words)

# Composing extensions
words_to_text = compose(words_to_sentences, sentences_to_text)

# Round-trip (not identity!)
round_trip = compose(text_to_words, words_to_text)
# "Hello, world!" → ["Hello,", "world!"] → "Hello, world!"
# But: "Hello,  world!" → ["Hello,", "world!"] → "Hello, world!"
# (Extra space lost!)
```

### Information Loss Accumulates

```python
# Each step loses some information
loss1 = text_to_sentences.info_loss  # 0.05
loss2 = sentences_to_words.info_loss  # 0.10

# Total loss (approximately)
total_loss = 1 - (1 - loss1) * (1 - loss2)  # ≈ 0.145
```

---

## Practical Guidelines

### When Designing Transformations

1. **Be explicit about direction**: Is this restriction or extension?
2. **Track information loss**: What's lost? What's preserved?
3. **Consider invertibility**: Can you go back? At what cost?
4. **Compose carefully**: Loss accumulates along paths

### Common Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| Encode-Decode | Restrict then extend | Autoencoder |
| Hierarchical | Multiple restriction levels | U-Net |
| Multi-view | Multiple restrictions from same source | CLIP |
| Fusion | Multiple extensions to same target | Sensor fusion |

---

## Summary

| Concept | Definition | ML Analog |
|---------|------------|-----------|
| Restriction | Global → Local | Encoder, crop, project |
| Extension | Local → Global | Decoder, upsample, generate |
| Isomorphism | Lossless both ways | Format conversion |
| Embedding | Lossless one way | Tokenization |
| Projection | Lossy, surjective | Dimensionality reduction |
| Adjunction | Optimal restriction-extension pair | Encoder-decoder |

---

## Next Steps

- [Gluing and Cohomology](04_gluing_and_cohomology.md): When extension requires multiple pieces
- [Applications to ML](05_applications_to_ml.md): Putting it all together

---

## References

- Robinson, M. (2014). *Topological Signal Processing*. Chapter 3.
- Curry, J. (2014). *Sheaves, Cosheaves and Applications*. Chapter 3.
- Fong & Spivak (2019). *An Invitation to Applied Category Theory*. Chapter 4.
