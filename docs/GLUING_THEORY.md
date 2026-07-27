# Gluing Theory in ModalSheaf

This document explains the mathematical concept of **gluing** and how it differs from restriction/extension maps.

## The Three Fundamental Operations

| Operation | Direction | What it does | Mathematical |
|-----------|-----------|--------------|--------------|
| **Restriction** | Global → Local | Extract a piece | ρ: F(U) → F(V) |
| **Extension** | Local → Global | Aggregate up | (not always possible) |
| **Gluing** | Many Locals → Global | Assemble pieces | Sheaf axiom |

### Key Insight

Restriction and extension are **point-to-point** transformations between modalities.

Gluing is a **many-to-one** assembly operation that requires:
1. Multiple local pieces
2. Overlap regions where pieces meet
3. Consistency on overlaps
4. An assembly algorithm

---

## The Gluing Axiom

The defining property of a sheaf is the **gluing axiom**:

> Given local sections {sᵢ} on a cover {Uᵢ} that agree on overlaps,
> there exists a **unique** global section s that restricts to each sᵢ.

```
     U₁        U₂        U₃
   ┌────┐    ┌────┐    ┌────┐
   │ s₁ │    │ s₂ │    │ s₃ │     Local sections
   └──┬─┘    └─┬──┘    └─┬──┘
      │   ∩    │   ∩     │
      └───┼────┴───┼─────┘
          │        │
          ▼        ▼
       s₁|∩ = s₂|∩ = s₃|∩         Agreement on overlaps
          │        │
          └────┬───┘
               │
               ▼
         ┌──────────┐
         │    s     │              Unique global section
         └──────────┘
```

---

## H¹ Measures the Obstruction

When gluing **fails**, it's because local data doesn't agree on overlaps. The first cohomology group H¹ measures this obstruction:

- **H¹ = 0**: Local data is consistent, gluing succeeds
- **H¹ ≠ 0**: There's a "twist" or inconsistency preventing gluing

### Čech Cohomology Perspective

Given a cover {Uᵢ} and local sections {sᵢ}:

1. **0-cochains**: C⁰ = {sᵢ} (the local sections)
2. **1-cochains**: C¹ = {sᵢⱼ} where sᵢⱼ = sᵢ|∩ - sⱼ|∩ (differences on overlaps)
3. **Cocycle condition**: sᵢⱼ + sⱼₖ + sₖᵢ = 0 on triple overlaps

H¹ = (1-cocycles) / (1-coboundaries) measures "twists" that can't be removed.

---

## Gluing Examples

### 1. Panorama Stitching

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Image 1   │     │   Image 2   │     │   Image 3   │
│             ├─────┤             ├─────┤             │
│             │ ∩₁₂ │             │ ∩₂₃ │             │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼ Glue
              ┌───────────────────────────┐
              │        Panorama           │
              └───────────────────────────┘
```

- **Local sections**: Individual photos
- **Overlaps**: Shared regions between adjacent photos
- **Consistency check**: Pixel values must match in overlaps
- **H¹ ≠ 0 when**: Parallax, exposure differences, moving objects

### 2. Sensor Fusion

```
     Camera Frame          Lidar Frame          Radar Frame
         (x,y,z)              (x',y',z')           (x'',y'',z'')
            │                    │                     │
            │    T₁₂             │    T₂₃              │
            └────────────────────┴─────────────────────┘
                                 │
                                 ▼ Glue
                          World Frame
                            (X,Y,Z)
```

- **Local sections**: Points in each sensor's coordinate frame
- **Overlaps**: Objects visible to multiple sensors
- **Consistency check**: Transformed points must align
- **H¹ ≠ 0 when**: Calibration errors, timing drift

### 3. Geographic Assembly

```
    ┌──────────┐
    │ Oregon   │
    ├──────────┤ ← Border must match
    │ Nevada   │ California │
    └──────────┴────────────┘
              │
              ▼ Glue
    ┌─────────────────────┐
    │   Western States    │
    └─────────────────────┘
```

- **Local sections**: State boundaries and data
- **Overlaps**: Shared borders
- **Consistency check**: Border definitions must agree
- **H¹ ≠ 0 when**: Border disputes!

### 4. Document Assembly

```
    Page 1          Page 2          Page 3
    "...ends"  →  "with this"  →  "sentence."
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼ Glue
              Complete Document
```

- **Local sections**: Individual pages
- **Overlaps**: Sentence continuity at page breaks
- **Consistency check**: No broken sentences
- **H¹ ≠ 0 when**: Missing pages, wrong order

### 5. Codebase Assembly

```
    utils.py ──exports──→ helper()
        │                    │
        │                    ▼
        │              models.py ──imports──→ helper()
        │                    │
        │                    ▼
        └──────────────→ main.py ──imports──→ Model, helper
                              │
                              ▼ Glue
                    Unified Module
```

- **Local sections**: Individual source files
- **Overlaps**: Import/export relationships
- **Consistency check**: All imports resolved
- **H¹ ≠ 0 when**: Missing dependencies, circular imports

---

## Comparison: Restriction vs Gluing

| Aspect | Restriction Map | Gluing |
|--------|-----------------|--------|
| Direction | 1 → 1 | Many → 1 |
| Input | Global section | Multiple local sections |
| Output | Local section | Global section |
| Always works? | Yes | Only if consistent |
| Obstruction | None | H¹ cohomology |
| Example | Crop image | Stitch panorama |
| Example | Extract chapter | Assemble book |
| Example | Get sensor reading | Fuse all sensors |

---

## Implementation in ModalSheaf

### GluingProtocol Interface

```python
class GluingProtocol(ABC):
    def extract_overlap(self, section1, section2, overlap):
        """Get data from overlap region."""
        pass
    
    def measure_consistency(self, data1, data2, overlap):
        """Measure disagreement (0 = perfect match)."""
        pass
    
    def glue(self, sections, overlaps):
        """Attempt to assemble global section."""
        pass
```

### Built-in Protocols

| Protocol | Use Case |
|----------|----------|
| `PanoramaGluing` | Image stitching |
| `CoordinateGluing` | Sensor frame fusion |
| `HierarchicalGluing` | Geographic/organizational |
| `DocumentGluing` | Page/section assembly |
| `CodebaseGluing` | Source file integration |

### Usage

```python
from modalsheaf import PanoramaGluing, glue_with_protocol

result = glue_with_protocol(
    PanoramaGluing(consistency_threshold=30.0),
    sections=[
        {"id": "left", "data": img1},
        {"id": "right", "data": img2},
    ],
    overlaps=[
        {"sections": ("left", "right"), "region": (100, 0)},
    ]
)

if result.success:
    panorama = result.global_section
else:
    print(f"Gluing failed! H¹ = {result.h1_obstruction}")
    for error in result.consistency_errors:
        print(f"  Mismatch at {error['overlap']}")
```

---

## When to Use Each

### Use Restriction Maps When:
- Extracting a subset from a whole
- Zooming in on local detail
- Converting between representations

### Use Extension Maps When:
- Aggregating local to global
- Summarizing or compressing
- Encoding to embeddings

### Use Gluing When:
- Assembling multiple pieces into a whole
- Pieces have overlapping regions
- Consistency must be verified
- You need to detect conflicts/errors

---

## Connection to Multimodal AI

In multimodal AI, gluing appears when:

1. **Multi-view 3D reconstruction**: Glue camera views into 3D model
2. **Video from frames**: Glue frames with temporal overlap
3. **Knowledge graph fusion**: Glue facts from multiple sources
4. **Ensemble models**: Glue predictions with confidence overlaps
5. **RAG systems**: Glue retrieved chunks into coherent answer

**H¹ detects**:
- Hallucinations (text doesn't match image)
- Contradictions in knowledge bases
- Calibration drift in sensor fusion
- Inconsistent predictions across modalities

---

## Further Reading

1. **Čech Cohomology**: The standard way to compute H¹ for covers
2. **Descent Theory**: Generalization of gluing to categories
3. **Sheaf Laplacian**: Diffusion toward consistent global sections
4. **Neural Sheaf Diffusion**: Learning to glue in neural networks
