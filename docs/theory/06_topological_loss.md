# Topological Characterization of Information Loss

> *"Information loss isn't just a number — it has shape, location, and structure."*

When we transform data between modalities, we lose information. But saying "70% lost" 
misses the rich structure of *what* was lost, *where* it was lost, and *how* the 
remaining information is shaped.

This chapter develops intuition for characterizing information loss topologically.

---

## The Problem with Scalar Loss

Consider encoding an image to an embedding:

```
1024×1024×3 image → 768-dimensional embedding
```

A naive view: "We went from 3M numbers to 768. That's 99.97% compression."

But this misses everything important:
- Did we lose **spatial** detail (where things are)?
- Did we lose **semantic** detail (what things mean)?
- Did we lose **relational** detail (how things connect)?
- Is the loss **uniform** or concentrated in certain regions?
- What **structure** does the preserved information have?

Topology gives us tools to answer these questions.

---

## Part 1: What KIND of Information Was Lost?

### The Loss Type Taxonomy

Different transformations lose different *kinds* of information:

| Loss Type | What's Lost | Example |
|-----------|-------------|---------|
| **Spatial** | Position, location, arrangement | Image → embedding loses pixel positions |
| **Temporal** | Sequence, timing, dynamics | Audio → embedding loses time structure |
| **Semantic** | Meaning, concepts, categories | Text → tokens loses word boundaries |
| **Relational** | Connections, dependencies | Graph → embedding loses edge structure |
| **Structural** | Topology, shape, connectivity | 3D mesh → point cloud loses faces |
| **Statistical** | Distribution, variance, moments | Quantization loses precision |

### Visualizing Loss Types

**Spatial Loss** (Image → Embedding):
```
Original:                    After:
┌─────────────┐              
│ 🐱    🌳    │              → [0.2, -0.5, 0.8, ...]
│      🏠    │              
│  🚗       │              768 numbers, no positions
└─────────────┘              
```
The embedding knows "cat, tree, house, car" but not where they are.

**Relational Loss** (Knowledge Graph → Embedding):
```
Original:                    After:
Einstein ──wrote──→ Relativity
    │                        → [0.3, 0.1, -0.7, ...]
    └──born_in──→ Ulm       
                             The relationships are implicit, not explicit
```

**Temporal Loss** (Audio → Spectrogram → Embedding):
```
Original: "Hello" spoken over 0.5 seconds
         ┌──────────────────┐
         │ H  e  l  l  o    │ (time →)
         └──────────────────┘
         
After:   [0.4, -0.2, 0.9, ...] — no time axis
```

### Why This Matters

If you're building a system that needs spatial reasoning, you need to know that your
embedding step loses spatial information. The loss *type* tells you what downstream
tasks will struggle.

---

## Part 2: WHERE Was Information Lost?

### Affected Dimensions

When we project from high to low dimensions, not all input dimensions are treated equally.

**Example: PCA on Images**

```python
# 1000 images, each 64×64 = 4096 dimensions
# PCA to 100 dimensions

# The first few principal components capture:
# - Overall brightness (PC1)
# - Left-right gradient (PC2)  
# - Top-bottom gradient (PC3)
# ...

# The last components (discarded) captured:
# - High-frequency texture
# - Fine edge details
# - Noise
```

The loss is *localized* in the high-frequency components.

### Affected Indices

For structured data, we can track exactly which elements were affected:

```python
# Original text: "The quick brown fox jumps over the lazy dog"
# After tokenization: [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290]

# After embedding, we lose:
# - Word boundaries (indices where words start/end)
# - Character-level detail within tokens
# - Punctuation nuances
```

### The Kernel and Cokernel

In linear algebra terms:
- **Kernel**: What gets mapped to zero (completely lost)
- **Cokernel**: What can't be reached (information that can't be represented)

For a transformation T: A → B:
- ker(T) = {a ∈ A : T(a) = 0} — inputs that produce no output
- coker(T) = B / im(T) — outputs that no input produces

**Example**: Grayscale conversion
```
RGB → Grayscale

Kernel: Color differences (red-green, blue-yellow)
        [1, 0, 0] - [0, 1, 0] = [1, -1, 0] → 0 (same gray)
        
Cokernel: Nothing (every grayscale value is reachable)
```

---

## Part 3: The SHAPE of What Remains (Betti Numbers)

### Counting Holes at Different Dimensions

Betti numbers are topological invariants that count "holes":

| Betti Number | What It Counts | Intuition |
|--------------|----------------|-----------|
| **b₀** | Connected components | "How many separate pieces?" |
| **b₁** | 1-dimensional holes | "How many loops/tunnels?" |
| **b₂** | 2-dimensional holes | "How many enclosed voids?" |

### Example: Point Cloud Topology

Consider a point cloud sampled from different shapes:

**Solid disk** (b₀=1, b₁=0):
```
    • • • • •
  • • • • • • •
  • • • • • • •
  • • • • • • •
    • • • • •
```
One connected piece, no holes.

**Circle/ring** (b₀=1, b₁=1):
```
    • • • • •
  •           •
  •           •
  •           •
    • • • • •
```
One connected piece, one hole in the middle.

**Two separate circles** (b₀=2, b₁=2):
```
  • • •     • • •
•       • •       •
•       • •       •
  • • •     • • •
```
Two connected pieces, each with a hole.

### How Transformations Change Topology

**Embedding typically reduces Betti numbers:**

```
Original data:                 After embedding:
b₀ = 1000 (many clusters)  →   b₀ = 1 (single manifold)
b₁ = 50 (loops in data)    →   b₁ = 0 (loops collapsed)
b₂ = 5 (voids)             →   b₂ = 0 (voids filled)
```

The embedding "smooths out" the topological structure.

**Entity extraction partially recovers structure:**

```
Embedding:                     After extraction:
b₀ = 1 (single blob)       →   b₀ = 15 (15 entities)
b₁ = 0 (no loops)          →   b₁ = 3 (3 relationship cycles)
```

Extracting entities creates discrete components; relationships create cycles.

### Persistent Homology: Tracking Features Across Scales

Not all topological features are equally important. **Persistent homology** tracks
which features survive as we vary a scale parameter.

```
Scale:  0.1    0.5    1.0    2.0    5.0
        
b₀:     100    50     20     5      1
        ↓      ↓      ↓      ↓      ↓
        noise  small  medium large  all
               clusters      clusters connected
```

Features that persist across many scales are "real"; features that appear and
disappear quickly are noise.

**Persistence diagram:**
```
Death
  │    
5 │         ×           (long-lived feature)
  │    
2 │    × ×              (medium-lived)
  │  × × × ×            (short-lived = noise)
1 │× × × × × ×
  └──────────────→ Birth
     0  1  2  3  4
```

Points far from the diagonal are significant features.

---

## Part 4: The Sheaf Perspective on Loss

### Restriction Maps and Their Kernels

In sheaf terms, a transformation is a **restriction map**:

```
ρ: F(U) → F(V)    where V ⊆ U
```

The kernel of ρ is the information that's "invisible" from V:

```
ker(ρ) = {s ∈ F(U) : ρ(s) = 0}
```

**Example**: Image → Embedding

```
F(image) = ℝ^(1024×1024×3)    (all possible images)
F(embedding) = ℝ^768           (all possible embeddings)

ρ = CLIP encoder

ker(ρ) ≈ {images that map to the zero vector}
       ≈ {adversarial noise patterns}
```

More usefully, we care about the **approximate kernel**: images that map to
*similar* embeddings.

### Cohomology of the Transformation

The cohomology groups measure the "failure" of the transformation:

- **H⁰(ρ)**: Global sections that survive the transformation
- **H¹(ρ)**: Obstructions — information that can't be recovered

When H¹ ≠ 0, there's information loss that creates inconsistency.

### The Exact Sequence

For a transformation T: A → B, we have:

```
0 → ker(T) → A → B → coker(T) → 0
```

This tells us:
- ker(T): What's completely lost
- im(T): What's preserved (but possibly transformed)
- coker(T): What can't be represented

**For embeddings:**
```
0 → [high-freq details] → Image → Embedding → [unreachable embeddings] → 0
         ker                         im              coker
```

---

## Part 5: Measuring Loss in Practice

### The TopologicalLossCharacterization Structure

In ModalSheaf, we capture loss with:

```python
@dataclass
class TopologicalLossCharacterization:
    # Scalar summary
    total_loss: float  # 0.0 to 1.0
    
    # Breakdown by type
    loss_regions: List[LossRegion]  # Each has type, magnitude, location
    
    # What's preserved
    preserved_dimensions: int       # Effective dimensionality
    preserved_betti: Tuple[int, ...]  # Topological invariants
    
    # Distribution of loss
    loss_entropy: float  # High = uniform, Low = concentrated
    
    # Confidence
    is_measured: bool    # Computed from data, or estimated?
    confidence: float    # How sure are we?
```

### Computing Betti Numbers

For point cloud data, use persistent homology:

```python
import numpy as np
from ripser import ripser  # TDA library

def compute_betti(points, max_dim=2):
    """Compute Betti numbers of a point cloud."""
    result = ripser(points, maxdim=max_dim)
    
    betti = []
    for dim in range(max_dim + 1):
        # Count features that persist significantly
        dgm = result['dgms'][dim]
        persistence = dgm[:, 1] - dgm[:, 0]
        significant = np.sum(persistence > 0.1)  # Threshold
        betti.append(significant)
    
    return tuple(betti)
```

### Comparing Before and After

```python
def measure_topological_loss(original, transformed, transform_fn):
    """Measure how topology changes through a transformation."""
    
    # Compute Betti numbers before
    betti_before = compute_betti(original)
    
    # Apply transformation
    result = transform_fn(original)
    
    # Compute Betti numbers after
    betti_after = compute_betti(transformed)
    
    # Measure change
    loss_regions = []
    
    # b₀ change (connected components)
    if betti_before[0] != betti_after[0]:
        loss_regions.append(LossRegion(
            loss_type=LossType.STRUCTURAL,
            magnitude=abs(betti_before[0] - betti_after[0]) / max(betti_before[0], 1),
            description=f"Components: {betti_before[0]} → {betti_after[0]}"
        ))
    
    # b₁ change (loops)
    if betti_before[1] != betti_after[1]:
        loss_regions.append(LossRegion(
            loss_type=LossType.RELATIONAL,
            magnitude=abs(betti_before[1] - betti_after[1]) / max(betti_before[1], 1),
            description=f"Loops: {betti_before[1]} → {betti_after[1]}"
        ))
    
    return TopologicalLossCharacterization(
        total_loss=compute_total_loss(betti_before, betti_after),
        loss_regions=loss_regions,
        preserved_betti=betti_after,
        is_measured=True
    )
```

---

## Part 6: Intuitive Examples

### Example 1: Text → Embedding → Entities → Text

Let's trace topology through a full pipeline:

**Original text:**
```
"Einstein was born in Ulm. He developed relativity. 
 Relativity changed physics. Physics explains the universe."
```

**Topology of original:**
- b₀ = 1 (one connected document)
- b₁ = 1 (circular reference: Einstein → relativity → physics → universe → ?)

**After embedding (768-dim vector):**
- b₀ = 1 (single point in embedding space)
- b₁ = 0 (no loops — it's just a point)

**Loss:** The circular narrative structure is destroyed.

**After entity extraction:**
```
Einstein ──born_in──→ Ulm
    │
    └──developed──→ Relativity ──changed──→ Physics ──explains──→ Universe
```
- b₀ = 5 (five entities)
- b₁ = 0 (no cycles in this graph)

**After text generation:**
```
"Einstein was born in Ulm and developed relativity, 
 which changed physics that explains the universe."
```
- b₀ = 1 (one document again)
- b₁ = 0 (linear narrative, no cycles)

**What was lost:** The original had an implicit cycle (universe connects back to
Einstein as a physicist). The reconstruction is linear.

### Example 2: Image → Patches → Embedding

**Original image (224×224×3):**
- b₀ = 1 (one connected image)
- b₁ = 3 (three "holes" — e.g., spaces between objects)
- Spatial structure: 224×224 grid

**After patching (14×14 patches of 16×16):**
- b₀ = 196 (196 separate patches)
- b₁ = 0 (patches don't form loops)
- Spatial structure: 14×14 grid (coarser)

**Loss:** Local connectivity within patches preserved, but global holes destroyed.

**After embedding (768-dim per patch, then averaged):**
- b₀ = 1 (single embedding)
- b₁ = 0 (no structure)
- Spatial structure: None

**Loss:** All spatial and topological structure gone.

### Example 3: Knowledge Graph → Embedding → Knowledge Graph

**Original graph:**
```
    A ←──→ B
    ↑      ↓
    D ←──→ C
```
- b₀ = 1 (connected)
- b₁ = 1 (one cycle: A→B→C→D→A)

**After graph embedding (e.g., node2vec):**
- Each node → 64-dim vector
- Graph structure → implicit in distances

**After reconstruction (from embeddings):**
```
    A ───→ B
    ↑      ↓
    D ←─── C
```
- b₀ = 1 (still connected)
- b₁ = 1 (cycle preserved!)

**In this case:** The topology was preserved because the embedding captured
the cyclic structure. But edge directions might be lost.

---

## Part 7: Practical Guidelines

### When to Care About Topological Loss

| Task | Critical Loss Types | Why |
|------|---------------------|-----|
| Object detection | Spatial | Need to know *where* things are |
| Sentiment analysis | Semantic | Need meaning, not structure |
| Relationship extraction | Relational | Need connections between entities |
| Time series forecasting | Temporal | Need sequence structure |
| Graph neural networks | Structural | Need topology of connections |

### Choosing Transforms to Minimize Critical Loss

If your task needs spatial information:
- Avoid global pooling (destroys position)
- Use position-preserving architectures (CNNs, ViTs with position embeddings)
- Track spatial loss explicitly

If your task needs relational information:
- Avoid bag-of-words (destroys relationships)
- Use graph-aware embeddings
- Preserve edge structure

### Warning Users About Loss

```python
def warn_if_critical_loss(loss: TopologicalLossCharacterization, task: str):
    """Warn if the loss type is critical for the task."""
    
    critical_types = {
        "object_detection": [LossType.SPATIAL],
        "sentiment": [LossType.SEMANTIC],
        "relation_extraction": [LossType.RELATIONAL],
        "forecasting": [LossType.TEMPORAL],
    }
    
    critical = critical_types.get(task, [])
    
    for region in loss.loss_regions:
        if region.loss_type in critical and region.magnitude > 0.3:
            warnings.warn(
                f"⚠️ High {region.loss_type.name} loss ({region.magnitude:.0%}) "
                f"may impact {task} performance. {region.description}"
            )
```

---

## Summary

### Key Concepts

1. **Loss has TYPE**: Spatial, temporal, semantic, relational, structural
2. **Loss has LOCATION**: Which dimensions, indices, or regions are affected
3. **Loss has SHAPE**: Betti numbers capture topological structure
4. **Loss can be MEASURED**: Not just estimated, but computed from actual data

### The Topological View

| Before | After | What Changed |
|--------|-------|--------------|
| b₀ = many | b₀ = 1 | Clusters merged |
| b₁ = some | b₁ = 0 | Loops collapsed |
| b₂ = few | b₂ = 0 | Voids filled |

### In ModalSheaf

```python
result = transform(data)

# Not just this:
print(f"Loss: {result.loss.total_loss:.0%}")

# But also this:
print(f"Dominant loss: {result.loss.dominant_loss_type().name}")
print(f"Preserved topology: b₀={result.loss.preserved_betti[0]}")
for region in result.loss.loss_regions:
    print(f"  {region.loss_type.name}: {region.magnitude:.0%}")
    print(f"    {region.description}")
```

---

## Further Reading

### Intuitive
1. **Ghrist, "Elementary Applied Topology"** — Free PDF, beautifully illustrated
2. **Carlsson, "Topology and Data"** — The foundational TDA paper, very readable

### Applied
3. **Robinson, "Topological Signal Processing"** — Sheaves for signal processing
4. **Chazal & Michel, "Introduction to TDA"** — Modern, practical

### Deep
5. **Curry, "Sheaves, Cosheaves and Applications"** — Localization and loss
6. **Edelsbrunner & Harer, "Computational Topology"** — Algorithms for Betti numbers

### Software
7. **Ripser** — Fast persistent homology: `pip install ripser`
8. **GUDHI** — Comprehensive TDA library: `pip install gudhi`
9. **giotto-tda** — TDA for machine learning: `pip install giotto-tda`

---

## Exercises

1. **Compute Betti numbers** for a point cloud sampled from a torus. Verify b₀=1, b₁=2, b₂=1.

2. **Track topology** through an image → CLIP → image reconstruction pipeline. What's lost?

3. **Design a transform** that preserves b₁ (loops) while reducing dimensionality.

4. **Measure loss types** for your favorite embedding model. Is it mostly spatial, semantic, or relational?
