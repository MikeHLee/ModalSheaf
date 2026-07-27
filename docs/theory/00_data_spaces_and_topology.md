# Data Spaces and Topology

> *"All data lives somewhere. Topology tells us what 'somewhere' means."*

Before diving into sheaves and cohomology, we need to build intuition for what it means for data to have **shape** and **structure**. This chapter starts from familiar ground—distance metrics—and shows why topology is the right language for formalizing and extending these ideas.

---

## Part I: Data Lives in Spaces

Every piece of data occupies a position in some space. The question is: what kind of space?

### The Intuition

When you have two data points, you instinctively ask: *how similar are they?* This question implies a **space** with some notion of distance or proximity.

| Data Type | Natural Space | What "Close" Means |
|-----------|---------------|-------------------|
| Embeddings | ℝⁿ | Small vector distance |
| Images | ℝ^(H×W×C) | Pixel-wise similarity |
| Text | Sequence space | Few edits apart |
| Graphs | Graph space | Similar structure |
| Rankings | Permutation space | Similar orderings |

The choice of space—and how we measure distance within it—shapes everything downstream: clustering, retrieval, classification, generation.

---

## Part II: Distance Metrics

A **metric** d(x, y) formalizes "how far apart" two points are. Every metric must satisfy:

1. **Non-negativity**: d(x, y) ≥ 0, with d(x, y) = 0 iff x = y
2. **Symmetry**: d(x, y) = d(y, x)
3. **Triangle inequality**: d(x, z) ≤ d(x, y) + d(y, z)

Let's explore the metrics you'll encounter in practice.

---

### Euclidean Distance (L2)

The straight-line distance through space.

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

**Example** (2D):
```
x = (1, 2)
y = (4, 6)

d(x, y) = √[(4-1)² + (6-2)²]
        = √[9 + 16]
        = √25 = 5
```

**When to use**: Dense embeddings, continuous features, when magnitude matters.

**Intuition**: Imagine stretching a string between two points. Euclidean distance is the length of that string.

---

### Manhattan Distance (L1)

Distance measured along axes only—like navigating a city grid.

$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

**Example** (2D):
```
x = (1, 2)
y = (4, 6)

d(x, y) = |4-1| + |6-2|
        = 3 + 4 = 7
```

**When to use**: Sparse data, when features are independent, robust to outliers.

**Intuition**: You can only move horizontally or vertically. The distance is how many blocks you walk.

---

### Cosine Distance

Measures the angle between vectors, ignoring magnitude.

$$d(x, y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$$

**Example**:
```
x = (1, 0)
y = (1, 1)

cos(θ) = (1·1 + 0·1) / (1 · √2) = 1/√2 ≈ 0.707
d(x, y) = 1 - 0.707 ≈ 0.293
```

**When to use**: Text embeddings, when direction matters more than magnitude, normalized representations.

**Intuition**: Two vectors pointing the same direction have distance 0, regardless of length. Perpendicular vectors have distance 1.

---

### Levenshtein (Edit) Distance

The minimum number of single-character edits (insertions, deletions, substitutions) to transform one string into another.

**Example**:
```
x = "kitten"
y = "sitting"

kitten → sitten  (substitute k→s)
sitten → sittin  (substitute e→i)
sittin → sitting (insert g)

d(x, y) = 3
```

**When to use**: String matching, DNA sequences, code similarity, typo correction.

**Intuition**: How many keystrokes to fix a typo?

---

### Spearman Distance

Measures how different two rankings are, based on rank correlation.

$$d(x, y) = 1 - \rho_s = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}$$

where $d_i$ is the difference in ranks for item $i$.

**Example**:
```
Ranking x: [A, B, C, D] → ranks [1, 2, 3, 4]
Ranking y: [B, A, D, C] → ranks [2, 1, 4, 3]

d_i = [1-2, 2-1, 3-4, 4-3] = [-1, 1, -1, 1]
d_i² = [1, 1, 1, 1]
Σd_i² = 4

ρ_s = 1 - 6(4)/(4·15) = 1 - 24/60 = 0.6
d(x, y) = 1 - 0.6 = 0.4
```

**When to use**: Comparing ranked lists, recommendation systems, search result evaluation.

**Intuition**: Do two judges agree on who's best, second-best, etc.?

---

### Higher-Dimensional Examples

As dimensions grow, distance behaves counterintuitively.

#### The Curse of Dimensionality

In high dimensions, all points become roughly equidistant:

```
Dimension    Ratio of max/min distance (random points)
    2              ~3.0
   10              ~1.5
  100              ~1.1
 1000              ~1.01
```

**Implication**: In 1000D, your "nearest neighbor" is barely closer than your "farthest neighbor." Raw distance becomes meaningless.

#### Example: 100D Embeddings

```python
import numpy as np

# Two "similar" embeddings (correlation 0.9)
x = np.random.randn(100)
y = 0.9 * x + 0.1 * np.random.randn(100)

# Two "random" embeddings
z = np.random.randn(100)

# Euclidean distances
d_xy = np.linalg.norm(x - y)  # ≈ 4.5
d_xz = np.linalg.norm(x - z)  # ≈ 14.0

# Cosine distances
cos_xy = 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))  # ≈ 0.01
cos_xz = 1 - np.dot(x, z) / (np.linalg.norm(x) * np.linalg.norm(z))  # ≈ 0.5
```

**Observation**: Cosine distance discriminates better in high dimensions because it ignores the "baseline" distance that all points share.

---

### Wasserstein (Earth Mover's) Distance

Measures the minimum "work" to transform one distribution into another.

$$W(P, Q) = \inf_{\gamma} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$

**Intuition**: If P is a pile of dirt and Q is a hole, Wasserstein distance is the minimum effort to fill the hole with the dirt.

**When to use**: Comparing distributions, generative models (WGAN), optimal transport.

---

## Part III: The Limits of Distance

Distance metrics are powerful, but they have fundamental limitations.

### Problem 1: Not All Similarities Are Metric

Some natural notions of "closeness" violate metric axioms:

| Similarity | Violation | Example |
|------------|-----------|---------|
| Semantic similarity | Triangle inequality | "bank" close to "river" and "money", but "river" far from "money" |
| Perceptual similarity | Symmetry | A child looks like their parent, but less so in reverse |
| Contextual similarity | Identity | Same word, different meanings |

### Problem 2: Local vs Global Structure

Metrics capture local structure (nearby points) but miss global structure:

```
Consider two circles:

    * * *           * * *
  *       *       *       *
 *         *     *         *
 *         *     *         *
  *       *       *       *
    * * *           * * *
    
    Circle A        Circle B
```

Locally, both look like line segments. But globally, they're the same shape (both have one hole). Metrics don't see this.

### Problem 3: Metrics Depend on Embedding

The same abstract data can have different distances depending on how it's embedded:

```
"cat" and "dog" in:
- One-hot encoding: d = √2 (orthogonal)
- Word2Vec: d ≈ 0.2 (similar)
- Character encoding: d = 2 (two edits)
```

Which is "right"? It depends on what structure you care about.

---

## Part IV: Why Topology?

Topology addresses these limitations by focusing on **structure** rather than **distance**.

### The Key Insight

Topology asks: *what properties are preserved under continuous deformation?*

- Stretch, bend, compress: OK
- Tear, glue, puncture: NOT OK

This captures **intrinsic** structure that doesn't depend on embedding.

### What Topology Sees

| Topological Property | What It Captures | ML Relevance |
|---------------------|------------------|--------------|
| Connectedness | Can you walk between points? | Cluster structure |
| Holes (H₁) | Loops that can't be contracted | Manifold structure |
| Voids (H₂) | Enclosed cavities | Higher-order structure |
| Dimension | Intrinsic degrees of freedom | Data manifold dimension |

### Example: The Data Manifold

Your 1000D embeddings might actually live on a 10D manifold:

```
Ambient space: ℝ¹⁰⁰⁰
Data manifold: ~10 dimensional surface in ℝ¹⁰⁰⁰

Euclidean distance in ℝ¹⁰⁰⁰: Misleading (curse of dimensionality)
Geodesic distance on manifold: Meaningful (follows the data)
```

Topology tells us the manifold exists and characterizes its shape.

---

## Part V: From Metrics to Topology

Every metric space is automatically a topological space. Here's how:

### Open Balls → Open Sets

Given a metric d, define an **open ball**:

$$B_r(x) = \{y : d(x, y) < r\}$$

The **topology induced by d** has open sets that are unions of open balls.

### What's Preserved?

When we "forget" the metric and keep only the topology, we preserve:

- Continuity of functions
- Connectedness
- Compactness
- Holes and voids

We lose:

- Exact distances
- Angles
- Volumes

### The Hierarchy

```
Metric Spaces (distances)
    ↓ forget exact distances
Topological Spaces (continuity, shape)
    ↓ forget continuity
Sets (just elements)
```

**Topology sits in the sweet spot**: abstract enough to ignore irrelevant details, concrete enough to capture meaningful structure.

---

## Part VI: Topological Spaces (Formal)

A **topological space** is a pair (X, τ) where:
- X is a set
- τ is a collection of "open sets" satisfying:
  1. ∅ and X are in τ
  2. Any union of sets in τ is in τ
  3. Any finite intersection of sets in τ is in τ

### Why This Definition?

Open sets capture "interior" vs "boundary":
- A point is **interior** if it has wiggle room (contained in an open set)
- A point is on the **boundary** if every neighborhood touches both inside and outside

### Examples Beyond Metrics

Not all topological spaces come from metrics:

| Space | Topology | ML Connection |
|-------|----------|---------------|
| Discrete | Every subset is open | Categorical features |
| Indiscrete | Only ∅ and X are open | No structure |
| Order topology | Based on ≤ ordering | Hierarchies, sequences |
| Quotient topology | Glued/identified points | Equivalence classes |

---

## Part VII: Continuity

A function f: X → Y is **continuous** if preimages of open sets are open.

**Intuition**: Small changes in input → small changes in output.

### Why Continuity Matters for ML

| Component | Should Be Continuous? | Why |
|-----------|----------------------|-----|
| Encoder | Yes | Similar inputs → similar embeddings |
| Decoder | Yes | Small embedding changes → small outputs |
| Loss function | Yes | For gradient descent to work |
| Attention | Mostly | Smooth interpolation between tokens |

### Discontinuities Are Vulnerabilities

If your model has discontinuities:
- Adversarial examples exploit them
- Small perturbations cause large changes
- The model is brittle

---

## Part VIII: The Manifold Hypothesis

A central assumption in modern ML:

> High-dimensional data lies on or near a low-dimensional manifold.

### Evidence

- **Images**: The space of "natural images" is tiny compared to all possible pixel arrays
- **Text**: Valid sentences are sparse in token-sequence space
- **Embeddings**: Learned representations cluster by semantic meaning

### Implications

1. **Dimensionality reduction** works because the intrinsic dimension is low
2. **Generative models** learn to sample from the manifold
3. **Interpolation** in latent space follows the manifold (hopefully)

### Topology of Data Manifolds

The manifold might have interesting topology:
- **Holes**: Concepts that are "surrounded" but distinct
- **Multiple components**: Disconnected clusters
- **Non-orientability**: Möbius-strip-like structure (rare but possible)

---

## Part IX: Connecting to Sheaves

Here's where it comes together for ModalSheaf:

### Modalities as a Space

Different modalities (text, image, audio) form a **discrete topological space**:
- Each modality is a point
- Open sets are subsets of modalities
- The topology encodes which modalities can "see" each other

### Data as Assignments

A **sheaf** assigns data to each region of the space:
- To each modality: the data in that modality
- To overlaps: how data should agree

### Consistency as Topology

When modalities disagree, there's a **topological obstruction**:
- Measured by cohomology (H¹)
- Zero obstruction = perfect consistency
- Nonzero obstruction = something's wrong

---

## Summary

| Concept | What It Captures | Limitation |
|---------|------------------|------------|
| Distance metrics | Local similarity | Curse of dimensionality, embedding-dependent |
| Topology | Global shape, intrinsic structure | More abstract, harder to compute |
| Sheaves | Local-to-global consistency | Requires careful setup |

The progression:
1. **Metrics** tell us which points are close
2. **Topology** tells us what shape the space has
3. **Sheaves** tell us how to consistently assign data across the space

---

## Next Steps

- [Sheaves Intuition](02_sheaves_intuition.md): The key abstraction for multimodal data
- [Gluing and Cohomology](04_gluing_and_cohomology.md): Measuring and resolving inconsistency

---

## References

- Carlsson, G. (2009). "Topology and data." *Bulletin of the AMS*.
- Ghrist, R. (2014). *Elementary Applied Topology*. [Free online](https://www.math.upenn.edu/~ghrist/notes.html)
- Munkres, J. (2000). *Topology*. Prentice Hall.
- Lee, J. (2010). *Introduction to Topological Manifolds*. Springer.
