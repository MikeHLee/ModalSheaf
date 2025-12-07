# Spaces and Continuity

> *"A topological space is a set with just enough structure to define continuity."*

This chapter introduces the formal language of topology, but always with an eye toward
ML applications. We'll see that "space" is far more general than ℝⁿ.

---

## What is a Space?

In ML, we work with many kinds of "spaces":

| Space | Elements | Example |
|-------|----------|---------|
| Euclidean ℝⁿ | Vectors | Embeddings |
| Image space | Pixel arrays | H×W×C tensors |
| Text space | Token sequences | Variable-length lists |
| Graph space | Node/edge structures | Knowledge graphs |
| Model space | Neural network weights | Parameter tensors |

What makes these "spaces" rather than just "sets"? They have **structure** that tells
us which elements are "close" or "similar."

---

## Topological Spaces (Formal Definition)

A **topological space** is a pair (X, τ) where:
- X is a set
- τ is a collection of subsets of X called "open sets"

The open sets must satisfy:
1. ∅ and X are open
2. Any union of open sets is open
3. Any finite intersection of open sets is open

### Why Open Sets?

Open sets capture the idea of "interior" vs "boundary." A point is in the interior of
a set if there's some "wiggle room" around it.

**Intuition**: In ℝ, the interval (0, 1) is open because every point has neighbors
also in the interval. The interval [0, 1] is closed because 0 and 1 are on the boundary.

### For ML

You rarely need to think about open sets directly. What matters is:

1. **Continuity**: A function f: X → Y is continuous if it "preserves nearness"
2. **Connectedness**: Can you walk between any two points?
3. **Compactness**: Is the space "finite" in a topological sense?

---

## Metric Spaces: A Special Case

Most ML spaces are **metric spaces**, where we have a distance function d(x, y).

A metric must satisfy:
1. d(x, y) ≥ 0, with d(x, y) = 0 iff x = y
2. d(x, y) = d(y, x) (symmetry)
3. d(x, z) ≤ d(x, y) + d(y, z) (triangle inequality)

### Common Metrics in ML

| Metric | Formula | Use Case |
|--------|---------|----------|
| Euclidean (L2) | √(Σ(xᵢ - yᵢ)²) | Embeddings |
| Manhattan (L1) | Σ|xᵢ - yᵢ| | Sparse data |
| Cosine | 1 - (x·y)/(‖x‖‖y‖) | Similarity search |
| Edit distance | Min edits to transform | Strings |
| Wasserstein | Optimal transport | Distributions |

### Metric → Topology

Every metric space is automatically a topological space:
- Open sets are unions of "open balls" Bᵣ(x) = {y : d(x,y) < r}

But topology is more general — some spaces have no natural metric.

---

## Vector Spaces and Tensor Spaces

ML lives in vector and tensor spaces. Let's connect these to topology.

### Vector Spaces

A **vector space** V over ℝ has:
- Addition: v + w
- Scalar multiplication: αv

Examples: ℝⁿ, function spaces, embedding spaces.

### Normed Spaces

A **norm** ‖·‖ measures vector "size":
1. ‖v‖ ≥ 0, with ‖v‖ = 0 iff v = 0
2. ‖αv‖ = |α|‖v‖
3. ‖v + w‖ ≤ ‖v‖ + ‖w‖

A norm induces a metric: d(v, w) = ‖v - w‖

### Tensor Spaces

Tensors are multi-dimensional arrays. An (n₁ × n₂ × ... × nₖ) tensor lives in
ℝ^(n₁·n₂·...·nₖ), which is a vector space.

**Key insight**: All finite-dimensional tensor spaces are topologically equivalent to
some ℝⁿ. The "shape" is just bookkeeping.

---

## Function Spaces

Here's where it gets interesting for ML. Consider:

- The space of all functions f: ℝ → ℝ
- The space of all neural networks with fixed architecture
- The space of all probability distributions on ℝⁿ

These are **infinite-dimensional** spaces with rich topological structure.

### Example: L² Space

The space L²([0,1]) consists of square-integrable functions on [0,1]:

```
L² = { f : ∫₀¹ |f(x)|² dx < ∞ }
```

This is a **Hilbert space** with inner product ⟨f, g⟩ = ∫ f(x)g(x) dx.

**ML connection**: Kernel methods implicitly work in such spaces.

### Example: Model Space

The space of neural networks with architecture A:

```
M_A = { θ ∈ ℝᵈ : θ are the parameters }
```

This is just ℝᵈ, but the **loss landscape** gives it interesting topology:
- Local minima are "valleys"
- Saddle points are "passes"
- The path between minima matters for optimization

---

## Continuity: The Key Property

A function f: X → Y is **continuous** if:
- Small changes in input produce small changes in output
- Formally: preimages of open sets are open

### Why Continuity Matters for ML

1. **Encoders should be continuous**: Similar inputs → similar embeddings
2. **Decoders should be continuous**: Small embedding changes → small output changes
3. **Loss functions should be continuous**: For gradient-based optimization

### Discontinuities are Problems

If your encoder is discontinuous:
- Adversarial examples exploit the discontinuity
- Small perturbations cause large output changes
- The model is not robust

---

## Homeomorphism: Topological Equivalence

Two spaces X and Y are **homeomorphic** (topologically equivalent) if there's a
continuous bijection f: X → Y with continuous inverse.

### Classic Example

A coffee cup and a donut are homeomorphic — you can continuously deform one into
the other (both have one hole).

### ML Example

The unit sphere Sⁿ⁻¹ in ℝⁿ is NOT homeomorphic to ℝⁿ⁻¹:
- The sphere is compact (bounded and closed)
- ℝⁿ⁻¹ is not compact

**Implication**: You can't continuously flatten a sphere without tearing or overlapping.
This is why projecting high-dimensional data to 2D always loses something.

---

## Manifolds: Locally Euclidean Spaces

A **manifold** is a space that locally looks like ℝⁿ, but globally may have
interesting topology.

### Examples

| Manifold | Local structure | Global structure |
|----------|-----------------|------------------|
| Circle S¹ | Looks like ℝ¹ | Wraps around |
| Sphere S² | Looks like ℝ² | Closed surface |
| Torus T² | Looks like ℝ² | Two holes |
| Klein bottle | Looks like ℝ² | Non-orientable! |

### The Manifold Hypothesis

A key assumption in ML:

> High-dimensional data often lies on or near a low-dimensional manifold.

For example:
- Images of faces form a manifold in pixel space
- Natural language forms a manifold in token space
- Valid programs form a manifold in code space

**Implication**: We should learn the manifold structure, not just memorize points.

---

## Posets and Order Topology

Not all spaces are geometric. A **partially ordered set (poset)** has a different
kind of structure.

### Definition

A poset (P, ≤) has a reflexive, antisymmetric, transitive order.

### Examples in ML

| Poset | Order | Use |
|-------|-------|-----|
| Subsets of a set | ⊆ | Feature hierarchies |
| Prefixes of strings | "is prefix of" | Autoregressive models |
| Coarsenings of a partition | "is coarser than" | Clustering hierarchies |
| Open sets of a topology | ⊆ | Sheaf theory! |

### Alexandrov Topology

Every poset has a natural topology where:
- Open sets are "upward closed": if x ∈ U and x ≤ y, then y ∈ U

This connects order structure to topological structure.

---

## Summary: Spaces in ModalSheaf

In ModalSheaf, we work with several kinds of spaces:

| Space | Type | Role |
|-------|------|------|
| Modality space | Discrete/poset | The "base space" |
| Data spaces | Vector/tensor | Where data lives |
| Embedding space | Normed vector | Common representation |
| Transformation space | Function space | Maps between modalities |

The key insight is that **modalities form a space**, and **sheaves assign data to
regions of that space**.

---

## Next Steps

- [Sheaves Intuition](02_sheaves_intuition.md): Assigning data to spaces
- [Restriction and Extension](03_restriction_extension.md): Moving between regions

---

## References

- Munkres, J. (2000). *Topology*. Prentice Hall. (The standard textbook)
- Lee, J. (2010). *Introduction to Topological Manifolds*. Springer.
- Ghrist, R. (2014). *Elementary Applied Topology*. Chapter 1.
