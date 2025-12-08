# Sheaves: The Key Abstraction

> *"A sheaf is a way of systematically assigning data to the open sets of a space,
> with compatibility conditions on overlaps."*

Sheaves are the central concept in this library. This chapter builds intuition before
diving into formalism.

---

## The Core Idea: Local Data, Global Consistency

Imagine you're assembling a map of a country:

1. Each surveyor maps their local region
2. Where regions overlap, the maps must agree
3. If all overlaps agree, you can glue the local maps into a global map

This is exactly what a sheaf formalizes.

---

## Motivating Examples

### Example 1: Temperature Field

Suppose we have temperature sensors scattered across a room:

```
    Sensor A: 20°C
         ↘
           (overlap)
         ↗
    Sensor B: 21°C
```

Each sensor measures temperature in its region. In the overlap, both sensors should
(approximately) agree. If they don't, something is wrong.

**Sheaf perspective**:
- Space: The room
- Open sets: Regions covered by sensors
- Stalk at a point: Temperature at that point
- Section: A consistent temperature assignment

### Example 2: Multimodal Embeddings

Consider an image-caption pair:

```
    Image → CLIP Image Encoder → Embedding (768-dim)
    Caption → CLIP Text Encoder → Embedding (768-dim)
```

Both embeddings should be "close" if the caption describes the image.

**Sheaf perspective**:
- Space: {image, text, shared}
- Stalks: Embedding spaces
- Restriction: Encoders
- Consistency: Embeddings should match in shared space

### Example 3: Sensor Fusion

Multiple sensors observe the same scene:

```
    Camera → 2D projection
    LiDAR → 3D point cloud
    Radar → Velocity field
```

All should be consistent with the underlying 3D world.

**Sheaf perspective**:
- Space: Sensor network
- Stalks: Each sensor's data format
- Restriction: Projection/transformation
- Global section: Consistent 3D reconstruction

---

## Formal Definition (Simplified)

A **sheaf** F on a topological space X assigns:

1. **To each open set U ⊆ X**: A set F(U) called "sections over U"
2. **To each inclusion V ⊆ U**: A "restriction map" ρ_{U,V}: F(U) → F(V)

Subject to:

1. **Identity**: ρ_{U,U} = id
2. **Composition**: ρ_{U,V} ∘ ρ_{V,W} = ρ_{U,W}
3. **Locality**: If s, t ∈ F(U) agree on a cover, then s = t
4. **Gluing**: If local sections agree on overlaps, they glue to a global section

### The Gluing Axiom (Key!)

Given:
- A cover {Uᵢ} of U (meaning U = ∪Uᵢ)
- Sections sᵢ ∈ F(Uᵢ) for each i
- Agreement: ρ(sᵢ)|_{Uᵢ∩Uⱼ} = ρ(sⱼ)|_{Uᵢ∩Uⱼ}

Then:
- There exists a unique s ∈ F(U) with ρ(s)|_{Uᵢ} = sᵢ

**In plain English**: If local data agrees on overlaps, it assembles into global data.

---

## Sheaves vs Presheaves

A **presheaf** satisfies conditions 1-2 but not necessarily 3-4.

| Property | Presheaf | Sheaf |
|----------|----------|-------|
| Assigns data to open sets | ✓ | ✓ |
| Has restriction maps | ✓ | ✓ |
| Locality (uniqueness) | ✗ | ✓ |
| Gluing (existence) | ✗ | ✓ |

**ML interpretation**: A presheaf is like having local models that don't necessarily
agree. A sheaf is when they're consistent.

---

## Stalks: Data at a Point

The **stalk** F_x at a point x is the "germ" of data at x — the limit of F(U) as U
shrinks to x.

**Intuition**: The stalk captures all the local information at a point, ignoring
what happens far away.

### For ML

| Concept | ML Interpretation |
|---------|-------------------|
| Point x | A specific modality or sensor |
| Stalk F_x | Data format at that modality |
| Section s | Consistent data across modalities |
| Restriction | Transformation between modalities |

---

## Examples of Sheaves

### The Constant Sheaf

F(U) = A for all non-empty U, where A is a fixed set.

**ML example**: Every modality uses the same embedding dimension.

### The Sheaf of Continuous Functions

F(U) = {continuous functions f: U → ℝ}

Restriction is just function restriction: ρ(f)|_V = f|_V.

**ML example**: The space of valid embeddings on each region.

### The Sheaf of Sections of a Bundle

Given a vector bundle E → X, the sheaf of sections assigns:
F(U) = {continuous sections s: U → E}

**ML example**: Each modality has its own embedding space (fiber), and sections
are consistent embeddings.

---

## Cellular Sheaves (Computational)

For computation, we often use **cellular sheaves** on graphs or cell complexes.

### On a Graph

Given a graph G = (V, E):
- Assign a vector space F(v) to each vertex v
- Assign a vector space F(e) to each edge e
- For each edge e = (u, v), have restriction maps:
  - F_e←u: F(u) → F(e)
  - F_e←v: F(v) → F(e)

**Consistency**: Data on u and v is consistent if their restrictions to e agree.

### Example: Sensor Network

```
    Sensor A -------- Sensor B
       |                 |
       |                 |
    Sensor C -------- Sensor D
```

- F(sensor) = ℝ (temperature reading)
- F(edge) = ℝ (should be equal)
- Restriction = identity

A global section exists iff all sensors agree.

---

## Sheaves in ModalSheaf

In this library:

```python
from modalsheaf import ModalityGraph

# Create the "space" (graph of modalities)
graph = ModalityGraph("my_sheaf")

# Add "stalks" (data types at each modality)
graph.add_modality("text")
graph.add_modality("image")
graph.add_modality("embedding", shape=(768,))

# Add "restriction maps" (transformations)
graph.add_transformation("text", "embedding", forward=text_encoder)
graph.add_transformation("image", "embedding", forward=image_encoder)
```

The graph structure defines the "space," and transformations define the "sheaf."

---

## Why Sheaves for ML?

1. **Natural framework for multimodal data**: Each modality is a region, embeddings
   are sections, encoders are restrictions.

2. **Consistency is built-in**: The sheaf axioms tell us when data is consistent.

3. **Cohomology measures failure**: When gluing fails, H¹ tells us how badly.

4. **Compositional**: Sheaves compose nicely — you can build complex systems from
   simple parts.

---

## Cosheaves: The Dual

A **cosheaf** is like a sheaf but with arrows reversed:
- Extension maps go from smaller to larger regions
- Gluing goes from global to local

| Sheaf | Cosheaf |
|-------|---------|
| Restriction (zoom in) | Extension (zoom out) |
| Local → Global | Global → Local |
| Contravariant | Covariant |

**ML example**:
- Sheaf: Encoder (image → embedding)
- Cosheaf: Decoder (embedding → image)

---

## Summary

| Concept | Intuition | ML Analog |
|---------|-----------|-----------|
| Space | Where data lives | Modality graph |
| Open set | Region of interest | Subset of modalities |
| Stalk | Data at a point | Data format |
| Section | Consistent assignment | Multimodal embedding |
| Restriction | Zoom in | Encoder |
| Gluing | Assemble locals | Fusion |
| Cohomology | Obstruction | Inconsistency measure |

---

## Next Steps

- [Restriction and Extension](03_restriction_extension.md): The maps between regions
- [Gluing and Cohomology](04_gluing_and_cohomology.md): When gluing fails

---

## References

- Robinson, M. (2014). *Topological Signal Processing*. Chapters 2-3.
- Curry, J. (2014). *Sheaves, Cosheaves and Applications*. Chapter 2.
- Mac Lane & Moerdijk (1994). *Sheaves in Geometry and Logic*. Chapter 2.
