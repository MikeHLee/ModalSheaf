# Flavors of Sheaves: Cellular, Topological, and Sites

> *"The same idea, dressed differently for different occasions."*

There are several "flavors" of sheaves in mathematics. This guide explains the
differences and why ModalSheaf uses **cellular sheaves**.

It also covers the crucial concept of **covers** — the foundation of all sheaf theory.

---

## Part 0: Covers — The Foundation of Everything

Before we talk about different types of sheaves, we need to understand **covers**.
This concept is absolutely fundamental.

### What is a Cover?

A **cover** of a space X is a collection of "pieces" that together make up all of X.

```
Space X = a country

Cover = {California, Nevada, Oregon, ...}

The states "cover" the country — every point in the country
is in at least one state.
```

### Formal Definition

A cover of X is a collection {Uᵢ} such that:

```
X = ∪ᵢ Uᵢ   (union of all pieces equals the whole)
```

### Why Covers Matter

Covers let us work **locally** and then assemble **globally**:

1. **Local**: Solve the problem on each piece Uᵢ
2. **Overlap**: Check that solutions agree where pieces overlap
3. **Global**: If they agree, glue into a solution on all of X

This is the **local-to-global principle** — the heart of sheaf theory.

### Examples of Covers

| Space | Cover | Overlaps |
|-------|-------|----------|
| Country | States | State borders |
| Image | Patches | Patch boundaries |
| Sensor network | Sensor ranges | Overlapping coverage |
| Time series | Windows | Window overlaps |
| Codebase | Files | Import dependencies |

### Covers in ModalSheaf

In ModalSheaf, the **modalities are the cover**:

```python
from modalsheaf import ModalityGraph

graph = ModalityGraph()

# Each modality is a "piece" of the cover
graph.add_modality("image")    # U₁
graph.add_modality("text")     # U₂  
graph.add_modality("audio")    # U₃

# Transformations define how pieces "overlap"
# (they can be compared via a common embedding)
graph.add_transformation("image", "embedding", ...)
graph.add_transformation("text", "embedding", ...)
```

The "overlap" between image and text is the shared embedding space where
both can be compared.

### Good Covers vs Bad Covers

Not all covers are equally useful:

**Good cover** (for computation):
- Finite number of pieces
- Simple overlaps (pairwise, maybe triple)
- Clear overlap structure

**Bad cover** (hard to compute):
- Infinite pieces
- Complex overlap patterns
- Overlaps of overlaps of overlaps...

Cellular sheaves use **good covers** by design — the vertices and edges
of a graph give a finite, well-structured cover.

### The Čech Perspective

The **Čech complex** organizes a cover for computation:

```
Cover: {U₁, U₂, U₃}

C⁰ = data on each piece
     {U₁: data₁, U₂: data₂, U₃: data₃}

C¹ = data on pairwise overlaps
     {U₁∩U₂: data₁₂, U₁∩U₃: data₁₃, U₂∩U₃: data₂₃}

C² = data on triple overlaps
     {U₁∩U₂∩U₃: data₁₂₃}
```

The **coboundary** δ checks if data agrees on overlaps.
**Cohomology** measures the obstruction to gluing.

### Key Insight

> **The choice of cover determines what "local" and "global" mean.**

Different covers give different perspectives:
- Fine cover (many small pieces): More local detail, more overlaps to check
- Coarse cover (few large pieces): Less detail, fewer overlaps

In ML terms:
- Fine: Each sensor is a piece → detailed fusion
- Coarse: Each modality type is a piece → high-level consistency

---

## The Big Picture: Three Flavors

| Flavor | Base Space | Best For | Computability |
|--------|------------|----------|---------------|
| **Classical Sheaves** | Topological space | Continuous data | Hard |
| **Sheaves on Sites** | Category with topology | Abstract relationships | Very hard |
| **Cellular Sheaves** | Graph or cell complex | Discrete/network data | Easy! ✓ |

ModalSheaf uses **cellular sheaves** because they're:
1. Computationally tractable (linear algebra!)
2. Perfect for network/graph data (sensors, modalities)
3. Still capture the essential sheaf properties

---

## Part 1: Classical Sheaves (The Original)

### The Setup

A classical sheaf lives on a **topological space** X (like ℝⁿ, a manifold, etc.).

```
Space X = a circle S¹

Open sets: arcs of the circle
    ___
   /   \    U₁ = top arc
  |     |
   \___/    U₂ = bottom arc
   
Sheaf assigns:
  F(U₁) = functions on top arc
  F(U₂) = functions on bottom arc
  F(U₁ ∩ U₂) = functions on overlap (two points!)
```

### The Problem

Classical sheaves require:
- Infinitely many open sets
- Continuous restriction maps
- Limits and colimits

**This is hard to compute!** You can't represent "all open sets" in a computer.

### When to Use

Classical sheaves are great for:
- Theoretical mathematics
- Algebraic geometry
- Complex analysis

But not for practical computation.

---

## Part 2: Sheaves on Sites (The Abstract)

### The Setup

A **site** is a category C with a "Grothendieck topology" — a way of saying
which collections of morphisms "cover" an object.

```
Category C:
  Objects: {A, B, C}
  Morphisms: A → B, B → C, A → C
  
Grothendieck topology:
  "A is covered by {A → B, A → C}"
  
Sheaf assigns:
  F(A), F(B), F(C) with compatibility conditions
```

### Why So Abstract?

Sites generalize topological spaces. They can represent:
- Algebraic varieties (étale site)
- Schemes (Zariski site)
- Categories themselves (presheaf topos)

### The Problem

Sites are **extremely abstract**:
- Requires category theory fluency
- Grothendieck topologies are subtle
- Computationally intractable in general

### When to Use

Sheaves on sites are for:
- Algebraic geometry (étale cohomology)
- Topos theory
- Foundations of mathematics

**Not** for practical ML applications!

---

## Part 3: Cellular Sheaves (The Practical)

### The Setup

A **cellular sheaf** lives on a **cell complex** — typically a graph.

```
Graph G:
    A ----e₁---- B
    |            |
   e₃           e₂
    |            |
    C ----e₄---- D

Vertices: {A, B, C, D}
Edges: {e₁, e₂, e₃, e₄}
```

A cellular sheaf assigns:
- A **vector space** F(v) to each vertex v
- A **vector space** F(e) to each edge e
- A **linear map** F_{e←v}: F(v) → F(e) for each incidence

### Example: Temperature Sensors

```
Sensor A -------- Sensor B
   |                  |
   |                  |
Sensor C -------- Sensor D

F(A) = ℝ  (temperature at A)
F(B) = ℝ  (temperature at B)
F(e₁) = ℝ (temperature on edge A-B)

Restriction F_{e₁←A}: F(A) → F(e₁)
  "What A thinks the edge temperature is"
  
Restriction F_{e₁←B}: F(B) → F(e₁)
  "What B thinks the edge temperature is"
```

**Consistency**: A and B agree on edge e₁ if F_{e₁←A}(data_A) = F_{e₁←B}(data_B)

### Why Cellular Sheaves Win

1. **Finite**: Only finitely many cells
2. **Linear**: All maps are matrices
3. **Computable**: Standard linear algebra
4. **Intuitive**: Graphs are familiar

### In ModalSheaf

```python
from modalsheaf import ModalityGraph

# This IS a cellular sheaf!
graph = ModalityGraph()

# Vertices (stalks)
graph.add_modality("image", shape=(768,))    # F(image) = ℝ⁷⁶⁸
graph.add_modality("text", shape=(768,))     # F(text) = ℝ⁷⁶⁸
graph.add_modality("embedding", shape=(768,)) # F(embedding) = ℝ⁷⁶⁸

# Edges (restriction maps)
graph.add_transformation("image", "embedding", forward=image_encoder)
graph.add_transformation("text", "embedding", forward=text_encoder)
```

---

## Part 4: The Sheaf Laplacian — Where Linear Algebra Meets Topology

### The Key Insight

For cellular sheaves, we can encode **all consistency conditions** in a single
matrix: the **sheaf Laplacian**.

### Building the Laplacian

#### Step 1: The Coboundary Map

For each edge e = (u, v), define:

```
δ: ⊕_v F(v) → ⊕_e F(e)

(δx)_e = F_{e←v}(x_v) - F_{e←u}(x_u)
```

This measures "disagreement" on each edge.

#### Step 2: The Laplacian

```
L = δᵀ δ
```

The Laplacian L is a matrix acting on vertex data.

### What the Laplacian Tells Us

| Property | Meaning |
|----------|---------|
| **Lx = 0** | x is globally consistent |
| **ker(L)** | Space of consistent states (= H⁰) |
| **eigenvalues** | "Frequencies" of inconsistency |
| **small eigenvalues** | Near-consistent modes |

### Intuition: Heat Diffusion

The Laplacian is like a **heat equation**:

```
dx/dt = -Lx
```

- Heat flows from hot to cold
- Data "flows" from inconsistent to consistent
- Equilibrium (steady state) = global consistency

### Example: Three Sensors

```
    A
   / \
  /   \
 B-----C
```

With identity restriction maps (all sensors should agree):

```python
import numpy as np
from modalsheaf.consistency import compute_sheaf_laplacian

# Incidence: edges × vertices
#          A   B   C
# e_AB  [  1  -1   0 ]
# e_AC  [  1   0  -1 ]
# e_BC  [  0   1  -1 ]

incidence = np.array([
    [1, -1, 0],
    [1, 0, -1],
    [0, 1, -1]
])

# Laplacian L = δᵀδ
L = incidence.T @ incidence
print("Laplacian:")
print(L)
# [[ 2 -1 -1]
#  [-1  2 -1]
#  [-1 -1  2]]

# Eigenvalues
eigenvalues = np.linalg.eigvalsh(L)
print(f"Eigenvalues: {eigenvalues}")
# [0, 3, 3]

# Zero eigenvalue → 1D kernel → H⁰ = ℝ¹
# The kernel is spanned by [1, 1, 1] (constant = consensus)
```

### The Laplacian in ModalSheaf

```python
from modalsheaf import ModalityGraph, ConsistencyChecker
from modalsheaf.consistency import compute_sheaf_laplacian

graph = ModalityGraph()
graph.add_modality("A")
graph.add_modality("B")
graph.add_modality("C")
graph.add_transformation("A", "B", forward=lambda x: x)
graph.add_transformation("B", "C", forward=lambda x: x)
graph.add_transformation("A", "C", forward=lambda x: x)

# Compute Laplacian
L = compute_sheaf_laplacian(graph, dim=1)

# Kernel dimension = H⁰ dimension
eigenvalues = np.linalg.eigvalsh(L)
h0_dim = np.sum(np.abs(eigenvalues) < 1e-10)
print(f"H⁰ dimension: {h0_dim}")
```

---

## Part 5: Diffusion to Consensus

### The Problem

Given inconsistent data, can we find the "best" consistent approximation?

### The Solution: Heat Diffusion

Run the heat equation:

```
dx/dt = -Lx
```

As t → ∞, x converges to the **projection onto ker(L)** = the consensus.

### Example

```python
from modalsheaf.consistency import diffuse_to_consensus

# Inconsistent readings
data = {
    'sensor_a': np.array([20.0]),
    'sensor_b': np.array([21.0]),
    'sensor_c': np.array([22.0]),
}

# Diffuse to consensus
consensus = diffuse_to_consensus(graph, data, steps=100)
print(f"Consensus: {consensus}")
# All sensors converge to ~21.0 (the average)
```

### Why This Works

The Laplacian has the property:

```
xᵀLx = Σ_e ||F_{e←u}(x_u) - F_{e←v}(x_v)||²
```

This is the **total squared disagreement**. Minimizing it gives the consensus.

---

## Part 6: Comparison Table

| Aspect | Classical | Site | Cellular |
|--------|-----------|------|----------|
| **Base** | Topological space | Category | Graph/complex |
| **Open sets** | Infinite | Abstract | Finite cells |
| **Stalks** | General | General | Vector spaces |
| **Restrictions** | Continuous | Morphisms | Linear maps |
| **Cohomology** | Čech/derived | Topos | Linear algebra |
| **Laplacian** | ✗ | ✗ | ✓ |
| **Computable** | Hard | Very hard | Easy |
| **ML-friendly** | ✗ | ✗ | ✓ |

---

## Part 7: When to Use What

### Use Cellular Sheaves (ModalSheaf) When:

- ✓ Data lives on a graph/network
- ✓ You have discrete sensors/modalities
- ✓ You need actual numbers (consistency scores, etc.)
- ✓ You want to compute cohomology
- ✓ You need diffusion/consensus algorithms

### Use Classical Sheaves When:

- You're doing pure mathematics
- Your space is genuinely continuous
- You don't need to compute anything

### Use Sites When:

- You're doing algebraic geometry
- You need étale cohomology
- You're a category theorist

---

## Summary

```
Classical Sheaves          Sites                  Cellular Sheaves
      |                      |                          |
  Continuous             Abstract                   Discrete
  Infinite               Categorical                Finite
  Theoretical            Very theoretical           Computational
      |                      |                          |
      +----------------------+----------> ModalSheaf uses this!
```

**Key takeaways:**

1. **Cellular sheaves** are the computational workhorse
2. The **Laplacian** encodes all consistency conditions
3. **Diffusion** finds consensus from inconsistent data
4. ModalSheaf gives you the full power of sheaf theory in a computable form

---

## References

1. **Cellular Sheaves**: Hansen & Ghrist (2019). "Toward a Spectral Theory of Cellular Sheaves."
2. **Laplacians**: Robinson (2014). *Topological Signal Processing*. Chapter 4.
3. **Sites**: Mac Lane & Moerdijk (1994). *Sheaves in Geometry and Logic*. Chapter III.
4. **Applications**: Curry (2014). "Sheaves, Cosheaves and Applications."
