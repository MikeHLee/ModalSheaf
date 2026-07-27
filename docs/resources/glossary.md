# Glossary

A reference of key terms used in ModalSheaf, with both mathematical definitions
and ML interpretations.

---

## A

### Adjoint Functors
**Math**: A pair of functors F ⊣ G where Hom(F(A), B) ≅ Hom(A, G(B)).
**ML**: Encoder-decoder pairs where encoding then decoding gives a "best approximation."

### Alexandrov Topology
**Math**: A topology on a poset where open sets are upward-closed.
**ML**: The natural topology on hierarchical data structures.

---

## B

### Base Space
**Math**: The topological space X on which a sheaf is defined.
**ML**: The graph/network of modalities in a multimodal system.

### Bundle
**Math**: A space E with a projection π: E → X to a base space.
**ML**: A family of data spaces parameterized by modality.

---

## C

### Čech Cohomology
**Math**: Cohomology computed using covers and their overlaps.
**ML**: A method for detecting inconsistencies in multimodal data.

### Cellular Sheaf
**Math**: A sheaf defined on a cell complex (vertices, edges, faces, etc.).
**ML**: A sheaf on a graph, with data at nodes and edges.

### Coboundary
**Math**: The image of the coboundary operator δ.
**ML**: Differences that come from restricting a global section.

### Cochain
**Math**: An element of the cochain complex Cⁿ.
**ML**: An assignment of data to n-fold overlaps.

### Cocycle
**Math**: An element of ker(δ) — a cochain with zero coboundary.
**ML**: Local data that satisfies the consistency condition.

### Cohomology
**Math**: The quotient ker(δⁿ) / im(δⁿ⁻¹).
**ML**: A measure of obstructions to gluing local data into global data.

### Consistency
**Math**: When local sections agree on overlaps.
**ML**: When different modalities/sensors give compatible information.

### Continuous Function
**Math**: A function where preimages of open sets are open.
**ML**: A transformation that preserves "nearness" — similar inputs give similar outputs.

### Contravariant
**Math**: A functor that reverses arrows.
**ML**: Restriction maps go "backwards" — from larger to smaller regions.

### Cosheaf
**Math**: A covariant functor from open sets to a category.
**ML**: A structure with extension maps (opposite of sheaf).

### Covariant
**Math**: A functor that preserves arrow direction.
**ML**: Extension maps go "forwards" — from smaller to larger regions.

### Cover
**Math**: A collection of open sets whose union is the whole space.
**ML**: A set of modalities/sensors that together observe everything.

---

## D

### Descent Data
**Math**: Local data with compatibility conditions for gluing.
**ML**: Multimodal data that's ready to be fused.

### Diffusion
**Math**: A process driven by the Laplacian: ds/dt = -Ls.
**ML**: Smoothing data toward consistency by iterative averaging.

---

## E

### Embedding (Topological)
**Math**: An injective continuous map with continuous inverse on its image.
**ML**: A lossless encoding into a larger space.

### Extension Map
**Math**: A map from F(V) to F(U) where V ⊆ U.
**ML**: Aggregating local data to global (decoder, upsampler).

---

## F

### Fiber
**Math**: The preimage π⁻¹(x) of a point in a bundle.
**ML**: The data space at a single modality.

### Functor
**Math**: A structure-preserving map between categories.
**ML**: A systematic way of transforming one type of structure to another.

---

## G

### Germ
**Math**: An equivalence class of local sections at a point.
**ML**: The "essential" local information at a modality.

### Global Section
**Math**: A section defined on the entire base space.
**ML**: A consistent assignment of data to all modalities.

### Gluing
**Math**: Assembling local sections into a global section.
**ML**: Fusing data from multiple sources/modalities.

### Gluing Axiom
**Math**: If local sections agree on overlaps, they glue to a unique global section.
**ML**: Consistent local data can be assembled into global data.

---

## H

### H⁰ (Zeroth Cohomology)
**Math**: The space of global sections.
**ML**: The set of globally consistent states.

### H¹ (First Cohomology)
**Math**: Obstructions to gluing.
**ML**: A measure of inconsistency in multimodal data.

### Harmonic Section
**Math**: A section in the kernel of the Laplacian.
**ML**: A maximally consistent data assignment.

### Homeomorphism
**Math**: A continuous bijection with continuous inverse.
**ML**: A lossless, reversible transformation.

### Homology
**Math**: Algebraic invariants counting "holes" in a space.
**ML**: Features of data that persist across scales.

### Homomorphism
**Math**: A structure-preserving map.
**ML**: A transformation that respects the data structure.

---

## I

### Isomorphism
**Math**: A bijective homomorphism with inverse also a homomorphism.
**ML**: A perfect, lossless, reversible transformation.

---

## L

### Laplacian (Sheaf)
**Math**: L = δᵀδ + δδᵀ, measuring total inconsistency.
**ML**: A matrix whose eigenvalues indicate inconsistency modes.

### Local Section
**Math**: A section defined on an open subset.
**ML**: Data from a single modality/sensor.

### Locality Axiom
**Math**: Sections are determined by their restrictions to a cover.
**ML**: Global data is determined by its local parts.

---

## M

### Manifold
**Math**: A space locally homeomorphic to ℝⁿ.
**ML**: A curved surface in high-dimensional space where data lives.

### Metric Space
**Math**: A set with a distance function.
**ML**: A space where we can measure similarity.

### Modality
**Math**: A point in the base space of a sheaf.
**ML**: A type of data (text, image, audio, etc.).

### Monomorphism
**Math**: An injective morphism.
**ML**: An embedding that loses no information.

---

## O

### Open Set
**Math**: An element of a topology.
**ML**: A "region" in the space of modalities.

### Overlap
**Math**: The intersection of two open sets.
**ML**: Where two modalities/sensors have shared coverage.

---

## P

### Persistent Homology
**Math**: Homology tracked across a filtration.
**ML**: Topological features that persist across scales.

### Poset
**Math**: A partially ordered set.
**ML**: A hierarchy (e.g., pixel → patch → image → video).

### Presheaf
**Math**: A contravariant functor from open sets to a category.
**ML**: Local data without consistency requirements.

### Projection
**Math**: A surjective map.
**ML**: A lossy transformation that reduces information.

---

## R

### Restriction Map
**Math**: A map from F(U) to F(V) where V ⊆ U.
**ML**: Extracting local from global (encoder, cropper).

---

## S

### Section
**Math**: A right inverse to a projection; an element of F(U).
**ML**: A data assignment to a region.

### Sheaf
**Math**: A presheaf satisfying locality and gluing axioms.
**ML**: A consistent system of local data with compatibility on overlaps.

### Sheaf Condition
**Math**: Locality + Gluing axioms.
**ML**: Local data determines global, and consistent locals glue.

### Stalk
**Math**: The colimit of F(U) over all U containing a point x.
**ML**: The data type at a single modality.

---

## T

### Topological Space
**Math**: A set with a collection of open sets satisfying axioms.
**ML**: Any space with a notion of "nearness" or "continuity."

### Topology
**Math**: The study of properties preserved under continuous deformation.
**ML**: The study of data structure beyond just distances.

### Transformation
**Math**: A map between spaces.
**ML**: An encoder, decoder, or converter between modalities.

---

## V

### Vector Bundle
**Math**: A bundle where each fiber is a vector space.
**ML**: A family of embedding spaces over modalities.

---

## Quick Reference Table

| Term | One-Line Definition | ML Analog |
|------|---------------------|-----------|
| Sheaf | Local data + compatibility | Multimodal system |
| Section | Data assignment | Embedding |
| Restriction | Global → Local | Encoder |
| Extension | Local → Global | Decoder |
| Gluing | Assemble locals | Fusion |
| H⁰ | Global sections | Consensus |
| H¹ | Obstructions | Inconsistency |
| Laplacian | Inconsistency measure | Fusion error |
| Stalk | Data at a point | Modality format |
| Cover | Regions that cover space | All sensors |
