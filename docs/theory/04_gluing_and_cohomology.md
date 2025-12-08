# Gluing and Cohomology

> *"Cohomology measures the obstruction to gluing local data into global data."*

This is the heart of sheaf theory and its application to ML. We'll see how to
detect and diagnose inconsistencies in multimodal data.

---

## The Gluing Problem

Given:
- A space X covered by regions {U₁, U₂, ..., Uₙ}
- Local data sᵢ on each region Uᵢ
- The data agrees on overlaps: sᵢ|_{Uᵢ∩Uⱼ} = sⱼ|_{Uᵢ∩Uⱼ}

Question: Does there exist global data s on X that restricts to each sᵢ?

```
     U₁        U₂        U₃
   ┌────┐    ┌────┐    ┌────┐
   │ s₁ │    │ s₂ │    │ s₃ │     Local sections
   └──┬─┘    └─┬──┘    └─┬──┘
      │   ∩    │   ∩     │
      └───┼────┴───┼─────┘
          ↓        ↓
       Agree?   Agree?              Check overlaps
          │        │
          └────┬───┘
               ↓
         ┌──────────┐
         │    s     │              Global section (if exists)
         └──────────┘
```

---

## When Gluing Fails

Gluing can fail for two reasons:

### 1. Locality Failure

Two different global sections restrict to the same local data.
(The local data doesn't uniquely determine the global.)

### 2. Gluing Failure

Local data that agrees on overlaps doesn't come from any global section.
(There's a "twist" that prevents assembly.)

**Cohomology measures the second kind of failure.**

---

## Čech Cohomology (Simplified)

Given a cover {Uᵢ} of X and a sheaf F, we define:

### Cochains

- **0-cochains** C⁰: Assignments of data to each region
  - c⁰ = (s₁, s₂, ..., sₙ) where sᵢ ∈ F(Uᵢ)

- **1-cochains** C¹: Assignments of data to each overlap
  - c¹ = (s₁₂, s₁₃, ...) where sᵢⱼ ∈ F(Uᵢ ∩ Uⱼ)

### Coboundary Operator

The coboundary δ: C⁰ → C¹ measures "how much local data disagrees":

```
(δc⁰)ᵢⱼ = ρ(sⱼ)|_{Uᵢ∩Uⱼ} - ρ(sᵢ)|_{Uᵢ∩Uⱼ}
```

If δc⁰ = 0, the local data agrees on all overlaps.

### Cohomology Groups

- **H⁰** = ker(δ⁰) = Global sections (data that's consistent everywhere)
- **H¹** = ker(δ¹) / im(δ⁰) = Obstructions to gluing

### Interpretation

| Group | Meaning | ML Interpretation |
|-------|---------|-------------------|
| H⁰ large | Many global consistent states | Strong consensus |
| H⁰ = 0 | No global consistency | Total disagreement |
| H¹ = 0 | No obstruction to gluing | All modalities agree |
| H¹ ≠ 0 | Gluing fails | Inconsistency/hallucination |

---

## The Cocycle Condition

For three overlapping regions Uᵢ, Uⱼ, Uₖ, the differences must satisfy:

```
sᵢⱼ + sⱼₖ + sₖᵢ = 0   (on Uᵢ ∩ Uⱼ ∩ Uₖ)
```

This is the **cocycle condition**. If it fails, there's a "twist" in the data.

### Example: The Möbius Strip

Consider a strip with a twist:

```
    A ────────── B
    │            │
    │   twist    │
    │            │
    B ────────── A
```

Locally, it looks like a cylinder. But globally, it has a twist that prevents
consistent orientation.

### Example: Inconsistent Sensors

Three sensors A, B, C measuring temperature:
- A-B overlap: A reads 1°C higher than B
- B-C overlap: B reads 1°C higher than C
- C-A overlap: C reads 1°C higher than A

Cocycle: 1 + 1 + 1 = 3 ≠ 0

**There's no consistent global temperature!** One sensor must be wrong.

---

## Computing Cohomology

### For Cellular Sheaves

On a graph G = (V, E), we have:
- F(v) = vector space at vertex v
- F(e) = vector space at edge e
- Restriction maps F_e←v: F(v) → F(e)

The **coboundary matrix** δ has entries:

```
δ[e, v] = F_e←v  if v is a vertex of e
        = 0      otherwise
```

Then:
- H⁰ = ker(δ)
- H¹ = coker(δ) = F(E) / im(δ)

### In ModalSheaf

```python
from modalsheaf import ConsistencyChecker

checker = ConsistencyChecker(graph)
result = checker.check_consistency(data)

print(f"H⁰ dimension: {result.h0_dim}")  # Global consistent states
print(f"H¹ dimension: {result.h1_dim}")  # Obstructions
print(f"Consistent: {result.is_consistent}")
```

---

## The Sheaf Laplacian

The **sheaf Laplacian** L is a matrix that measures total inconsistency:

```
L = δᵀδ + δδᵀ
```

Properties:
- L is positive semi-definite
- ker(L) = harmonic sections (globally consistent)
- Eigenvalues measure "how inconsistent" different modes are

### Diffusion

We can "smooth" inconsistent data by diffusing toward consistency:

```
ds/dt = -L·s
```

This drives the data toward the kernel of L (the consistent subspace).

### In ModalSheaf

```python
from modalsheaf.consistency import compute_sheaf_laplacian, diffuse_to_consensus

L = compute_sheaf_laplacian(graph)
smoothed = diffuse_to_consensus(data, L, steps=10)
```

---

## Practical Examples

### Example 1: Image-Caption Consistency

```python
# Image and caption should have similar embeddings
image_emb = clip.encode_image(image)
text_emb = clip.encode_text(caption)

# Check consistency
sections = [
    LocalSection("image", image_emb),
    LocalSection("text", text_emb),
]
overlaps = [
    Overlap(("image", "text")),  # Should match in embedding space
]

result = diagnose_gluing_problem(protocol, sections, overlaps)

if result.h1_obstruction > 0.5:
    print("Warning: Image and caption don't match!")
```

### Example 2: Sensor Fusion

```python
# Multiple sensors observing the same scene
sections = [
    LocalSection("camera", camera_data),
    LocalSection("lidar", lidar_data),
    LocalSection("radar", radar_data),
]

# Check all pairwise overlaps
overlaps = [
    Overlap(("camera", "lidar"), transform=cam_to_lidar),
    Overlap(("camera", "radar"), transform=cam_to_radar),
    Overlap(("lidar", "radar"), transform=lidar_to_radar),
]

result = diagnose_gluing_problem(protocol, sections, overlaps)

if not result.is_consistent:
    print(f"Sensor disagreement! Outliers: {result.outliers}")
```

### Example 3: Knowledge Graph Consistency

```python
# Facts from multiple sources
sections = [
    LocalSection("wikipedia", wiki_facts),
    LocalSection("wikidata", wikidata_facts),
    LocalSection("dbpedia", dbpedia_facts),
]

# Check for contradictions
result = diagnose_gluing_problem(protocol, sections, overlaps)

if result.h1_obstruction > 0:
    print("Contradictory facts detected!")
    for error in result.consistency_errors:
        print(f"  {error}")
```

---

## Diagnosing Failures

When H¹ ≠ 0, we want to know **why**. The diagnostic tools help:

### 1. Identify Outliers

```python
report = diagnose_gluing_problem(protocol, sections, overlaps)

for outlier in report.outliers:
    score = report.contributor_scores[outlier]
    print(f"{outlier}: trust={score.trust_score:.2f}")
    print(f"  Disagrees with: {score.disagreement_partners}")
```

### 2. Find Factions

```python
if report.clusters.is_polarized:
    print(f"Detected {report.clusters.num_factions} factions:")
    for i, cluster in enumerate(report.clusters.clusters):
        print(f"  Faction {i}: {cluster}")
```

### 3. Build Consensus

```python
result, excluded = find_consensus(protocol, sections, overlaps)

print(f"Excluded {excluded} to achieve consensus")
print(f"Remaining data is consistent: {result.success}")
```

---

## Connection to Homology

Cohomology is "dual" to homology:

| Homology | Cohomology |
|----------|------------|
| Counts holes | Measures obstructions |
| Cycles / Boundaries | Cocycles / Coboundaries |
| H₀ = components | H⁰ = global sections |
| H₁ = loops | H¹ = gluing failures |

**Intuition**:
- Homology asks: "What holes does my space have?"
- Cohomology asks: "What data can I consistently assign to my space?"

---

## Summary

| Concept | Definition | ML Meaning |
|---------|------------|------------|
| Gluing | Assembling locals into global | Fusion |
| Cocycle | Consistent local data | Agreeing modalities |
| Coboundary | Data from a global section | Perfectly consistent |
| H⁰ | Global sections | Consensus states |
| H¹ | Obstructions | Inconsistencies |
| Laplacian | Measures total inconsistency | Fusion error |
| Diffusion | Smooth toward consistency | Error correction |

---

## Key Takeaways

1. **H¹ = 0 is the goal**: It means all your data is consistent
2. **H¹ ≠ 0 means trouble**: Something is wrong (sensor, model, data)
3. **Diagnostics help**: Identify which source is causing problems
4. **Diffusion helps**: Smooth toward the most consistent state

---

## Next Steps

- [Applications to ML](05_applications_to_ml.md): Real-world use cases
- [Topological Loss Characterization](06_topological_loss.md): Understanding what's lost in transformations
- [API Reference](../api/consistency.md): Implementation details

---

## References

- Robinson, M. (2014). *Topological Signal Processing*. Chapters 4-5.
- Bodnar et al. (2022). "Neural Sheaf Diffusion." NeurIPS.
- Hansen & Ghrist (2019). "Toward a spectral theory of cellular sheaves."
- Ghrist, R. (2014). *Elementary Applied Topology*. Chapter 4.
