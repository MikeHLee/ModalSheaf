# Advanced Cohomology: From Theory to Computation

> *"The purpose of computing is insight, not numbers."* — Richard Hamming

This document bridges the gap between the intuitive introduction and the rigorous
mathematical machinery. We'll see how abstract concepts become concrete algorithms.

---

## Part 1: The Čech Complex — Making Cohomology Computable

### The Problem with "Abstract" Cohomology

In the intuitive guide, we said H¹ measures "inconsistency." But how do you actually
*compute* it? The answer is the **Čech complex**.

### The Setup: A Cover

Imagine you're mapping a city with drones. Each drone covers a region:

```
    Drone A        Drone B        Drone C
    ┌─────┐       ┌─────┐       ┌─────┐
    │     │       │     │       │     │
    │  A  │───────│  B  │───────│  C  │
    │     │       │     │       │     │
    └─────┘       └─────┘       └─────┘
         └───┬───┘     └───┬───┘
           A∩B           B∩C
```

The regions overlap. In the overlaps, both drones see the same area.

### The Čech Complex

We organize this into a **chain** of data:

```
C⁰: Data on each region
    {A: data_A, B: data_B, C: data_C}

C¹: Data on pairwise overlaps  
    {A∩B: data_AB, B∩C: data_BC}

C²: Data on triple overlaps
    {A∩B∩C: data_ABC}  (if all three overlap)
```

### The Coboundary Map δ

The magic is in the **coboundary map** δ: Cⁿ → Cⁿ⁺¹.

**δ⁰: C⁰ → C¹** measures disagreement on overlaps:

```
(δ⁰s)_{AB} = s_B|_{A∩B} - s_A|_{A∩B}
```

In plain English: "What B sees in the overlap, minus what A sees."

If δ⁰s = 0, then A and B agree everywhere they overlap.

**δ¹: C¹ → C²** checks the cocycle condition:

```
(δ¹t)_{ABC} = t_{BC} - t_{AC} + t_{AB}
```

This is the "going around a loop" check. If you go A→B→C→A, do you end up
where you started?

### Computing Cohomology

Now we can define:

- **H⁰ = ker(δ⁰)**: Data that's consistent on all overlaps
- **H¹ = ker(δ¹) / im(δ⁰)**: Inconsistencies that can't be "fixed"

The dimension of H¹ tells us *how many independent inconsistencies* exist.

### Real Example: Temperature Sensors

```python
from modalsheaf import compute_cech_cohomology
import numpy as np

# Three temperature sensors
data = {
    'kitchen': np.array([72.0]),
    'living_room': np.array([71.5]),
    'bedroom': np.array([70.0]),
}

result = compute_cech_cohomology(data)
print(result.summary())

# Output:
# Čech Cohomology Result
# ========================================
# Cover size: 3
# H⁰ = H0 ≅ ℝ^1
# H¹ = H1 ≅ ℝ^1
# 
# ✗ Inconsistency detected (dim H¹ = 1)
# 
# Conflict locations:
#   kitchen ∩ bedroom: [2.0]
```

The H¹ = ℝ¹ tells us there's one "dimension" of inconsistency. The conflict
is between kitchen (72°) and bedroom (70°) — a 2° difference.

---

## Part 2: Persistent Cohomology — Handling Noisy Data

### The Problem with Exact Equality

Real sensors aren't perfect. A 0.1° difference might be noise, but a 5° 
difference is probably real. How do we distinguish?

### The Idea: Vary the Threshold

Instead of asking "is the data consistent?", we ask:
"**At what tolerance does it become consistent?**"

```
Tolerance ε = 0.0:  Every tiny difference is "inconsistent"
Tolerance ε = 0.5:  Differences < 0.5 are ignored
Tolerance ε = 2.0:  Differences < 2.0 are ignored
Tolerance ε = ∞:    Everything is "consistent"
```

### Persistence Diagrams

We track when features are "born" and when they "die":

```
                    death
                      ^
                      |
                  5.0 +           * (significant!)
                      |
                  2.0 +       *
                      |     *
                  0.5 +   *   * (noise)
                      | * *
                  0.0 +---+---+---+---> birth
                      0   0.5 2.0 5.0
```

Points far from the diagonal (high persistence) are **real features**.
Points near the diagonal (low persistence) are **noise**.

### Real Example: Noisy Sensor Readings

```python
from modalsheaf import compute_persistent_cohomology
import numpy as np

# Sensors with noise
data = {
    'sensor_a': np.array([20.0]),   # Accurate
    'sensor_b': np.array([20.1]),   # Small noise
    'sensor_c': np.array([20.05]),  # Small noise
    'sensor_d': np.array([22.5]),   # Miscalibrated!
}

result = compute_persistent_cohomology(data)
print(result.summary())

# Output:
# Persistent Cohomology Result
# ========================================
# Thresholds: 50 values from 0.000 to 3.000
# 
# H1 Persistence Diagram
#   3 intervals
#   Max persistence: 2.450
#   Total persistence: 2.600
#   Top intervals:
#     H1[0.000, 2.450) pers=2.450  ← sensor_d is the problem!
#     H1[0.000, 0.100) pers=0.100  ← noise
#     H1[0.000, 0.050) pers=0.050  ← noise
# 
# Recommended threshold: 0.150
#   Consistent at this threshold: False
```

The high-persistence interval (2.45) tells us sensor_d is miscalibrated.
The low-persistence intervals (0.1, 0.05) are just measurement noise.

### Practical Application: CLIP Embedding Matching

```python
from modalsheaf import persistence_based_consistency
import numpy as np

# Image and caption embeddings from CLIP
image_emb = np.array([0.1, 0.2, 0.3, ...])  # 768-dim
text_emb = np.array([0.11, 0.19, 0.31, ...])  # Similar

is_consistent, confidence, explanation = persistence_based_consistency(
    {'image': image_emb, 'text': text_emb},
    noise_threshold=0.1  # Typical CLIP noise level
)

print(f"Match: {is_consistent} (confidence: {confidence:.1%})")
print(explanation)

# Output:
# Match: True (confidence: 92.3%)
# All inconsistencies below noise threshold (0.078 < 0.1)
```

---

## Part 3: The Cocycle Condition — Why Loops Matter

### The Problem: Pairwise Isn't Enough

Suppose three sensors each agree pairwise:
- A and B agree ✓
- B and C agree ✓
- C and A agree ✓

Does that mean they're all consistent? **Not necessarily!**

### The Currency Exchange Example

```
USD → EUR: ×0.85
EUR → GBP: ×0.86  
GBP → USD: ×1.37

Going around: 0.85 × 0.86 × 1.37 = 1.001
```

If this product ≠ 1.0, there's an arbitrage opportunity!

The **cocycle condition** says: going around any loop must return you
to where you started.

### The Math

For three overlapping regions A, B, C with transition functions g:

```
g_AB ∘ g_BC ∘ g_CA = identity
```

Or equivalently:

```
g_AB ∘ g_BC = g_AC
```

### Real Example: Sensor Calibration

```python
from modalsheaf import check_cocycle, repair_cocycle
import numpy as np

# Rotation matrices between sensor frames
# (In reality, these come from calibration)
R_cam_lidar = np.array([
    [0.9998, -0.0175, 0.0],
    [0.0175, 0.9998, 0.0],
    [0.0, 0.0, 1.0]
])

R_lidar_radar = np.array([
    [0.9998, 0.0, 0.0175],
    [0.0, 1.0, 0.0],
    [-0.0175, 0.0, 0.9998]
])

R_radar_cam = np.array([
    [0.9996, 0.0175, -0.0175],
    [-0.0175, 0.9998, 0.0],
    [0.0175, 0.0, 0.9998]
])

# Check cocycle condition
result = check_cocycle({
    ('camera', 'lidar'): R_cam_lidar,
    ('lidar', 'radar'): R_lidar_radar,
    ('radar', 'camera'): R_radar_cam,
})

print(result.summary())

# Output:
# ✗ Cocycle condition VIOLATED
#   1 violations in 1 triples
#   Max error: 0.0006
#   Total error: 0.0006
# 
#   Violations:
#     CocycleViolation(camera → lidar → radar, error=0.0006)
#       Matrix product deviates from identity by 0.0006
```

The error is small (0.0006), suggesting minor calibration drift.

### Repairing Cocycle Violations

```python
# Fix the calibration
fixed = repair_cocycle({
    ('camera', 'lidar'): R_cam_lidar,
    ('lidar', 'radar'): R_lidar_radar,
    ('radar', 'camera'): R_radar_cam,
}, method='average')

# Verify
result2 = check_cocycle(fixed)
print(f"Error after repair: {result2.max_error:.6f}")

# Output:
# Error after repair: 0.000001
```

---

## Part 4: Putting It All Together

### The Full Pipeline

For robust multimodal consistency checking:

```python
from modalsheaf import (
    ModalityGraph,
    compute_persistent_cohomology,
    check_cocycle,
    persistence_based_consistency,
)

# 1. Define your modality graph
graph = ModalityGraph()
graph.add_modality("image", shape=(768,))
graph.add_modality("text", shape=(768,))
graph.add_modality("audio", shape=(768,))

# 2. Get embeddings from your encoders
embeddings = {
    'image': image_encoder(image),
    'text': text_encoder(caption),
    'audio': audio_encoder(audio),
}

# 3. Check consistency with noise tolerance
is_consistent, confidence, explanation = persistence_based_consistency(
    embeddings,
    noise_threshold=0.15  # Adjust based on your encoders
)

print(f"Consistent: {is_consistent}")
print(f"Confidence: {confidence:.1%}")
print(explanation)

# 4. If inconsistent, find the culprit
if not is_consistent:
    result = compute_persistent_cohomology(embeddings)
    h1 = result.get_diagram(1)
    
    # The highest-persistence interval indicates the main conflict
    if h1.intervals:
        worst = max(h1.intervals, key=lambda i: i.persistence)
        print(f"Main conflict: persistence = {worst.persistence:.3f}")
```

### When to Use Each Tool

| Situation | Tool | Why |
|-----------|------|-----|
| Quick consistency check | `ConsistencyChecker` | Fast, simple |
| Noisy data | `compute_persistent_cohomology` | Filters noise |
| Rigorous analysis | `compute_cech_cohomology` | Full math |
| Calibration checking | `check_cocycle` | Verifies transforms |
| Fixing calibration | `repair_cocycle` | Corrects transforms |

---

## Part 5: The Bigger Picture

### Why This Matters for AI

1. **Hallucination Detection**: When an LLM says something inconsistent with
   the image, H¹ ≠ 0. Persistence tells us how bad the hallucination is.

2. **Sensor Fusion**: Autonomous vehicles fuse camera, LiDAR, radar. Cocycle
   violations indicate calibration drift or sensor failure.

3. **Knowledge Graphs**: When merging knowledge from multiple sources,
   cohomology detects contradictions.

4. **Distributed Systems**: In distributed databases, H¹ measures how
   "out of sync" replicas are.

### The Mathematical Insight

All of these are instances of the same abstract pattern:

> **Local data + Overlap constraints → Global consistency question**

Sheaf cohomology provides a *universal* framework for this pattern.
The specific domain (images, sensors, databases) just determines what
"local data" and "overlap" mean.

---

## Summary

| Concept | Intuition | Computation |
|---------|-----------|-------------|
| **Čech complex** | Organize data by overlaps | `CechComplex` |
| **Coboundary δ** | Measure disagreement | `complex.coboundary()` |
| **H⁰** | Global consensus | `ker(δ⁰)` |
| **H¹** | Obstructions | `ker(δ¹) / im(δ⁰)` |
| **Persistence** | Filter noise | `compute_persistent_cohomology()` |
| **Cocycle** | Loop consistency | `check_cocycle()` |

---

## References

1. **Computational**: Edelsbrunner & Harer (2010). *Computational Topology*.
2. **Applied**: Robinson (2014). *Topological Signal Processing*.
3. **Theoretical**: Bott & Tu (1982). *Differential Forms in Algebraic Topology*.
4. **Software**: GUDHI library for persistent homology.
