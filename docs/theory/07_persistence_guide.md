# Persistent Cohomology: A Practical Guide

> *"Not everything that counts can be counted, and not everything that can be counted counts."* — William Bruce Cameron

Real-world data is messy. Sensors have noise. Embeddings have variance. 
Persistent cohomology helps us separate **signal from noise**.

---

## The Problem: When is "Different" Really Different?

Consider three temperature sensors:

```
Sensor A: 20.0°C
Sensor B: 20.1°C  
Sensor C: 20.05°C
```

Are they consistent? It depends on what you mean by "consistent":

- **Exact equality**: No! They're all different.
- **Within 0.5°C**: Yes! They all agree.
- **Within 0.01°C**: No! A and B differ by 0.1°C.

The answer depends on your **tolerance threshold**.

---

## The Solution: Vary the Threshold

Instead of picking one threshold, we look at **all thresholds**:

```
ε = 0.00:  A ≠ B ≠ C  (3 clusters, H⁰ = 3)
ε = 0.05:  A ≠ B, C joins A  (2 clusters)
ε = 0.10:  A = B = C  (1 cluster, H⁰ = 1)
```

We track how the topology **changes** as ε increases.

### Birth and Death

Each topological feature has a **lifetime**:

- **Birth**: The threshold where the feature first appears
- **Death**: The threshold where the feature disappears
- **Persistence** = Death - Birth

Features with **high persistence** are real.
Features with **low persistence** are noise.

---

## Persistence Diagrams

We visualize this with a **persistence diagram**:

```
        death
          ^
          |
      1.0 +               * (real feature)
          |
      0.5 +           *
          |       *
      0.2 +   * *     (noise)
          | * *
      0.0 +---+---+---+----> birth
          0  0.2 0.5 1.0
```

- **Diagonal** (y = x): Zero persistence (instant birth/death)
- **Far from diagonal**: High persistence (significant)
- **Near diagonal**: Low persistence (noise)

### Reading the Diagram

```python
from modalsheaf import compute_persistent_cohomology
import numpy as np

data = {
    'sensor_a': np.array([20.0]),
    'sensor_b': np.array([20.1]),
    'sensor_c': np.array([20.05]),
    'sensor_d': np.array([25.0]),  # Outlier!
}

result = compute_persistent_cohomology(data)
h1_diagram = result.get_diagram(1)

print("H¹ Persistence Intervals:")
for interval in sorted(h1_diagram.intervals, key=lambda i: -i.persistence):
    print(f"  [{interval.birth:.2f}, {interval.death:.2f}) "
          f"persistence = {interval.persistence:.2f}")
```

**Output:**
```
H¹ Persistence Intervals:
  [0.00, 4.95) persistence = 4.95  ← sensor_d is way off!
  [0.00, 0.10) persistence = 0.10  ← small noise
  [0.00, 0.05) persistence = 0.05  ← small noise
```

The 4.95 persistence interval tells us sensor_d is **significantly different**.
The 0.1 and 0.05 intervals are just measurement noise.

---

## Real-World Example 1: CLIP Embedding Matching

### The Problem

You have an image and a caption. CLIP gives you embeddings:

```python
image_emb = clip.encode_image(image)  # 768-dim vector
text_emb = clip.encode_text(caption)   # 768-dim vector
```

How do you know if they "match"?

### Naive Approach

```python
distance = np.linalg.norm(image_emb - text_emb)
if distance < 0.5:  # Magic threshold!
    print("Match!")
```

But where does 0.5 come from? What if your images are different from the training set?

### Persistence Approach

```python
from modalsheaf import persistence_based_consistency

# Check with automatic noise filtering
is_match, confidence, explanation = persistence_based_consistency(
    {'image': image_emb, 'text': text_emb},
    noise_threshold=0.15  # Based on typical CLIP variance
)

print(f"Match: {is_match}")
print(f"Confidence: {confidence:.1%}")
print(explanation)
```

**Good match:**
```
Match: True
Confidence: 94.2%
All inconsistencies below noise threshold (0.089 < 0.15)
```

**Bad match:**
```
Match: False  
Confidence: 12.3%
Significant inconsistency detected (persistence = 1.217)
```

### Why This Works

- The **persistence** of the H¹ feature tells us how "real" the mismatch is
- Low persistence = within normal CLIP variance = good match
- High persistence = semantic difference = bad match

---

## Real-World Example 2: Multi-Sensor Fusion

### The Problem

A robot has three sensors measuring distance to a wall:

```
Ultrasonic: 2.05 m
LiDAR:      2.00 m
Infrared:   2.10 m
```

Which reading should we trust?

### Persistence Analysis

```python
from modalsheaf import compute_persistent_cohomology
import numpy as np

readings = {
    'ultrasonic': np.array([2.05]),
    'lidar': np.array([2.00]),
    'infrared': np.array([2.10]),
}

result = compute_persistent_cohomology(readings)

# Find the recommended threshold
threshold = result.recommended_threshold(max_noise=0.05)
print(f"Recommended tolerance: {threshold:.3f} m")

# Check consistency at that threshold
is_consistent = result.consistency_at_threshold(threshold)
print(f"Sensors consistent: {is_consistent}")
```

**Output:**
```
Recommended tolerance: 0.050 m
Sensors consistent: False

H¹ intervals:
  [0.00, 0.10) persistence = 0.10  ← ultrasonic-infrared gap
  [0.00, 0.05) persistence = 0.05  ← lidar-ultrasonic gap
```

### Interpretation

- The 0.10m persistence (ultrasonic vs infrared) is above our noise threshold
- This suggests a real discrepancy, not just noise
- We might want to investigate the infrared sensor

### Weighted Fusion

```python
# Use persistence to weight the sensors
# Lower persistence with others = more trustworthy
def sensor_weight(sensor_name, diagram):
    # Sum of persistences involving this sensor
    total_pers = sum(
        i.persistence for i in diagram.intervals
        if sensor_name in str(i.representative)
    )
    return 1.0 / (1.0 + total_pers)

weights = {name: sensor_weight(name, h1_diagram) for name in readings}
# Normalize
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}

# Weighted average
fused = sum(readings[k][0] * weights[k] for k in readings)
print(f"Fused reading: {fused:.3f} m")
```

---

## Real-World Example 3: Anomaly Detection

### The Problem

You're monitoring a data center. Each server reports CPU temperature:

```python
temps = {
    'server_01': np.array([45.2]),
    'server_02': np.array([44.8]),
    'server_03': np.array([45.5]),
    'server_04': np.array([46.1]),
    'server_05': np.array([72.3]),  # Overheating!
    'server_06': np.array([45.0]),
}
```

### Finding the Anomaly

```python
from modalsheaf import compute_persistent_cohomology

result = compute_persistent_cohomology(temps)
h1 = result.get_diagram(1)

# Find high-persistence features
anomalies = h1.filter_by_persistence(min_persistence=5.0)

print(f"Found {len(anomalies.intervals)} anomalies:")
for interval in anomalies.intervals:
    print(f"  Persistence: {interval.persistence:.1f}")
```

**Output:**
```
Found 1 anomalies:
  Persistence: 26.2
```

The huge persistence (26.2°C) immediately flags server_05.

### Automatic Threshold Selection

```python
# What tolerance makes the "normal" servers consistent?
threshold = result.recommended_threshold(max_noise=2.0)
print(f"Normal variance threshold: {threshold:.1f}°C")

# Which servers are outside this?
for name, temp in temps.items():
    # Check if this server is "far" from the consensus
    others = {k: v for k, v in temps.items() if k != name}
    distances = [abs(temp[0] - v[0]) for v in others.values()]
    if max(distances) > threshold + 5:  # Significant outlier
        print(f"  ALERT: {name} at {temp[0]}°C")
```

---

## Real-World Example 4: Time Series Consistency

### The Problem

You have stock prices from three data providers:

```python
import numpy as np

# Price of AAPL at different times (simplified)
provider_a = np.array([150.0, 151.2, 149.8, 152.0, 151.5])
provider_b = np.array([150.1, 151.0, 149.9, 152.1, 151.4])
provider_c = np.array([150.0, 151.1, 155.0, 152.0, 151.6])  # Spike at t=2!
```

### Detecting the Discrepancy

```python
from modalsheaf import compute_persistent_cohomology

# Check each time point
for t in range(5):
    data = {
        'provider_a': np.array([provider_a[t]]),
        'provider_b': np.array([provider_b[t]]),
        'provider_c': np.array([provider_c[t]]),
    }
    
    result = compute_persistent_cohomology(data)
    max_pers = result.get_diagram(1).max_persistence()
    
    if max_pers > 1.0:  # Threshold for "significant"
        print(f"t={t}: DISCREPANCY detected (persistence={max_pers:.1f})")
    else:
        print(f"t={t}: consistent (persistence={max_pers:.2f})")
```

**Output:**
```
t=0: consistent (persistence=0.10)
t=1: consistent (persistence=0.20)
t=2: DISCREPANCY detected (persistence=5.1)
t=3: consistent (persistence=0.10)
t=4: consistent (persistence=0.20)
```

At t=2, provider_c reported 155.0 while others reported ~149.9. 
The persistence of 5.1 flags this as a real discrepancy, not noise.

---

## The Mathematics (Briefly)

### Filtration

A **filtration** is a nested sequence of spaces:

```
∅ ⊆ X₀ ⊆ X₁ ⊆ X₂ ⊆ ... ⊆ X
```

In our case, Xₑ contains all pairs with distance ≤ ε.

### Persistent Homology/Cohomology

We compute H*(Xₑ) for each ε and track how classes appear/disappear.

```
ε = 0.0:  H¹ = 0 (no edges yet)
ε = 0.1:  H¹ = ℝ² (two independent inconsistencies)
ε = 0.5:  H¹ = ℝ¹ (one inconsistency remains)
ε = 1.0:  H¹ = 0 (all consistent)
```

### Stability Theorem

Small perturbations in data cause small changes in persistence diagrams.
This is why persistence is **robust to noise**.

---

## API Reference

### Quick Functions

```python
from modalsheaf import (
    compute_persistent_cohomology,
    persistence_based_consistency,
)

# Full computation
result = compute_persistent_cohomology(data, num_thresholds=50)

# Quick consistency check
is_consistent, confidence, explanation = persistence_based_consistency(
    data, 
    noise_threshold=0.1
)
```

### Working with Results

```python
result = compute_persistent_cohomology(data)

# Get persistence diagram for H¹
h1 = result.get_diagram(1)

# Filter by persistence
significant = h1.filter_by_persistence(min_persistence=0.5)

# Get statistics
print(f"Max persistence: {h1.max_persistence()}")
print(f"Total persistence: {h1.total_persistence()}")
print(f"Significant features: {h1.num_significant(threshold=0.5)}")

# Recommended threshold
threshold = result.recommended_threshold(max_noise=0.1)

# Check consistency at threshold
is_consistent = result.consistency_at_threshold(threshold)
```

### Custom Distance Functions

```python
from modalsheaf import PersistentCohomology

# Cosine distance for embeddings
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

pc = PersistentCohomology(distance_fn=cosine_distance)
result = pc.compute(embeddings)
```

---

## Best Practices

### 1. Choose Appropriate Noise Thresholds

| Data Type | Typical Noise | Suggested Threshold |
|-----------|---------------|---------------------|
| Temperature sensors | 0.1-0.5°C | 0.5°C |
| CLIP embeddings | 0.05-0.15 | 0.15 |
| GPS coordinates | 1-5 meters | 5 meters |
| Stock prices | 0.01-0.1% | 0.1% |

### 2. Use Domain Knowledge

```python
# Bad: arbitrary threshold
result = persistence_based_consistency(data, noise_threshold=0.5)

# Good: threshold based on sensor specs
SENSOR_ACCURACY = 0.1  # From datasheet
result = persistence_based_consistency(data, noise_threshold=SENSOR_ACCURACY * 2)
```

### 3. Visualize When Debugging

```python
import matplotlib.pyplot as plt

h1 = result.get_diagram(1)
points = h1.to_array()

plt.figure(figsize=(8, 8))
plt.scatter(points[:, 0], points[:, 1], s=50)
plt.plot([0, max(points.max(), 1)], [0, max(points.max(), 1)], 'k--', alpha=0.3)
plt.xlabel('Birth')
plt.ylabel('Death')
plt.title('H¹ Persistence Diagram')
plt.show()
```

### 4. Monitor Persistence Over Time

```python
# Track persistence as a health metric
persistence_history = []

for batch in data_stream:
    result = compute_persistent_cohomology(batch)
    max_pers = result.get_diagram(1).max_persistence()
    persistence_history.append(max_pers)
    
    if max_pers > ALERT_THRESHOLD:
        send_alert(f"High inconsistency detected: {max_pers}")
```

---

## Summary

| Concept | Meaning | Use Case |
|---------|---------|----------|
| **Persistence** | How "real" a feature is | Distinguish signal from noise |
| **Birth** | When feature appears | Find threshold where inconsistency starts |
| **Death** | When feature disappears | Find threshold where data becomes consistent |
| **Diagram** | All features visualized | Overview of data quality |
| **Recommended threshold** | Automatic noise filtering | Production systems |

---

## References

1. **Foundational**: Edelsbrunner, Letscher, Zomorodian (2002). "Topological Persistence and Simplification."
2. **Practical**: Carlsson (2009). "Topology and Data." *Bull. AMS*.
3. **Software**: GUDHI, Ripser, giotto-tda libraries.
4. **Applications**: Perea (2018). "A Brief History of Persistence."
