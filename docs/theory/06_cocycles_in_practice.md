# Cocycles in Practice: Real-World Applications

> *"The cocycle condition is the mathematician's way of saying 'what goes around, comes around.'"*

This document explores the cocycle condition through concrete, real-world examples.
We'll see how this abstract mathematical concept solves practical engineering problems.

---

## What is a Cocycle?

A **cocycle** is data that satisfies a consistency condition around loops.

Think of it like this: if you walk around a block and return to your starting point,
you should be at the same place. If you're not, something is wrong with your map.

### The General Pattern

Given three overlapping regions A, B, C with "transition functions" between them:

```
        g_AB
    A ───────→ B
     ↖         │
      \        │ g_BC
  g_CA \       │
        \      ↓
         ──── C
```

The **cocycle condition** states:

```
g_CA ∘ g_BC ∘ g_AB = identity
```

Going A → B → C → A should bring you back to exactly where you started.

---

## Example 1: Currency Arbitrage

### The Setup

You're a currency trader with access to three markets:

```
USD ←→ EUR ←→ GBP ←→ USD
```

Each market has an exchange rate:

```python
rates = {
    ('USD', 'EUR'): 0.85,   # 1 USD = 0.85 EUR
    ('EUR', 'GBP'): 0.86,   # 1 EUR = 0.86 GBP
    ('GBP', 'USD'): 1.38,   # 1 GBP = 1.38 USD
}
```

### The Cocycle Check

```python
from modalsheaf import check_cocycle
import numpy as np

# Transition functions are scalar multiplications
transitions = {
    ('USD', 'EUR'): np.array([[0.85]]),
    ('EUR', 'GBP'): np.array([[0.86]]),
    ('GBP', 'USD'): np.array([[1.38]]),
}

result = check_cocycle(transitions, tolerance=0.001)
print(result.summary())
```

**Output:**
```
✗ Cocycle condition VIOLATED
  1 violations in 1 triples
  Max error: 0.0087

  Violations:
    CocycleViolation(USD → EUR → GBP, error=0.0087)
      Matrix product deviates from identity by 0.0087
```

### Interpretation

The product is: 0.85 × 0.86 × 1.38 = **1.0087**

This means: Start with $1000 USD, convert around the loop, end with **$1008.70**.

That's an **arbitrage opportunity** of 0.87%! The cocycle violation reveals
a market inefficiency.

### In Practice

High-frequency trading algorithms constantly check for cocycle violations
across currency pairs. When found, they execute trades faster than the
market can correct itself.

---

## Example 2: Sensor Calibration (Robotics)

### The Setup

A self-driving car has three sensors:
- **Camera**: Sees in 2D, needs to know where it is relative to the car
- **LiDAR**: Sees in 3D, has its own coordinate frame
- **Radar**: Detects velocity, another coordinate frame

Each sensor is calibrated relative to the others:

```
Camera ←→ LiDAR ←→ Radar ←→ Camera
```

### The Problem

Over time, sensors drift. A bump might shift the LiDAR slightly.
How do we detect this?

### The Cocycle Check

```python
from modalsheaf import check_cocycle, repair_cocycle
import numpy as np

# Rotation matrices from calibration
# (Simplified to 2D for clarity)
R_cam_lidar = np.array([
    [0.9998, -0.0200],
    [0.0200,  0.9998]
])  # ~1.15° rotation

R_lidar_radar = np.array([
    [0.9994, -0.0350],
    [0.0350,  0.9994]
])  # ~2.0° rotation

R_radar_cam = np.array([
    [0.9988,  0.0500],  # Should be ~3.15° but is ~2.87°
    [-0.0500, 0.9988]
])  # MISCALIBRATED!

result = check_cocycle({
    ('camera', 'lidar'): R_cam_lidar,
    ('lidar', 'radar'): R_lidar_radar,
    ('radar', 'camera'): R_radar_cam,
})

print(result.summary())
```

**Output:**
```
✗ Cocycle condition VIOLATED
  1 violations in 1 triples
  Max error: 0.0098

  Violations:
    CocycleViolation(camera → lidar → radar, error=0.0098)
      Matrix product deviates from identity by 0.0098
```

### The Fix

```python
# Repair the calibration
fixed = repair_cocycle({
    ('camera', 'lidar'): R_cam_lidar,
    ('lidar', 'radar'): R_lidar_radar,
    ('radar', 'camera'): R_radar_cam,
}, method='average')

# Verify
result2 = check_cocycle(fixed)
print(f"Error after repair: {result2.max_error:.6f}")
# Output: Error after repair: 0.000003
```

### In Practice

Autonomous vehicles run cocycle checks continuously. A sudden increase in
cocycle error triggers recalibration or alerts the safety system.

---

## Example 3: Time Zone Consistency

### The Setup

A distributed system has servers in three cities:

```
New York ←→ London ←→ Tokyo ←→ New York
```

Each server knows the time offset to its neighbors:

```python
offsets = {
    ('NYC', 'London'): +5,   # London is 5 hours ahead
    ('London', 'Tokyo'): +9,  # Tokyo is 9 hours ahead of London
    ('Tokyo', 'NYC'): -14,    # NYC is 14 hours behind Tokyo
}
```

### The Cocycle Check

```python
# For additive groups, cocycle condition is: sum around loop = 0
total = 5 + 9 + (-14)
print(f"Cocycle sum: {total}")  # Should be 0
```

**Output:**
```
Cocycle sum: 0
```

✓ The time zones are consistent!

### What if They Weren't?

```python
# Suppose someone made an error
bad_offsets = {
    ('NYC', 'London'): +5,
    ('London', 'Tokyo'): +9,
    ('Tokyo', 'NYC'): -13,  # Wrong! Should be -14
}

total = 5 + 9 + (-13)
print(f"Cocycle sum: {total}")  # = 1, not 0!
```

This would cause:
- Scheduled events happening at wrong times
- Database timestamps being inconsistent
- Logs that don't make chronological sense

### In Practice

Distributed systems use protocols like NTP (Network Time Protocol) that
essentially enforce the cocycle condition across all server pairs.

---

## Example 4: Version Control (Git)

### The Setup

Three developers are working on branches:

```
main ←→ feature-A ←→ feature-B ←→ main
```

Each branch has changes (patches) relative to others:

```
P_main→A: Changes to go from main to feature-A
P_A→B:    Changes to go from feature-A to feature-B  
P_B→main: Changes to go from feature-B back to main
```

### The Cocycle Condition

```
P_B→main ∘ P_A→B ∘ P_main→A = identity (no changes)
```

If this fails, you have a **merge conflict**!

### Example

```python
# Simplified: each "patch" is a set of line changes
P_main_A = {"file.py": {10: "new line A"}}
P_A_B = {"file.py": {10: "new line B"}}  # Conflict!
P_B_main = {"file.py": {10: "original"}}

# Composing patches:
# main → A: line 10 becomes "new line A"
# A → B: line 10 becomes "new line B"
# B → main: line 10 becomes "original"

# Round trip: "original" → "new line A" → "new line B" → "original"
# This works! But only because B→main overwrites everything.

# The REAL cocycle violation is:
# P_main_A and P_A_B both modify line 10
# Git detects this as a merge conflict
```

### In Practice

Git's merge algorithm is essentially checking a cocycle condition:
- If patches commute (order doesn't matter), no conflict
- If patches don't commute, cocycle fails → merge conflict

---

## Example 5: Multi-Camera 3D Reconstruction

### The Setup

Three cameras observe a scene:

```
    Camera 1
       /\
      /  \
     /    \
Camera 2--Camera 3
```

Each pair of cameras has a **fundamental matrix** F that encodes their
geometric relationship.

### The Cocycle Condition

For three cameras with fundamental matrices:

```
F_12, F_23, F_31
```

The cocycle condition (in projective geometry) is:

```
F_31 · F_23 · F_12 ∝ identity (up to scale)
```

### Why It Matters

If the cocycle fails:
- 3D reconstruction will be inconsistent
- Points will appear in different places depending on which camera pair you use
- The resulting 3D model will be "twisted"

### Code Example

```python
from modalsheaf import check_cocycle
import numpy as np

# Fundamental matrices (simplified 3x3)
F_12 = np.array([
    [0, -0.1, 0.5],
    [0.1, 0, -0.3],
    [-0.5, 0.3, 0]
])

F_23 = np.array([
    [0, -0.2, 0.4],
    [0.2, 0, -0.1],
    [-0.4, 0.1, 0]
])

F_31 = np.array([
    [0, 0.15, -0.45],
    [-0.15, 0, 0.2],
    [0.45, -0.2, 0]
])

result = check_cocycle({
    ('cam1', 'cam2'): F_12,
    ('cam2', 'cam3'): F_23,
    ('cam3', 'cam1'): F_31,
})

if not result.is_satisfied:
    print("Camera calibration is inconsistent!")
    print(f"Error: {result.max_error:.4f}")
```

---

## Example 6: Knowledge Graph Consistency

### The Setup

You're merging knowledge from three sources:

```
Wikipedia ←→ Wikidata ←→ DBpedia ←→ Wikipedia
```

Each source has mappings to the others:

```
"Barack Obama" (Wikipedia) → Q76 (Wikidata)
Q76 (Wikidata) → dbr:Barack_Obama (DBpedia)
dbr:Barack_Obama (DBpedia) → "Barack Obama" (Wikipedia)
```

### The Cocycle Condition

Following the chain should return to the same entity:

```
Wikipedia → Wikidata → DBpedia → Wikipedia = identity
```

### When It Fails

```python
# Suppose there's an error in the DBpedia mapping
mappings = {
    ('wiki', 'wikidata'): {'Barack Obama': 'Q76'},
    ('wikidata', 'dbpedia'): {'Q76': 'dbr:Barack_Obama'},
    ('dbpedia', 'wiki'): {'dbr:Barack_Obama': 'Barack Hussein Obama'},  # Different!
}

# Cocycle check would reveal:
# "Barack Obama" → Q76 → dbr:Barack_Obama → "Barack Hussein Obama"
# ≠ "Barack Obama"
```

### In Practice

Knowledge graph alignment tools check cocycle conditions to ensure
entity mappings are consistent across sources.

---

## Summary: The Cocycle Pattern

| Domain | Regions | Transitions | Cocycle Meaning |
|--------|---------|-------------|-----------------|
| **Finance** | Currencies | Exchange rates | No arbitrage |
| **Robotics** | Sensors | Calibration matrices | Consistent localization |
| **Time** | Time zones | Hour offsets | Consistent scheduling |
| **Version Control** | Branches | Patches | No merge conflicts |
| **3D Vision** | Cameras | Fundamental matrices | Consistent reconstruction |
| **Knowledge** | Databases | Entity mappings | Consistent ontology |

### The Universal Insight

All these problems share the same structure:

1. **Local relationships** between pairs of things
2. **Global consistency** requirement around loops
3. **Cocycle condition** as the mathematical test

When the cocycle fails, something is wrong. The *size* of the failure
tells you *how wrong*.

---

## Using ModalSheaf for Cocycle Checking

```python
from modalsheaf import (
    check_cocycle,
    repair_cocycle,
    CocycleChecker,
    TransitionFunction,
)

# Quick check with matrices
result = check_cocycle({
    ('A', 'B'): matrix_AB,
    ('B', 'C'): matrix_BC,
    ('C', 'A'): matrix_CA,
})

# Quick check with functions
result = check_cocycle({
    ('A', 'B'): lambda x: transform_AB(x),
    ('B', 'C'): lambda x: transform_BC(x),
    ('C', 'A'): lambda x: transform_CA(x),
})

# Repair violations
fixed = repair_cocycle(transitions, method='average')

# Advanced: custom transition functions
checker = CocycleChecker({
    ('A', 'B'): TransitionFunction(
        source='A',
        target='B',
        forward=my_forward_fn,
        inverse=my_inverse_fn,
    ),
    ...
})
result = checker.check()
```

---

## References

1. **Fiber Bundles**: Husemöller (1994). *Fibre Bundles*. Chapter 3.
2. **Robotics**: Barfoot (2017). *State Estimation for Robotics*. Chapter 7.
3. **Computer Vision**: Hartley & Zisserman (2004). *Multiple View Geometry*.
4. **Knowledge Graphs**: Spivak (2014). *Category Theory for the Sciences*.
