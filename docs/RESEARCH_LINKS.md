# Research Links

This document connects `modalsheaf` development to ongoing research in the `ai_research` repository.

---

## 🔬 Related Research Topics

`modalsheaf` is the library of choice for implementing sheaf-theoretic ideas across these research topics:

| Topic | Location | Integration |
|-------|----------|-------------|
| **Visual Token Compression** | `ai_research/topics/visual_token_compression/` | Restriction maps for compression |
| **High-Dimensional Reward Spaces** | `ai_research/topics/high_dimensional_reward_spaces/` | `modalsheaf.applications.rl` |
| **Multimodality and Sheaves** | `ai_research/topics/multimodality_and_sheaves/` | Core theory documentation |

---

## 📦 Application Modules

### `modalsheaf.applications.rl`

Sheaf-theoretic reward spaces for reinforcement learning. Ported from `high_dimensional_reward_spaces` research.

**Key Components:**
- `HodgeCritic` — Neural network for Hodge decomposition (separates potential V from harmonic ω)
- `RewardSheaf` — Discrete graph analysis for detecting positive cycles (H¹ obstructions)
- `CycleResult` — Dataclass for cycle analysis results

**Example:**
```python
from modalsheaf.applications.rl import HodgeCritic, RewardSheaf

# Discrete cycle detection
sheaf = RewardSheaf()
sheaf.add_transition("A", "B", reward=1.0)
sheaf.add_transition("B", "C", reward=1.0)
sheaf.add_transition("C", "A", reward=1.0)
cycles = sheaf.find_positive_cycles()  # Detects H¹ obstruction

# Neural Hodge decomposition (requires torch)
critic = HodgeCritic(state_dim=2)
potential, harmonic = critic(state)  # V(s) and ω
```

**Run example:**
```bash
python examples/07_reward_sheaves.py
```

### `modalsheaf.applications.neuro`

Brain network analysis using sheaf theory.

**Key Components:**
- `BrainSheaf` — Sheaf over brain connectivity graph
- `DissonanceResult` — Coboundary analysis for detecting inconsistencies
- `PersistentCycleResult` — Persistent homology for H¹ cycles

---

## 🛠️ Environment Setup

```bash
# Create and activate a venv, then install in development mode
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 📚 Key Features

- `ModalityGraph` — Define modalities and transformations
- `ConsistencyChecker` — Compute H⁰, H¹ cohomology for consistency analysis
- `GluingProtocol` — Compose local data into global sections
- `DiagnosticAnalyzer` — Identify outliers, factions, and drift

---

*Last updated: January 2026*
