"""
ModalSheaf Applications - Domain-specific sheaf implementations.

This package contains specialized sheaf structures for various domains:
- neuro: BrainSheaf for computational neuroscience / fMRI analysis
"""

from .neuro import (
    BrainSheaf,
    BrainRegion,
    DissonanceResult,
    PersistentCycleResult,
    load_fmri_data,
    load_connectivity_matrix,
)

__all__ = [
    "BrainSheaf",
    "BrainRegion",
    "DissonanceResult",
    "PersistentCycleResult",
    "load_fmri_data",
    "load_connectivity_matrix",
    # RL
    "HodgeCritic",
    "RewardSheaf",
    "CycleResult",
]
