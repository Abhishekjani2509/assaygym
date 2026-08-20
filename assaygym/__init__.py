"""AssayGym — a synthetic RL environment / eval for wet-lab experimental design.

The biology is a deliberate caricature over invented loci (``SYN01``, ``SYN02``,
...). It tests experimental reasoning under noise and budget, not biological
realism.
"""

from assaygym.world import (
    HIT_THRESHOLD,
    TIERS,
    Difficulty,
    World,
    override_phenotype_from_deltas,
    sample_world,
)

__all__ = [
    "HIT_THRESHOLD",
    "TIERS",
    "Difficulty",
    "World",
    "override_phenotype_from_deltas",
    "sample_world",
]

__version__ = "0.1.0"
