"""AssayGym — a synthetic RL environment / eval for wet-lab experimental design.

The biology is a deliberate caricature over invented loci (``SYN01``, ``SYN02``,
...). It tests experimental reasoning under noise and budget, not biological
realism.

Phase 1 (``world.py``) samples the hidden ground truth. Phase 2 (``assay.py``)
turns it into the noisy measurements an agent actually sees.
"""

from assaygym.assay import (
    COLS,
    ROWS,
    WELLS,
    PlateResult,
    is_edge,
    quadrant,
    run_plate,
    z_prime,
)
from assaygym.world import (
    HIT_THRESHOLD,
    TIERS,
    Difficulty,
    World,
    override_phenotype_from_deltas,
    sample_world,
)

__all__ = [
    # world (Phase 1)
    "HIT_THRESHOLD",
    "TIERS",
    "Difficulty",
    "World",
    "override_phenotype_from_deltas",
    "sample_world",
    # assay (Phase 2)
    "ROWS",
    "COLS",
    "WELLS",
    "PlateResult",
    "is_edge",
    "quadrant",
    "run_plate",
    "z_prime",
]

__version__ = "0.2.0"
