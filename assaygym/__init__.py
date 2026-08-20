"""AssayGym — a synthetic RL environment / eval for wet-lab experimental design.

The biology is a deliberate caricature over invented loci (``SYN01``, ``SYN02``,
...). It tests experimental reasoning under noise and budget, not biological
realism.

Phase 1 (``world.py``) samples the hidden ground truth. Phase 2 (``assay.py``)
turns it into the noisy measurements an agent actually sees. Phase 3
(``env.py``) is the lab bench the agent operates: budget, four tools, one
submission.
"""

from assaygym.env import (
    ASSAY_RNG_OFFSET,
    LOTS,
    PER_WELL_COST,
    PLATE_BASE_COST,
    PLATE_DAYS,
    PRIOR_CAVEAT,
    TOOL_SPEC,
    AssayGym,
    plate_cost,
)
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
    # env (Phase 3)
    "AssayGym",
    "TOOL_SPEC",
    "PRIOR_CAVEAT",
    "PLATE_BASE_COST",
    "PER_WELL_COST",
    "PLATE_DAYS",
    "LOTS",
    "ASSAY_RNG_OFFSET",
    "plate_cost",
]

__version__ = "0.3.0"
