"""AssayGym — a synthetic RL environment / eval for wet-lab experimental design.

The biology is a deliberate caricature over invented loci (``SYN01``, ``SYN02``,
...). It tests experimental reasoning under noise and budget, not biological
realism.

Phase 1 (``world.py``) samples the hidden ground truth. Phase 2 (``assay.py``)
turns it into the noisy measurements an agent actually sees. Phase 3
(``env.py``) is the lab bench the agent operates: budget, four tools, one
submission. Phase 4 (``rewards.py``) is the judge: three separate numbers plus
diagnostics that are reported but never scored. Phase 5 (``policies.py``) is
four scripted baselines whose ledger proves the reward separates competence
from noise.
"""

from assaygym.policies import (
    ABLATIONS,
    DOSES,
    INTERIOR,
    POLICIES,
    POLICY_RNG_OFFSET,
    call_everything_policy,
    competent_doe_policy,
    naive_screen_policy,
    prior_parrot_policy,
    random_policy,
    run_episode,
    run_policy,
)
from assaygym.rewards import (
    EC50_TOLERANCE,
    EFFICIENCY_GATE,
    ENDPOINT_WEIGHTS,
    SHAPED_WEIGHTS,
    diagnostics,
    endpoint_terms,
    score,
    score_trajectory,
    shaped_terms,
    strict_pass,
)
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
    # rewards (Phase 4)
    "score",
    "score_trajectory",
    "strict_pass",
    "endpoint_terms",
    "shaped_terms",
    "diagnostics",
    "ENDPOINT_WEIGHTS",
    "SHAPED_WEIGHTS",
    "EC50_TOLERANCE",
    "EFFICIENCY_GATE",
    # policies (Phase 5)
    "POLICIES",
    "ABLATIONS",
    "INTERIOR",
    "DOSES",
    "POLICY_RNG_OFFSET",
    "random_policy",
    "prior_parrot_policy",
    "naive_screen_policy",
    "competent_doe_policy",
    "call_everything_policy",
    "run_episode",
    "run_policy",
]

__version__ = "0.5.0"
