"""AssayGym Phase 2 — the dirty window.

The agent never sees the truth in ``world.py``, only measurements. This module
is what happens in between: six artifacts that stand between a condition's
noise-free value and the number that lands in the agent's hands.

**Each artifact exists to punish one specific missing experimental skill.** The
comments on each step in :func:`run_plate` name which. That mapping is the whole
point of the file — an artifact that punishes nothing is just noise, and noise
alone would make the task harder without making it more diagnostic.

The order of application is load-bearing and is not free to rearrange. Lot
potency scales the *effect above baseline* rather than the raw observed value,
so a degraded lot shrinks the assay window instead of translating the whole
plate. Pipetting error is multiplicative and lands *before* the additive batch
shift, so a per-plate offset stays separable from per-well scatter. Getting
either wrong leaves the observations superficially plausible while quietly
destroying the signal an agent needs to detect the problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from assaygym.world import World

__all__ = [
    "ROWS",
    "COLS",
    "WELLS",
    "is_edge",
    "quadrant",
    "PlateResult",
    "run_plate",
    "z_prime",
]


ROWS = "ABCDEFGH"
COLS = list(range(1, 13))
WELLS = [f"{r}{c}" for r in ROWS for c in COLS]
_WELL_SET = frozenset(WELLS)

# Contamination offset. Large relative to a real effect (HIT_THRESHOLD = 0.20),
# so a blown quadrant masquerades as a strong hit unless it is detected.
CONTAMINATION_MEAN = 0.45
CONTAMINATION_SD = 0.15


def _split_well(well: str) -> tuple[int, int]:
    """``"C7"`` -> ``(2, 6)``: zero-based (row index, column index)."""
    if not isinstance(well, str) or len(well) < 2:
        raise ValueError(f"malformed well {well!r}")
    row, col = well[0], well[1:]
    if row not in ROWS or not col.isdigit():
        raise ValueError(f"malformed well {well!r}")
    col_num = int(col)
    if col_num not in COLS:
        raise ValueError(f"column out of range in well {well!r}")
    return ROWS.index(row), col_num - 1


def is_edge(well: str) -> bool:
    """True for perimeter wells — row A/H or column 1/12."""
    row_i, col_i = _split_well(well)
    return row_i in (0, len(ROWS) - 1) or col_i in (0, len(COLS) - 1)


def quadrant(well: str) -> int:
    """Quadrant 0-3: top-left, top-right, bottom-left, bottom-right."""
    row_i, col_i = _split_well(well)
    return (0 if row_i < 4 else 2) + (0 if col_i < 6 else 1)


@dataclass
class PlateResult:
    """One plate's worth of measurements, as handed back to the agent."""

    plate_id: str
    lot: str
    layout: Dict[str, str]
    values: Dict[str, float]
    day_run: int
    cost_usd: float
    excluded: bool = False
    notes: List[str] = field(default_factory=list)


def run_plate(
    world: World,
    plate_id: str,
    layout: Mapping[str, str],
    lot: str,
    rng: np.random.Generator,
    day: int = 0,
    cost: float = 0.0,
) -> PlateResult:
    """Run ``layout`` on one plate and return the observed values.

    ``layout`` maps well -> condition string (``"NTC"``, ``"POS"``,
    ``"KD:SYN03"``, ``"CMPD@100"``). ``lot`` names a reagent lot in
    ``world.lot_potency``. All measurement randomness comes from ``rng``, which
    the environment keeps separate from the world-generation rng so that noise
    and ground truth are independent.
    """
    if world.condition_value is None:
        raise ValueError(
            "world has no condition_value; call override_phenotype_from_deltas first"
        )
    if lot not in world.lot_potency:
        raise ValueError(f"unknown lot {lot!r}; expected one of {sorted(world.lot_potency)}")
    for well in layout:
        if well not in _WELL_SET:
            raise ValueError(f"well {well!r} is not on a 96-well plate")

    diff = world.diff
    base = world.baseline_phenotype
    lot_potency = world.lot_potency[lot]

    # --- drawn ONCE per plate, not per well --------------------------------
    # A per-well batch shift would average out across the plate and be
    # invisible; drawn once, it displaces the whole plate and can only be
    # removed by normalising against a control carried on that same plate.
    batch_shift = float(rng.normal(0.0, diff.batch_sigma))

    contaminated_quadrant: Optional[int] = None
    if rng.random() < diff.p_contamination:
        contaminated_quadrant = int(rng.integers(0, 4))
        world.contaminated_plates[plate_id] = contaminated_quadrant

    values: Dict[str, float] = {}
    for well, condition in layout.items():
        truth = world.condition_value(condition)
        effect = truth - base

        # 1. Reagent lot — multiplies the EFFECT above baseline, never the raw
        #    value, so a degraded lot flattens the plate toward the noise floor
        #    while leaving the baseline where it was. The only tell is a
        #    collapsed positive-control window.
        #    Punishes: never running positive controls.
        obs = base + effect * lot_potency

        # 2. Pipetting error — multiplicative, and applied BEFORE the additive
        #    shifts below so volume error scales the signal rather than
        #    offsetting it.
        #    Punishes: trusting a single well instead of replicating.
        obs *= 1.0 + rng.normal(0.0, diff.pipet_cv)

        # 3. Batch shift — one additive offset for the entire plate.
        #    Punishes: comparing raw values across plates without normalising
        #    each plate against its own on-plate negative control.
        obs += batch_shift

        # 4. Edge bias — evaporation on the perimeter, a fixed additive offset.
        #    Punishes: filling the plate from A1 and letting the layout decide
        #    which conditions land on the edge.
        if is_edge(well):
            obs += diff.edge_bias

        # 5. Contamination — one blown quadrant, offset far above a real effect.
        #    Punishes: bunching all controls in one corner, which leaves the
        #    agent unable to tell a blown quadrant from a genuine hit.
        if contaminated_quadrant is not None and quadrant(well) == contaminated_quadrant:
            obs += rng.normal(CONTAMINATION_MEAN, CONTAMINATION_SD)

        # 6. Measurement noise — plain additive read noise, applied last.
        #    Punishes: reading one well as truth; averaging replicates is the fix.
        obs += rng.normal(0.0, diff.well_noise)

        values[well] = float(obs)

    notes: List[str] = []
    if contaminated_quadrant is not None:
        # Recorded on the world (hidden), not surfaced to the agent.
        notes.append("plate flagged internally")

    return PlateResult(
        plate_id=plate_id,
        lot=lot,
        layout=dict(layout),
        values=values,
        day_run=day,
        cost_usd=cost,
        excluded=False,
        notes=notes,
    )


def z_prime(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Z-prime assay quality: ``1 - 3*(sd(pos) + sd(neg)) / |mean(pos) - mean(neg)|``.

    Uses the sample standard deviation. Returns ``nan`` with fewer than 2 of
    either control, and ``-inf`` when the assay window has collapsed to zero
    (the limit of the expression as the window shrinks).
    """
    pos_a = np.asarray(pos, dtype=float)
    neg_a = np.asarray(neg, dtype=float)
    if pos_a.size < 2 or neg_a.size < 2:
        return float("nan")

    window = abs(float(pos_a.mean()) - float(neg_a.mean()))
    if window == 0.0:
        return float("-inf")
    spread = float(pos_a.std(ddof=1)) + float(neg_a.std(ddof=1))
    return 1.0 - 3.0 * spread / window
