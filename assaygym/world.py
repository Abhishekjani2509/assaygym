"""AssayGym Phase 1 — the sealed envelope.

Everything in this module is hidden ground truth. The agent never observes any
of it directly; it only ever sees noisy measurements derived from it in
``assay.py``. Because the truth is written down before the agent exists,
grading is arithmetic rather than judgement.

All randomness in world generation flows from a single ``np.random.default_rng``
seeded per world, so ``sample_world(seed, tier)`` is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "HIT_THRESHOLD",
    "Difficulty",
    "TIERS",
    "World",
    "sample_world",
    "override_phenotype_from_deltas",
]


# Absolute, in readout units. Identical across every difficulty tier: what
# counts as a hit is a property of the assay, not of the tier.
HIT_THRESHOLD = 0.20

LOTS = ["LOT-A", "LOT-B", "LOT-C"]

# Effect-size structure, shared by every tier (see Difficulty docstring).
_HIT_MULT_RANGE = (1.5, 3.0)      # x HIT_THRESHOLD
_GRAY_MULT_RANGE = (0.60, 0.92)   # x HIT_THRESHOLD
_NULL_SD_MULT = 0.25              # x HIT_THRESHOLD

# Network / phenotype generation.
_P_EDGE = 0.28
_EDGE_WEIGHT_SD = 0.45
_BASELINE_RANGE = (0.7, 1.3)
_READOUT_WEIGHT_RANGE = (0.4, 1.0)
_KD_EFFICIENCY_RANGE = (0.65, 0.92)

# Compound.
_LOG_EC50_RANGE = (0.5, 3.5)      # log10 nM: 3 nM to ~3 uM, the full testable range
_HILL_RANGE = (0.9, 2.0)

# Reagent lots.
_LOT_POTENCY_MEAN = 1.0
_LOT_POTENCY_SD = 0.03
_BAD_LOT_POTENCY_RANGE = (0.30, 0.55)

# Positive control sits well above the largest real knockdown effect.
_POS_CONTROL_MULT = 1.6


@dataclass(frozen=True)
class Difficulty:
    """Per-tier knobs.

    Design rule: difficulty increases by adding **noise, traps and scarcity** —
    never by shrinking true effect sizes. The effect-size structure (hit
    magnitudes, gray-zone magnitudes, null spread, the hit threshold) is
    identical across ``clean``, ``standard`` and ``hard``; only ``well_noise``,
    ``pipet_cv``, ``batch_sigma``, ``edge_bias``, ``p_bad_lot``,
    ``p_contamination``, the prior-trap parameters and the budget change.

    The reason: a score gap between tiers must be a statement about the agent's
    experimental design, not about the signal having been quietly made
    undetectable. If hard-tier effects were smaller, a low hard-tier score would
    be ambiguous between "the agent designs badly" and "the task is impossible",
    and the tier ladder would measure nothing. With this rule the signal is
    always there to be found; what varies is how much design skill it takes to
    find it.

    Do not rebalance these numbers casually — they are calibrated to produce the
    Phase 5 baseline ladder.
    """

    name: str
    n_genes: int
    n_true_hits: int
    n_gray: int
    well_noise: float
    pipet_cv: float
    batch_sigma: float
    edge_bias: float
    p_bad_lot: float
    p_contamination: float
    p_prior_trap: float
    n_decoys: int
    n_omitted: int
    budget_usd: float
    budget_days: int


TIERS: Dict[str, Difficulty] = {
    "clean": Difficulty(
        name="clean",
        n_genes=8,
        n_true_hits=3,
        n_gray=0,
        well_noise=0.02,
        pipet_cv=0.02,
        batch_sigma=0.02,
        edge_bias=0.0,
        p_bad_lot=0.0,
        p_contamination=0.0,
        p_prior_trap=0.0,
        n_decoys=0,
        n_omitted=0,
        budget_usd=6000.0,
        budget_days=18,
    ),
    "standard": Difficulty(
        name="standard",
        n_genes=10,
        n_true_hits=3,
        n_gray=2,
        well_noise=0.07,
        pipet_cv=0.05,
        batch_sigma=0.12,
        edge_bias=0.10,
        p_bad_lot=0.35,
        p_contamination=0.25,
        p_prior_trap=0.70,
        n_decoys=2,
        n_omitted=1,
        budget_usd=4500.0,
        budget_days=12,
    ),
    "hard": Difficulty(
        name="hard",
        n_genes=12,
        n_true_hits=4,
        n_gray=3,
        well_noise=0.13,
        pipet_cv=0.09,
        batch_sigma=0.20,
        edge_bias=0.14,
        p_bad_lot=0.55,
        p_contamination=0.40,
        p_prior_trap=1.0,
        n_decoys=2,
        n_omitted=2,
        budget_usd=3300.0,
        budget_days=9,
    ),
}


@dataclass
class World:
    """A sampled ground truth. Never exposed to the agent."""

    seed: int
    diff: Difficulty
    genes: List[str]
    adj: np.ndarray
    baseline: np.ndarray
    readout_w: np.ndarray
    kd_efficiency: np.ndarray
    true_delta: np.ndarray
    true_hits: List[str]
    true_signs: Dict[str, int]
    gray_zone: List[str]
    hit_threshold: float
    compound_target: str
    true_log_ec50: float
    true_hill: float
    lot_potency: Dict[str, float]
    bad_lots: List[str]
    contaminated_plates: Dict[str, int] = field(default_factory=dict)
    reported_hits: List[str] = field(default_factory=list)
    decoys: List[str] = field(default_factory=list)
    omitted: List[str] = field(default_factory=list)

    # Installed by override_phenotype_from_deltas.
    baseline_phenotype: float = 0.0
    condition_value: Optional[Callable[[str], float]] = None

    def gene_index(self, gene: str) -> int:
        return self.genes.index(gene)


def _propagate(adj: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Single forward pass through the upper-triangular DAG.

    Index order is topological because ``adj`` is upper-triangular, so no cycle
    handling is needed.
    """
    n = baseline.shape[0]
    levels = baseline.astype(float).copy()
    for j in range(n):
        if j > 0:
            levels[j] += float(adj[:j, j] @ levels[:j])
    return levels


def sample_world(seed: int, tier: str = "standard") -> World:
    """Sample a hidden world for ``seed`` under difficulty ``tier``."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")
    diff = TIERS[tier]
    rng = np.random.default_rng(seed)

    n = diff.n_genes
    genes = [f"SYN{i + 1:02d}" for i in range(n)]

    # --- Step 1: the network -------------------------------------------------
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < _P_EDGE:
                adj[i, j] = rng.normal(0.0, _EDGE_WEIGHT_SD)

    baseline = rng.uniform(*_BASELINE_RANGE, size=n)

    n_readout = max(2, n // 4)
    readout_nodes = np.sort(rng.choice(n, size=n_readout, replace=False))
    readout_w = np.zeros(n, dtype=float)
    raw_w = rng.uniform(*_READOUT_WEIGHT_RANGE, size=n_readout)
    readout_w[readout_nodes] = raw_w / raw_w.sum()

    # Knockdowns are never total, which is true to life.
    kd_efficiency = rng.uniform(*_KD_EFFICIENCY_RANGE, size=n)

    # --- Step 2: impose the effect structure --------------------------------
    # Effects are imposed, not emergent, so the difficulty of the task is known
    # exactly rather than being an accident of the sampled network.
    perm = rng.permutation(n)
    hit_idx = perm[: diff.n_true_hits]
    gray_idx = perm[diff.n_true_hits : diff.n_true_hits + diff.n_gray]
    null_idx = perm[diff.n_true_hits + diff.n_gray :]

    true_delta = np.zeros(n, dtype=float)

    # Signs are drawn uniformly from {-1, +1}. This is load-bearing: with skewed
    # signs a policy that blindly guesses one direction for every knockdown
    # scores well on direction without running a single plate, and the
    # environment stops measuring experimental work. Randomising makes that
    # guess worth exactly 50%.
    if len(hit_idx) > 0:
        hit_signs = rng.choice(np.array([-1.0, 1.0]), size=len(hit_idx))
        hit_mag = rng.uniform(*_HIT_MULT_RANGE, size=len(hit_idx)) * HIT_THRESHOLD
        true_delta[hit_idx] = hit_signs * hit_mag

    # The gray zone: real but sub-threshold effects — what an underpowered study
    # would have called significant. These are what make a single well per
    # condition insufficient.
    if len(gray_idx) > 0:
        gray_signs = rng.choice(np.array([-1.0, 1.0]), size=len(gray_idx))
        gray_mag = rng.uniform(*_GRAY_MULT_RANGE, size=len(gray_idx)) * HIT_THRESHOLD
        true_delta[gray_idx] = gray_signs * gray_mag

    if len(null_idx) > 0:
        true_delta[null_idx] = rng.normal(
            0.0, _NULL_SD_MULT * HIT_THRESHOLD, size=len(null_idx)
        )

    true_hits = sorted(genes[i] for i in hit_idx)
    gray_zone = sorted(genes[i] for i in gray_idx)
    true_signs = {g: int(np.sign(true_delta[genes.index(g)])) for g in true_hits}

    # --- Step 3: the compound ------------------------------------------------
    compound_target = str(rng.choice(np.array(true_hits)))
    # Spans the entire testable dose range. A narrow range lets a fixed blind
    # guess land inside the scoring tolerance too often, which inflates the
    # do-nothing policy.
    true_log_ec50 = float(rng.uniform(*_LOG_EC50_RANGE))
    true_hill = float(rng.uniform(*_HILL_RANGE))

    # --- Step 4: reagent lots ------------------------------------------------
    lot_potency = {
        lot: float(rng.normal(_LOT_POTENCY_MEAN, _LOT_POTENCY_SD)) for lot in LOTS
    }
    bad_lots: List[str] = []
    if rng.random() < diff.p_bad_lot:
        bad = str(rng.choice(np.array(LOTS)))
        lot_potency[bad] = float(rng.uniform(*_BAD_LOT_POTENCY_RANGE))
        bad_lots.append(bad)

    # --- Step 5: the prior trap ---------------------------------------------
    decoys: List[str] = []
    omitted: List[str] = []
    nulls = sorted(genes[i] for i in null_idx)

    trap_possible = (
        n >= 4
        and len(true_hits) > diff.n_omitted  # never omit every genuine hit
        and (diff.n_decoys > 0 or diff.n_omitted > 0)
        and (len(gray_zone) + len(nulls)) >= diff.n_decoys
    )
    if rng.random() < diff.p_prior_trap and trap_possible:
        # Decoys come from the gray zone first, falling back to nulls only if
        # the gray zone is too small. A plausible false published result is not
        # a gene with an obviously zero effect; it is one sitting near 0.7x
        # threshold — exactly the kind of thing a real literature contains.
        n_from_gray = min(diff.n_decoys, len(gray_zone))
        if n_from_gray > 0:
            picked = rng.choice(
                np.array(gray_zone), size=n_from_gray, replace=False
            )
            decoys.extend(str(g) for g in picked)
        n_from_null = diff.n_decoys - n_from_gray
        if n_from_null > 0 and len(nulls) > 0:
            picked = rng.choice(
                np.array(nulls), size=min(n_from_null, len(nulls)), replace=False
            )
            decoys.extend(str(g) for g in picked)

        if diff.n_omitted > 0:
            n_omit = min(diff.n_omitted, len(true_hits) - 1)
            if n_omit > 0:
                picked = rng.choice(
                    np.array(true_hits), size=n_omit, replace=False
                )
                omitted.extend(str(g) for g in picked)

        decoys = sorted(set(decoys))
        omitted = sorted(set(omitted))
        reported_hits = sorted((set(true_hits) - set(omitted)) | set(decoys))
    else:
        reported_hits = sorted(true_hits)

    world = World(
        seed=seed,
        diff=diff,
        genes=genes,
        adj=adj,
        baseline=baseline,
        readout_w=readout_w,
        kd_efficiency=kd_efficiency,
        true_delta=true_delta,
        true_hits=true_hits,
        true_signs=true_signs,
        gray_zone=gray_zone,
        hit_threshold=HIT_THRESHOLD,
        compound_target=compound_target,
        true_log_ec50=true_log_ec50,
        true_hill=true_hill,
        lot_potency=lot_potency,
        bad_lots=bad_lots,
        contaminated_plates={},
        reported_hits=reported_hits,
        decoys=decoys,
        omitted=omitted,
    )
    return world


def override_phenotype_from_deltas(world: World) -> World:
    """Install the noise-free ``condition_value`` closure on ``world``.

    The network propagation of Step 1 exists to generate a plausible baseline
    readout value, but the deltas imposed in Step 2 are what must actually be
    measurable. So the phenotype of every condition is defined directly from
    ``true_delta`` on top of the propagated baseline.
    """
    levels = _propagate(world.adj, world.baseline)
    base = float(world.readout_w @ levels)
    world.baseline_phenotype = base

    max_abs_delta = float(np.max(np.abs(world.true_delta))) if world.true_delta.size else 0.0
    pos_value = base + _POS_CONTROL_MULT * max_abs_delta
    gene_index = {g: i for i, g in enumerate(world.genes)}
    target_delta = float(world.true_delta[gene_index[world.compound_target]])

    def condition_value(condition: str) -> float:
        if not isinstance(condition, str):
            raise ValueError(f"condition must be a string, got {condition!r}")
        cond = condition.strip()

        if cond == "NTC":
            return base
        if cond == "POS":
            return pos_value
        if cond.startswith("KD:"):
            gene = cond[3:]
            if gene not in gene_index:
                raise ValueError(f"unknown locus in condition {condition!r}")
            return base + float(world.true_delta[gene_index[gene]])
        if cond.startswith("CMPD@"):
            try:
                dose = float(cond[5:])
            except ValueError:
                raise ValueError(f"unparseable dose in condition {condition!r}") from None
            if dose <= 0:
                return base
            ec50 = 10.0 ** world.true_log_ec50
            occupancy = 1.0 / (1.0 + (ec50 / dose) ** world.true_hill)
            return base + occupancy * target_delta

        raise ValueError(f"unrecognised condition {condition!r}")

    world.condition_value = condition_value
    return world
