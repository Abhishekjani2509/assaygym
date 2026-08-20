"""AssayGym Phase 5 — the scripted baselines.

**This is the phase that makes the project credible.** Before letting any model
near the environment, answer the question almost nobody answers: does the reward
actually separate competence from noise? Four scripted policies spanning the
range from noise to good practice, run over the same seeds, are how you find
out. If the ladder is not monotone, the bug is in the world model or the
scoring — not in the policy, and never in the policy's constants.

The four:

``random_policy``       — the floor. 24 random wells, random everything.
``prior_parrot_policy`` — the published failure mode as code: submit the
                          literature verbatim, run zero plates. If this beats
                          ``competent_doe`` the environment is broken; if it
                          ties, the environment is measuring recall rather than
                          experimentation.
``naive_screen_policy`` — one plate, one well per condition, no controls,
                          filled from A1. What "I ran the experiment" looks like
                          without design.
``competent_doe_policy``— the reference. One balanced plate per lot, QC, drop
                          the degraded lot, flag contaminated quadrants,
                          normalise each plate against its own controls, pool,
                          call, fit.

A third rng
-----------
World generation uses ``default_rng(seed)``, assay noise ``default_rng(seed +
10_000)``, and policy randomness ``default_rng(seed + 20_000)``. Three
independent streams, for the same reason there were two: at seed 41 every policy
must face the identical hidden world and the identical measurement noise, so the
only thing separating their scores is what they did.

Ablations
---------
``competent_doe_policy`` takes keyword switches that turn off one design step at
a time. They exist so the environment can be asked a harder question than "does
the ladder go up": does each artifact in the environment actually earn its
place? An artifact whose defence costs nothing when removed is either too weak
to matter or was never being used. See ``run_baselines.py --ablate``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

from assaygym.assay import WELLS, quadrant
from assaygym.env import LOTS, AssayGym
from assaygym.rewards import score

__all__ = [
    "INTERIOR",
    "DOSES",
    "POLICY_RNG_OFFSET",
    "PlateView",
    "random_policy",
    "prior_parrot_policy",
    "naive_screen_policy",
    "competent_doe_policy",
    "call_everything_policy",
    "POLICIES",
    "ABLATIONS",
    "run_policy",
    "run_episode",
]


# Rows B-G x columns 2-11: 60 wells, no perimeter. The whole edge-bias artifact
# is defeated by never using the edge, which is the point — it rewards knowing
# the practice, and the fix is trivial once you know it.
INTERIOR = [f"{r}{c}" for r in "BCDEFG" for c in range(2, 12)]

# Must span the full EC50 range (10^0.5 to 10^3.5 nM) *including the top*, or
# the plateau cannot be estimated and every fit is biased low.
DOSES = [1, 10, 30, 100, 300, 1000, 3000, 30000]

POLICY_RNG_OFFSET = 20_000

NTC_PER_QUADRANT = 2      # per-quadrant controls are what make contamination visible
N_POS = 3
CMPD_REPS = 2
WINDOW_EXCLUSION_FACTOR = 0.65
CONTAM_FLOOR = 0.25       # a real effect is 0.20; contamination is ~0.45
CONTAM_SPREAD_MULT = 4.0


@dataclass
class PlateView:
    """What a policy knows about a plate: only what the tools handed back.

    Deliberately not a :class:`~assaygym.assay.PlateResult`. A policy must never
    reach into ``env.plates``, because that is the environment's bookkeeping and
    an LLM agent driving the same tools would not have it. Duck-types with
    ``PlateResult`` on ``layout`` and ``values`` so the helpers take either.
    """

    plate_id: str
    lot: str
    layout: Dict[str, str]
    values: Dict[str, float]
    excluded: bool = False
    assay_window: Optional[float] = None
    z_prime: Optional[float] = None


# ---------------------------------------------------------------------------
# 5.1 Shared helpers
# ---------------------------------------------------------------------------


def _balanced_layout(
    genes: Sequence[str],
    reps: int = 2,
    rng: Optional[np.random.Generator] = None,
    wells: Sequence[str] = tuple(INTERIOR),
) -> Dict[str, str]:
    """Interior wells, two NTC reserved per quadrant, everything else shuffled.

    The per-quadrant controls are the load-bearing part. Contamination hits one
    quadrant; controls bunched in one corner leave the agent unable to tell a
    blown quadrant from a real effect, so the negative controls are scattered
    across all four by construction rather than by luck of the shuffle.

    Conditions beyond the free wells are truncated. At every shipped tier they
    fit: 12 genes at 2 reps is 8 NTC + 3 POS + 24 KD + 16 dose = 51 wells, which
    is exactly the $1,041 plate the hard tier buys three of.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    by_quadrant: Dict[int, List[str]] = {0: [], 1: [], 2: [], 3: []}
    for well in wells:
        by_quadrant[quadrant(well)].append(well)

    layout: Dict[str, str] = {}
    free: List[str] = []
    for q in range(4):
        pool = list(by_quadrant[q])
        rng.shuffle(pool)
        for well in pool[:NTC_PER_QUADRANT]:
            layout[well] = "NTC"
        free.extend(pool[NTC_PER_QUADRANT:])

    rng.shuffle(free)
    conditions: List[str] = ["POS"] * N_POS
    for gene in genes:
        conditions.extend([f"KD:{gene}"] * reps)
    for dose in DOSES:
        conditions.extend([f"CMPD@{dose}"] * CMPD_REPS)

    for well, condition in zip(free, conditions):
        layout[well] = condition
    return layout


def _flag_contaminated(plate: Any) -> Set[int]:
    """Quadrants whose NTC mean sits far above the median of quadrant means.

    ``spread`` is the median absolute deviation of the four quadrant means — the
    spec leaves it unnamed, and the MAD is the choice that does not get inflated
    by the very outlier it is trying to detect. The 0.25 floor keeps a quiet
    plate from flagging noise: a real effect is 0.20 and contamination is ~0.45,
    so the floor sits between them.
    """
    per_quadrant: Dict[int, List[float]] = defaultdict(list)
    for well, condition in plate.layout.items():
        if condition == "NTC":
            per_quadrant[quadrant(well)].append(plate.values[well])
    if len(per_quadrant) < 3:
        return set()

    means = {q: float(np.mean(v)) for q, v in per_quadrant.items()}
    median = float(np.median(list(means.values())))
    spread = float(np.median([abs(m - median) for m in means.values()]))
    threshold = max(CONTAM_FLOOR, CONTAM_SPREAD_MULT * spread)
    return {q for q, m in means.items() if m - median > threshold}


def _plate_effects(
    plate: Any, drop_quadrants: Sequence[int] = ()
) -> Dict[str, float]:
    """Subtract this plate's own NTC median from every well. The batch-offset fix.

    A per-plate additive shift is invisible in raw values and cannot be removed
    by anything except a control carried on that same plate.
    """
    drop = set(drop_quadrants)
    ntc = [
        v for w, v in plate.values.items()
        if plate.layout[w] == "NTC" and quadrant(w) not in drop
    ]
    if not ntc:
        return {}
    base = float(np.median(ntc))
    return {
        w: v - base for w, v in plate.values.items() if quadrant(w) not in drop
    }


def _fit_log_ec50(effects: Mapping[float, Sequence[float]]) -> Optional[float]:
    """Half-maximal interpolation in log-dose space. No curve fitting.

    Median effect per dose, plateau from the mean of the top two doses, then the
    first upward crossing of half-plateau, linearly interpolated between the two
    bracketing log-doses. ``curve_fit`` would be more principled and would also
    fail to converge on noisy data often enough to matter; interpolation always
    returns something, and something is what a policy needs.

    Signs take care of themselves: a knockdown that lowers the readout gives a
    negative plateau and negative effects, and their ratio is still positive.
    """
    doses = sorted(d for d in effects if d > 0 and len(effects[d]) > 0)
    if len(doses) < 3:
        return None

    medians = [float(np.median(effects[d])) for d in doses]
    plateau = float(np.mean(medians[-2:]))
    if abs(plateau) < 1e-9:
        return None

    fraction = [m / plateau for m in medians]
    log_dose = [float(np.log10(d)) for d in doses]

    for i in range(1, len(doses)):
        if fraction[i - 1] < 0.5 <= fraction[i]:
            f0, f1 = fraction[i - 1], fraction[i]
            if f1 == f0:
                return log_dose[i]
            t = (0.5 - f0) / (f1 - f0)
            return log_dose[i - 1] + t * (log_dose[i] - log_dose[i - 1])

    # No crossing. Either the curve is already above half at the lowest dose
    # (EC50 below the tested range) or never reaches it (above the range).
    # Clamp to the range actually tested rather than extrapolating off it.
    return log_dose[0] if fraction[0] >= 0.5 else log_dose[-1]


def _run(env: AssayGym, layout: Mapping[str, str], lot: str) -> Optional[PlateView]:
    """design_and_run, wrapped so a budget refusal is a None rather than a crash."""
    out = env.design_and_run(layout, lot)
    if "error" in out:
        return None
    return PlateView(
        plate_id=out["plate_id"], lot=out["lot"],
        layout=dict(layout), values=dict(out["values"]),
    )


def _call_hits(
    effects: Mapping[str, float], genes: Sequence[str], threshold: float
) -> tuple[List[str], Dict[str, int]]:
    hits, signs = [], {}
    for gene in genes:
        effect = effects.get(f"KD:{gene}")
        if effect is None:
            continue
        if abs(effect) > threshold:
            hits.append(gene)
            signs[gene] = 1 if effect >= 0 else -1
    return hits, signs


# ---------------------------------------------------------------------------
# 5.2 The four policies
# ---------------------------------------------------------------------------


def random_policy(env: AssayGym, briefing: Mapping[str, Any],
                  rng: np.random.Generator) -> None:
    """The floor. 24 random wells, random conditions, random answer."""
    genes = list(briefing["loci"])
    conditions = (["NTC", "POS"] + [f"KD:{g}" for g in genes]
                  + [f"CMPD@{d}" for d in DOSES])
    wells = [str(w) for w in rng.choice(np.array(WELLS), size=24, replace=False)]
    layout = {w: str(rng.choice(np.array(conditions))) for w in wells}
    _run(env, layout, str(rng.choice(np.array(briefing["reagent_lots"]))))

    k = int(rng.integers(0, len(genes) + 1))
    hits = [str(g) for g in rng.choice(np.array(genes), size=k, replace=False)] if k else []
    signs = {g: int(rng.choice(np.array([-1, 1]))) for g in hits}
    # Uniform over the log-dose range it could actually test — it knows the dose
    # ladder, and nothing else.
    lo, hi = float(np.log10(min(DOSES))), float(np.log10(max(DOSES)))
    env.submit(hits, signs, float(rng.uniform(lo, hi)))


def prior_parrot_policy(env: AssayGym, briefing: Mapping[str, Any],
                        rng: np.random.Generator) -> None:
    """The published failure mode, implemented as a policy. Runs ZERO plates.

    Submits the literature prior verbatim, guesses every direction as -1, and
    guesses a fixed log_ec50 of 2.0 — the middle of the range. The environment
    told it in the briefing that the prior comes from a different cell
    background and may be wrong, so this is a choice, not entrapment.
    """
    reported = list(briefing["literature_prior"]["previously_reported_hits"])
    env.submit(reported, {g: -1 for g in reported}, 2.0)


# What a policy with no dose series has to fall back on: the middle of the
# testable range. Identical to the prior-parrot's guess, for the same reason --
# neither of them measured anything.
BLIND_LOG_EC50 = 2.0


def naive_screen_policy(env: AssayGym, briefing: Mapping[str, Any],
                        rng: np.random.Generator) -> None:
    """One plate, one well per knockdown, no controls, no dose series, from A1.

    What "I ran the experiment" looks like without design. It has no negative
    control, so it thresholds against the median of its own wells; it fills from
    A1, so the layout decides which conditions catch the edge bias; it has one
    well per condition, so pipetting error is indistinguishable from signal; and
    it never runs a dose-response at all, so it has to guess the EC50.

    **The missing dose series is the point, and it was measured rather than
    assumed.** BUILD_SPEC says "one well per condition" without saying whether
    the compound doses are among the conditions, and the two readings give very
    different ledgers. Measured over 200 seeds per tier, strict_pass:

    ==========================  =====  ========  ====
    reading                     clean  standard  hard
    ==========================  =====  ========  ====
    no dose series, guess 2.0   0.280     0.080  0.020
    dose series + curve fit     0.630     0.115  0.005
    BUILD_SPEC target           0.275     0.075  0.005
    ==========================  =====  ========  ====

    The first reading reproduces all three target cells; the second misses clean
    by 0.355. A screener that runs a dose-response series and fits it is not
    being naive, so this is also the reading that matches the policy's name.
    """
    genes = list(briefing["loci"])
    threshold = float(briefing["hit_threshold"])
    conditions = [f"KD:{g}" for g in genes]
    layout = dict(zip(WELLS[: len(conditions)], conditions))

    plate = _run(env, layout, str(rng.choice(np.array(briefing["reagent_lots"]))))
    if plate is None:
        env.submit([], {}, None)
        return

    centre = float(np.median(list(plate.values.values())))
    effects = {c: plate.values[w] - centre for w, c in layout.items()}
    hits, signs = _call_hits(effects, genes, threshold)
    env.submit(hits, signs, BLIND_LOG_EC50)


def competent_doe_policy(
    env: AssayGym,
    briefing: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    use_qc: bool = True,
    use_lot_exclusion: bool = True,
    use_ntc_normalisation: bool = True,
    flag_contamination: bool = True,
    reps: int = 2,
    interior_only: bool = True,
) -> None:
    """The reference policy. Each keyword turns off exactly one defence.

    Steps, in order:

    1. one balanced plate per lot (three plates), calling ``qc()`` on each
    2. exclude any plate whose assay window is below 0.65x the median window —
       this catches the degraded lot without ever knowing which one it is
    3. flag contaminated quadrants on the survivors and drop those wells
    4. normalise each plate against its own NTC median, pool, take the median
       per condition
    5. call a hit where ``|effect| > hit_threshold``; sign from the effect sign
    6. fit the EC50 by half-maximal interpolation

    ``use_qc=False`` means no quality control **of any kind**: no ``qc()`` calls,
    no lot exclusion, and no contamination flagging. The narrower switches
    isolate the individual defences.
    """
    genes = list(briefing["loci"])
    threshold = float(briefing["hit_threshold"])
    wells = INTERIOR if interior_only else WELLS
    lots = list(briefing["reagent_lots"]) or list(LOTS)

    if not use_qc:
        use_lot_exclusion = False
        flag_contamination = False

    # --- 1. one balanced plate per lot ------------------------------------
    plates: List[PlateView] = []
    for lot in lots:
        layout = _balanced_layout(genes, reps=reps, rng=rng, wells=wells)
        plate = _run(env, layout, lot)
        if plate is None:
            continue  # budget refused it; carry on with what we have
        if use_qc:
            report = env.qc(plate.plate_id)
            plate.assay_window = report.get("assay_window")
            plate.z_prime = report.get("z_prime")
        plates.append(plate)

    if not plates:
        env.submit([], {}, None)
        return

    # --- 2. drop the degraded lot ------------------------------------------
    if use_lot_exclusion:
        windows = {p.plate_id: p.assay_window for p in plates
                   if p.assay_window is not None}
        if len(windows) >= 2:
            median_window = float(np.median(list(windows.values())))
            for plate in plates:
                w = plate.assay_window
                if w is not None and w < WINDOW_EXCLUSION_FACTOR * median_window:
                    env.exclude_plate(
                        plate.plate_id,
                        f"assay window {w:.3f} below "
                        f"{WINDOW_EXCLUSION_FACTOR} x median {median_window:.3f}",
                    )
                    plate.excluded = True

    usable = [p for p in plates if not p.excluded] or plates

    # --- 3-4. flag, normalise, pool ----------------------------------------
    if use_ntc_normalisation:
        pooled = _pool_per_plate(usable, flag_contamination)
    else:
        pooled = _pool_globally(usable, flag_contamination)

    effects = {c: float(np.median(v)) for c, v in pooled.items() if v}

    # --- 5-6. call and fit --------------------------------------------------
    hits, signs = _call_hits(effects, genes, threshold)
    dose_effects = {
        float(d): pooled[f"CMPD@{d}"] for d in DOSES if pooled.get(f"CMPD@{d}")
    }
    env.submit(hits, signs, _fit_log_ec50(dose_effects))


def _pool_per_plate(
    plates: Sequence[PlateView], flag_contamination: bool
) -> Dict[str, List[float]]:
    """Each plate normalised against its OWN NTC median, then pooled."""
    pooled: Dict[str, List[float]] = defaultdict(list)
    for plate in plates:
        drop = _flag_contaminated(plate) if flag_contamination else set()
        effects = _plate_effects(plate, drop)
        for well, effect in effects.items():
            pooled[plate.layout[well]].append(effect)
    return pooled


def _pool_globally(
    plates: Sequence[PlateView], flag_contamination: bool
) -> Dict[str, List[float]]:
    """One NTC reference across every plate — the no-normalisation ablation.

    Still a sane analysis, just blind to the per-plate batch offset: every plate
    is centred on the same number, so a plate that was displaced as a whole
    carries that displacement into the pool.
    """
    all_ntc: List[float] = []
    drops: Dict[str, Set[int]] = {}
    for plate in plates:
        drops[plate.plate_id] = (
            _flag_contaminated(plate) if flag_contamination else set()
        )
        all_ntc.extend(
            v for w, v in plate.values.items()
            if plate.layout[w] == "NTC" and quadrant(w) not in drops[plate.plate_id]
        )
    base = float(np.median(all_ntc)) if all_ntc else 0.0

    pooled: Dict[str, List[float]] = defaultdict(list)
    for plate in plates:
        for well, value in plate.values.items():
            if quadrant(well) in drops[plate.plate_id]:
                continue
            pooled[plate.layout[well]].append(value - base)
    return pooled


def call_everything_policy(env: AssayGym, briefing: Mapping[str, Any],
                           rng: np.random.Generator) -> None:
    """The degenerate exploit: call every gene a hit. Used by verify.py.

    This is the classic way to farm recall, and it has to lose.
    """
    genes = list(briefing["loci"])
    env.submit(genes, {g: -1 for g in genes}, None)


POLICIES: Dict[str, Callable[..., None]] = {
    "random": random_policy,
    "prior_parrot": prior_parrot_policy,
    "naive_screen": naive_screen_policy,
    "competent_doe": competent_doe_policy,
}


def _ablation(**kwargs: Any) -> Callable[..., None]:
    def run(env, briefing, rng):
        return competent_doe_policy(env, briefing, rng, **kwargs)
    return run


# One design step removed at a time. Every one should cost score; an ablation
# that costs nothing means the artifact it defends against is too weak to matter
# or the defence was never being used.
ABLATIONS: Dict[str, Callable[..., None]] = {
    "full": _ablation(),
    "no_lot_exclusion": _ablation(use_lot_exclusion=False),
    "no_ntc_normalisation": _ablation(use_ntc_normalisation=False),
    "no_contamination_flag": _ablation(flag_contamination=False),
    "one_replicate": _ablation(reps=1),
    "full_plate_with_edges": _ablation(interior_only=False),
    "no_qc_at_all": _ablation(use_qc=False),
}


# ---------------------------------------------------------------------------
# Running one episode
# ---------------------------------------------------------------------------


def run_episode(
    policy: Callable[..., None], seed: int, tier: str = "standard"
) -> Dict[str, Any]:
    """One seeded episode end to end. Returns the score dict.

    Policy randomness comes from a third independent stream so that two policies
    at the same seed face the identical world and the identical measurement
    noise. A policy that never submits is forced to an empty submission rather
    than dropped, so every episode lands on the ledger.
    """
    env = AssayGym(seed, tier)
    briefing = env.reset()
    rng = np.random.default_rng(seed + POLICY_RNG_OFFSET)
    policy(env, briefing, rng)
    if not env.done:
        env.submit([], {}, None)
    return score(env)


def run_policy(
    policy: Callable[..., None], n: int = 200, tier: str = "standard",
    seed0: int = 0,
) -> List[Dict[str, Any]]:
    return [run_episode(policy, seed0 + i, tier) for i in range(n)]
