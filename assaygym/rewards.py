"""AssayGym Phase 4 — the judge.

Three numbers, and they are kept conceptually separate on purpose. Collapsing
them into one would hide the thing each is for.

``strict_pass`` — all or nothing. **This is the headline number.** The hit set
exactly right, every direction right, the EC50 inside tolerance. Lead with it.

``endpoint`` — sparse and verifiable, weighted over hit F1, sign accuracy and
EC50 error. It gives partial credit generously enough that a policy which just
parrots the literature prior scores 0.4-0.67 on it, which makes the environment
look weaker than it is. Report it, never lead with it.

``shaped`` — dense and mechanical, for RL. ``endpoint`` plus five process terms.

The critical property of the process terms: every one is a **checkable fact
about the trajectory**, not an opinion about it. Count the control wells. Check
the plate ids. Compare lots against the ones the world degraded. A persuasive
transcript can flatter an LLM judge, but it cannot retroactively put control
wells on a plate that was already run. That is the whole argument for this
design over rubric-graded shaping, and it is why every term here is arithmetic
over ``PlateResult`` objects rather than a reading of what the agent said it did.

Diagnostics are computed and **never summed into reward**. ``decoy_called`` in
particular is the direct measurement of prior-dependence — the whole point of
the environment — and it has to be reportable without contaminating the score
it is used to explain.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from assaygym.assay import PlateResult
from assaygym.world import World

__all__ = [
    "ENDPOINT_WEIGHTS",
    "SHAPED_WEIGHTS",
    "EC50_TOLERANCE",
    "EFFICIENCY_GATE",
    "MIN_NTC_FOR_CONTROLS",
    "MIN_POS_FOR_CONTROLS",
    "MIN_PLATES_FOR_REPLICATION",
    "endpoint_terms",
    "strict_pass",
    "shaped_terms",
    "diagnostics",
    "score",
    "score_trajectory",
]


# --- 4.1 endpoint ----------------------------------------------------------
ENDPOINT_WEIGHTS: Dict[str, float] = {
    "hit_f1": 0.55,
    "sign_acc": 0.15,
    "ec50": 0.30,
}

# Full credit within 0.4 log units, roughly a 2.5x window — about what you would
# accept from a real curve fit.
EC50_TOLERANCE = 0.40

# Whether a submission at exactly the tolerance passes should not depend on the
# binary rounding of a subtraction, so the strict comparison carries a slack far
# below any difference that means anything scientifically.
_EC50_EPS = 1e-9

# --- 4.3 shaped ------------------------------------------------------------
SHAPED_WEIGHTS: Dict[str, float] = {
    "endpoint": 0.55,
    "controls": 0.10,
    "replication": 0.10,
    "self_normalizable": 0.05,
    "qc_hygiene": 0.12,
    "efficiency": 0.08,
}

# efficiency pays only once the answer is actually worth something. Without this
# gate the reward-optimal policy is to run nothing and bank the whole budget,
# and the environment stops measuring experimental work entirely. It is the
# single most important line in this file.
EFFICIENCY_GATE = 0.40

MIN_NTC_FOR_CONTROLS = 4
MIN_POS_FOR_CONTROLS = 2
MIN_PLATES_FOR_REPLICATION = 2


def _is_test_condition(condition: str) -> bool:
    """A measurement of something unknown, as opposed to a control well."""
    return condition.startswith("KD:") or condition.startswith("CMPD@")


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _submission_parts(
    submission: Optional[Mapping[str, Any]]
) -> tuple[List[str], Dict[str, int], Optional[float]]:
    if not submission:
        return [], {}, None
    hits = [str(h) for h in (submission.get("hits") or [])]
    signs = {str(k): int(v) for k, v in (submission.get("signs") or {}).items()}
    log_ec50 = submission.get("log_ec50")
    return hits, signs, (None if log_ec50 is None else float(log_ec50))


# ---------------------------------------------------------------------------
# 4.1 endpoint — sparse, verifiable
# ---------------------------------------------------------------------------


def endpoint_terms(
    world: World, submission: Optional[Mapping[str, Any]]
) -> Dict[str, float]:
    """The three endpoint components and their weighted total."""
    hits, signs, log_ec50 = _submission_parts(submission)
    submitted = set(hits)
    truth = set(world.true_hits)

    true_pos = submitted & truth
    n_tp = len(true_pos)

    # F1, not precision and not recall. Precision alone is farmed by submitting
    # one confident gene; recall alone is farmed by submitting every gene.
    precision = n_tp / len(submitted) if submitted else 0.0
    recall = n_tp / len(truth) if truth else 0.0
    hit_f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Over correctly-identified hits ONLY, and 0.0 — never a vacuous 1.0 — when
    # none were identified. Vacuous truth here would pay an agent for submitting
    # nothing, which is exactly the behaviour the whole design is built against.
    if n_tp == 0:
        sign_acc = 0.0
    else:
        correct = sum(
            1 for g in true_pos if signs.get(g, 0) == world.true_signs.get(g, 0)
        )
        sign_acc = correct / n_tp

    if log_ec50 is None:
        ec50 = 0.0
    else:
        err = abs(log_ec50 - world.true_log_ec50)
        ec50 = _clip01(1.0 - err / EC50_TOLERANCE)

    terms = {"hit_f1": float(hit_f1), "sign_acc": float(sign_acc), "ec50": float(ec50)}
    terms["endpoint"] = float(sum(ENDPOINT_WEIGHTS[k] * terms[k] for k in ENDPOINT_WEIGHTS))
    return terms


# ---------------------------------------------------------------------------
# 4.2 strict_pass — all or nothing, the headline
# ---------------------------------------------------------------------------


def strict_pass(world: World, submission: Optional[Mapping[str, Any]]) -> float:
    """1.0 only for a completely correct answer. Otherwise 0.0.

    Exact set equality, not a superset and not a subset: calling every gene a
    hit is the classic way to farm recall, and it must score zero here.
    """
    hits, signs, log_ec50 = _submission_parts(submission)
    if set(hits) != set(world.true_hits):
        return 0.0
    for gene in world.true_hits:
        if signs.get(gene, 0) != world.true_signs.get(gene, 0):
            return 0.0
    if log_ec50 is None:
        return 0.0
    if abs(log_ec50 - world.true_log_ec50) > EC50_TOLERANCE + _EC50_EPS:
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# 4.3 shaped — dense, mechanical
# ---------------------------------------------------------------------------


def _controls(plates: Sequence[PlateResult]) -> float:
    """Fraction of non-excluded plates carrying >=4 NTC and >=2 POS."""
    active = [p for p in plates if not p.excluded]
    if not active:
        return 0.0
    ok = 0
    for p in active:
        conds = list(p.layout.values())
        if (conds.count("NTC") >= MIN_NTC_FOR_CONTROLS
                and conds.count("POS") >= MIN_POS_FOR_CONTROLS):
            ok += 1
    return ok / len(active)


def _replication(plates: Sequence[PlateResult], hits: Sequence[str]) -> float:
    """Fraction of SUBMITTED hits measured on >=2 distinct non-excluded plates.

    Submitted, not measured: an agent that replicates every condition beautifully
    and then calls the wrong genes gets nothing here. And distinct *plates*, not
    distinct wells — two wells on one plate share that plate's batch offset, its
    lot and its contamination, so they are not independent measurements.
    """
    if not hits:
        return 0.0
    active = [p for p in plates if not p.excluded]
    replicated = 0
    for gene in set(hits):
        condition = f"KD:{gene}"
        n_plates = sum(1 for p in active if condition in p.layout.values())
        if n_plates >= MIN_PLATES_FOR_REPLICATION:
            replicated += 1
    return replicated / len(set(hits))


def _self_normalizable(plates: Sequence[PlateResult]) -> float:
    """Fraction of plates carrying >=1 NTC alongside >=1 test condition.

    Every plate, including excluded ones. Self-normalisability is a property of
    how the plate was laid out, decided before it was run, so excluding a plate
    afterwards must not launder a design that could never have been normalised
    against itself.
    """
    if not plates:
        return 0.0
    ok = 0
    for p in plates:
        conds = list(p.layout.values())
        if "NTC" in conds and any(_is_test_condition(c) for c in conds):
            ok += 1
    return ok / len(plates)


def _qc_hygiene(world: World, plates: Sequence[PlateResult]) -> float:
    """clip(caught_bad_lot_fraction - wrongly_excluded_good_fraction, 0, 1).

    Two branches, and they are genuinely different code paths:

    * A bad lot exists and the agent ran plates on it — the first term is the
      fraction of those it excluded. This is the skill the term is for: notice
      the collapsed assay window, drop the plate.
    * No bad-lot plates were run, either because the world has no bad lot or
      because the agent never touched it. There is nothing to catch, so the
      first term is 1.0 and the metric reduces to a pure penalty for excluding
      good plates.

    Running no plates at all scores 0.0 rather than 1.0. The literal formula
    would hand a do-nothing policy a free 0.12 of shaped reward for hygiene it
    never demonstrated, which is the same failure the efficiency gate exists to
    prevent. Process credit has to require process.
    """
    if not plates:
        return 0.0

    bad_lots = set(world.bad_lots)
    bad_plates = [p for p in plates if p.lot in bad_lots]
    good_plates = [p for p in plates if p.lot not in bad_lots]

    if bad_plates:
        caught = sum(1 for p in bad_plates if p.excluded) / len(bad_plates)
    else:
        caught = 1.0

    if good_plates:
        wrongly = sum(1 for p in good_plates if p.excluded) / len(good_plates)
    else:
        wrongly = 0.0

    return _clip01(caught - wrongly)


def _efficiency(endpoint: float, usd_spent: float, budget_usd: float) -> float:
    """Fraction of budget unspent, GATED on endpoint > 0.4.

    Thrift only pays if it worked. Without the gate the reward-optimal policy is
    to run nothing and bank the entire budget, which would make the environment
    reward exactly the behaviour it exists to detect.
    """
    if endpoint <= EFFICIENCY_GATE:
        return 0.0
    if budget_usd <= 0:
        return 0.0
    return _clip01(1.0 - usd_spent / budget_usd)


def shaped_terms(
    world: World,
    plates: Sequence[PlateResult],
    submission: Optional[Mapping[str, Any]],
    usd_spent: float,
    budget_usd: float,
    endpoint: float,
) -> Dict[str, float]:
    """The six shaped components and their weighted total."""
    hits, _, _ = _submission_parts(submission)
    terms = {
        "endpoint": float(endpoint),
        "controls": _controls(plates),
        "replication": _replication(plates, hits),
        "self_normalizable": _self_normalizable(plates),
        "qc_hygiene": _qc_hygiene(world, plates),
        "efficiency": _efficiency(endpoint, usd_spent, budget_usd),
    }
    terms["shaped"] = float(sum(SHAPED_WEIGHTS[k] * terms[k] for k in SHAPED_WEIGHTS))
    return terms


# ---------------------------------------------------------------------------
# 4.4 Diagnostics — computed, never summed into reward
# ---------------------------------------------------------------------------


def diagnostics(
    world: World,
    plates: Sequence[PlateResult],
    submission: Optional[Mapping[str, Any]],
    usd_spent: float,
) -> Dict[str, float]:
    """Reported alongside the reward, never folded into it.

    ``decoy_called`` is the direct measurement of prior-dependence: how many of
    the literature's plausible-but-false hits the agent repeated back. It is the
    number the whole environment exists to produce, and keeping it out of the
    reward is what lets it be reported without the reward having been tuned
    against it.
    """
    hits, _, _ = _submission_parts(submission)
    submitted = set(hits)
    truth = set(world.true_hits)
    n_tp = len(submitted & truth)

    return {
        "precision": float(n_tp / len(submitted)) if submitted else 0.0,
        "recall": float(n_tp / len(truth)) if truth else 0.0,
        "decoy_called": int(len(submitted & set(world.decoys))),
        "omitted_recovered": int(len(submitted & set(world.omitted))),
        "prior_trap": int(bool(world.decoys or world.omitted)),
        "n_plates": int(len(plates)),
        "n_excluded": int(sum(1 for p in plates if p.excluded)),
        "usd_spent": float(usd_spent),
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def score_trajectory(
    world: World,
    plates: Sequence[PlateResult],
    submission: Optional[Mapping[str, Any]],
    usd_spent: float,
    budget_usd: float,
) -> Dict[str, Any]:
    """Score an episode from its raw parts. ``score(env)`` is the usual door."""
    ep = endpoint_terms(world, submission)
    sh = shaped_terms(
        world, plates, submission, usd_spent, budget_usd, ep["endpoint"]
    )
    return {
        # Lead with strict_pass. It is the headline.
        "strict_pass": strict_pass(world, submission),
        "endpoint": ep["endpoint"],
        "shaped": sh["shaped"],
        "endpoint_terms": {k: ep[k] for k in ENDPOINT_WEIGHTS},
        "shaped_terms": {k: sh[k] for k in SHAPED_WEIGHTS},
        "diagnostics": diagnostics(world, plates, submission, usd_spent),
    }


def score(env: Any) -> Dict[str, Any]:
    """Score a finished :class:`~assaygym.env.AssayGym` episode.

    An episode with no submission is scored as an empty submission rather than
    refused, so a policy that runs out of turns still lands on the ledger.
    """
    if env.world is None:
        raise RuntimeError("environment has no world; call reset() first")
    return score_trajectory(
        world=env.world,
        plates=list(env.plates.values()),
        submission=env.submission,
        usd_spent=env.usd_spent,
        budget_usd=float(env.world.diff.budget_usd),
    )
