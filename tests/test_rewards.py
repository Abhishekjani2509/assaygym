"""Phase 4 acceptance checks for assaygym/rewards.py.

Run: ./.venv/bin/python -m pytest tests/ -q -s   (see CONTRIBUTING.md)

Three acceptance checks from BUILD_SPEC section "Phase 4":
  1. a perfect submission on `clean` scores endpoint ~= 1.0, strict_pass = 1.0
  2. an empty submission scores endpoint = 0.0
  3. an agent that runs zero plates and submits gets controls = 0,
     replication = 0, efficiency = 0

Plus the properties those three do not pin down: that strict_pass and endpoint
decouple, that the efficiency gate holds, that both qc_hygiene branches work,
that sign_acc is not vacuously 1.0, and that the diagnostics stay out of the
reward.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assaygym.assay import PlateResult  # noqa: E402
from assaygym.env import AssayGym  # noqa: E402
from assaygym.rewards import (  # noqa: E402
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
from assaygym.world import (  # noqa: E402
    override_phenotype_from_deltas,
    sample_world,
)

INTERIOR = [f"{r}{c}" for r in "BCDEFG" for c in range(2, 12)]


def _world(seed=1, tier="clean"):
    return override_phenotype_from_deltas(sample_world(seed, tier))


def _perfect(world, ec50_offset=0.0):
    return {
        "hits": list(world.true_hits),
        "signs": dict(world.true_signs),
        "log_ec50": world.true_log_ec50 + ec50_offset,
    }


def _plate(plate_id, layout, lot="LOT-A", excluded=False, cost=1041.0):
    """A PlateResult with the fields the judge reads. Values are irrelevant:
    every process term is a fact about the *layout*, not about the numbers."""
    return PlateResult(
        plate_id=plate_id, lot=lot, layout=dict(layout),
        values={w: 0.0 for w in layout}, day_run=0, cost_usd=cost,
        excluded=excluded,
    )


def _good_layout(genes, n_ntc=8, n_pos=4, doses=(100,)):
    wells = iter(INTERIOR)
    layout = {}
    for _ in range(n_ntc):
        layout[next(wells)] = "NTC"
    for _ in range(n_pos):
        layout[next(wells)] = "POS"
    for g in genes:
        layout[next(wells)] = f"KD:{g}"
    for d in doses:
        layout[next(wells)] = f"CMPD@{d}"
    return layout


def _run_env(seed=1, tier="clean", n_plates=3, lots=("LOT-A", "LOT-B", "LOT-C")):
    env = AssayGym(seed, tier)
    env.reset()
    layout = _good_layout(env.world.genes)
    for i in range(n_plates):
        assert "error" not in env.design_and_run(layout, lots[i % len(lots)])
    return env


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def test_weights_sum_to_one_and_match_the_spec():
    assert ENDPOINT_WEIGHTS == {"hit_f1": 0.55, "sign_acc": 0.15, "ec50": 0.30}
    assert SHAPED_WEIGHTS == {
        "endpoint": 0.55, "controls": 0.10, "replication": 0.10,
        "self_normalizable": 0.05, "qc_hygiene": 0.12, "efficiency": 0.08,
    }
    print(f"\n[measured] endpoint weights sum = {sum(ENDPOINT_WEIGHTS.values()):.10f}")
    print(f"[measured] shaped weights sum   = {sum(SHAPED_WEIGHTS.values()):.10f}")
    assert sum(ENDPOINT_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(SHAPED_WEIGHTS.values()) == pytest.approx(1.0)


def test_totals_are_fully_explained_by_the_declared_terms():
    """Nothing outside the weight tables can leak into a reward.

    This is the structural form of "diagnostics are never summed into reward":
    both totals reproduce exactly from their declared components, so any hidden
    contribution would show up as a mismatch.
    """
    env = _run_env(seed=4, tier="hard", n_plates=2)
    env.exclude_plate("P1", "window")
    env.submit(**_perfect(env.world, 0.1))
    res = score(env)

    ep = sum(ENDPOINT_WEIGHTS[k] * res["endpoint_terms"][k] for k in ENDPOINT_WEIGHTS)
    sh = sum(SHAPED_WEIGHTS[k] * res["shaped_terms"][k] for k in SHAPED_WEIGHTS)
    assert res["endpoint"] == pytest.approx(ep, abs=1e-12)
    assert res["shaped"] == pytest.approx(sh, abs=1e-12)

    # And no diagnostic is a reward term.
    assert set(res["diagnostics"]) & (set(ENDPOINT_WEIGHTS) | set(SHAPED_WEIGHTS)) == set()


# --------------------------------------------------------------------------
# Acceptance 1 — a perfect submission on clean
# --------------------------------------------------------------------------


def test_perfect_submission_on_clean_scores_one():
    stricts, endpoints = [], []
    for seed in range(50):
        env = _run_env(seed, "clean", n_plates=3)
        env.submit(**_perfect(env.world))
        res = score(env)
        stricts.append(res["strict_pass"])
        endpoints.append(res["endpoint"])
        assert res["endpoint_terms"] == {"hit_f1": 1.0, "sign_acc": 1.0, "ec50": 1.0}
    print(f"[measured] perfect submission, clean, 50 seeds: strict_pass = "
          f"{np.mean(stricts):.3f}, endpoint = {np.mean(endpoints):.6f}")
    assert all(s == 1.0 for s in stricts)
    assert all(e == pytest.approx(1.0) for e in endpoints)


# --------------------------------------------------------------------------
# Acceptance 2 — an empty submission
# --------------------------------------------------------------------------


def test_empty_submission_scores_zero_endpoint():
    for tier in ("clean", "standard", "hard"):
        env = _run_env(1, tier, n_plates=2)
        env.submit([], {}, None)
        res = score(env)
        assert res["endpoint"] == 0.0, tier
        assert res["strict_pass"] == 0.0, tier
        assert res["endpoint_terms"] == {"hit_f1": 0.0, "sign_acc": 0.0, "ec50": 0.0}
    print("[measured] empty submission: endpoint = 0.0 on all three tiers")

    # No submission at all (the model never called submit) scores the same.
    env = _run_env(1, "clean", n_plates=1)
    assert env.submission is None
    assert score(env)["endpoint"] == 0.0


def test_sign_acc_is_zero_not_vacuously_one():
    """No correctly-identified hits means 0.0. A vacuous 1.0 would pay for silence."""
    w = _world(1, "clean")
    wrong = [g for g in w.genes if g not in w.true_hits][:3]

    # Every submitted gene is wrong, but every sign is confidently supplied.
    terms = endpoint_terms(w, {"hits": wrong, "signs": {g: -1 for g in wrong},
                               "log_ec50": None})
    assert terms["hit_f1"] == 0.0
    assert terms["sign_acc"] == 0.0, "vacuous truth would make this 1.0"

    assert endpoint_terms(w, {"hits": [], "signs": {}, "log_ec50": None})["sign_acc"] == 0.0
    assert endpoint_terms(w, None)["sign_acc"] == 0.0
    print("[measured] sign_acc with zero true positives = 0.0 (not 1.0)")


def test_sign_acc_is_over_true_positives_only():
    """False positives must not dilute the denominator.

    Found by tools/mutate.py: every existing sign test either had zero true
    positives or an exactly-correct hit set, and in both cases the submitted set
    and the true-positive set coincide -- so nothing separated "over correctly
    identified hits" from "over everything submitted". A false positive
    alongside correct hits does.
    """
    w = _world(3, "clean")
    extra = [g for g in w.genes if g not in w.true_hits][0]

    sub = {"hits": list(w.true_hits) + [extra],
           "signs": dict(w.true_signs) | {extra: 1},
           "log_ec50": w.true_log_ec50}
    terms = endpoint_terms(w, sub)
    print(f"[measured] 3 correct hits with right signs + 1 false positive: "
          f"sign_acc = {terms['sign_acc']:.4f} "
          f"(over everything submitted it would be {3 / 4:.4f})")
    assert terms["sign_acc"] == 1.0
    # The false positive is paid for on hit_f1, which is where it belongs.
    assert terms["hit_f1"] < 1.0

    victim = w.true_hits[0]
    sub2 = dict(sub, signs=dict(sub["signs"]) | {victim: -w.true_signs[victim]})
    t2 = endpoint_terms(w, sub2)
    print(f"[measured] same, one true-positive direction flipped: sign_acc = "
          f"{t2['sign_acc']:.4f} (analytic 2/3 = {2 / 3:.4f}; diluted "
          f"{2 / 4:.4f})")
    assert t2["sign_acc"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# strict_pass and endpoint must decouple
# --------------------------------------------------------------------------


def test_one_wrong_sign_kills_strict_pass_but_not_endpoint():
    """The whole point of two numbers: they must be able to disagree."""
    w = _world(3, "clean")
    good = _perfect(w)
    flipped = dict(good)
    victim = w.true_hits[0]
    flipped["signs"] = dict(good["signs"])
    flipped["signs"][victim] = -good["signs"][victim]

    ok = endpoint_terms(w, good)
    bad = endpoint_terms(w, flipped)
    print(f"[measured] one sign flipped ({victim}): endpoint "
          f"{ok['endpoint']:.4f} -> {bad['endpoint']:.4f} "
          f"(sign_acc {ok['sign_acc']:.4f} -> {bad['sign_acc']:.4f}), "
          f"strict_pass {strict_pass(w, good):.1f} -> {strict_pass(w, flipped):.1f}")

    assert strict_pass(w, good) == 1.0
    assert strict_pass(w, flipped) == 0.0     # all-or-nothing
    assert bad["hit_f1"] == 1.0               # the hit set is still perfect
    assert bad["ec50"] == 1.0
    assert bad["sign_acc"] == pytest.approx(2 / 3)
    assert bad["endpoint"] == pytest.approx(1.0 - 0.15 * (1 / 3))
    assert bad["endpoint"] > 0.9              # endpoint stays high


def test_strict_pass_requires_exact_set_equality():
    """Not a superset, not a subset. Calling every gene farms recall."""
    w = _world(3, "clean")
    good = _perfect(w)
    extra = [g for g in w.genes if g not in w.true_hits][0]

    superset = dict(good, hits=list(w.true_hits) + [extra])
    subset = dict(good, hits=list(w.true_hits)[:-1])
    every = dict(good, hits=list(w.genes),
                 signs={g: 1 for g in w.genes} | dict(w.true_signs))

    assert strict_pass(w, good) == 1.0
    assert strict_pass(w, superset) == 0.0
    assert strict_pass(w, subset) == 0.0
    assert strict_pass(w, every) == 0.0

    ep_every = endpoint_terms(w, every)
    print(f"[measured] 'call every gene a hit' on clean: strict_pass = 0.0, "
          f"endpoint = {ep_every['endpoint']:.4f} (hit_f1 = {ep_every['hit_f1']:.4f})")
    # Recall is 1.0 but precision is 3/8, so F1 caps the payoff.
    assert ep_every["hit_f1"] == pytest.approx(2 * (3 / 8) / (1 + 3 / 8))


def test_hit_f1_is_f1_not_precision_or_recall():
    w = _world(3, "clean")
    one_right = {"hits": [w.true_hits[0]], "signs": {}, "log_ec50": None}
    terms = endpoint_terms(w, one_right)
    diag = diagnostics(w, [], one_right, 0.0)

    print(f"[measured] one correct hit of three: precision = {diag['precision']:.4f}, "
          f"recall = {diag['recall']:.4f}, hit_f1 = {terms['hit_f1']:.4f}")
    assert diag["precision"] == 1.0            # precision alone would pay 1.0
    assert diag["recall"] == pytest.approx(1 / 3)
    assert terms["hit_f1"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# The EC50 boundary
# --------------------------------------------------------------------------


def test_ec50_tolerance_boundary():
    """Error of 0.4 passes; 0.401 does not. Both score 0.0 on the endpoint term."""
    assert EC50_TOLERANCE == 0.40
    w = _world(3, "clean")

    for sign in (+1, -1):
        at = _perfect(w, sign * 0.4)
        past = _perfect(w, sign * 0.401)
        err_at = abs(at["log_ec50"] - w.true_log_ec50)
        err_past = abs(past["log_ec50"] - w.true_log_ec50)
        print(f"[measured] offset {sign * 0.4:+.3f}: |err| = {err_at:.17f} -> "
              f"strict_pass {strict_pass(w, at):.1f}; "
              f"offset {sign * 0.401:+.4f}: |err| = {err_past:.17f} -> "
              f"strict_pass {strict_pass(w, past):.1f}")
        assert strict_pass(w, at) == 1.0
        assert strict_pass(w, past) == 0.0
        # The endpoint term reaches zero at the same place, from the other side.
        assert endpoint_terms(w, at)["ec50"] == pytest.approx(0.0, abs=1e-9)
        assert endpoint_terms(w, past)["ec50"] == 0.0

    # Inside the window the term is linear, and a missing answer scores 0.
    assert endpoint_terms(w, _perfect(w, 0.2))["ec50"] == pytest.approx(0.5)
    assert endpoint_terms(w, _perfect(w, 0.0))["ec50"] == pytest.approx(1.0)
    none_sub = dict(_perfect(w), log_ec50=None)
    assert endpoint_terms(w, none_sub)["ec50"] == 0.0
    assert strict_pass(w, none_sub) == 0.0


# --------------------------------------------------------------------------
# Acceptance 3 — the do-nothing policy
# --------------------------------------------------------------------------


def test_do_nothing_policy_gets_no_process_credit():
    """Runs zero plates, submits. controls / replication / efficiency all 0."""
    for tier in ("clean", "standard", "hard"):
        env = AssayGym(2, tier)
        env.reset()
        env.submit([], {}, None)
        res = score(env)
        t = res["shaped_terms"]
        assert res["diagnostics"]["n_plates"] == 0
        assert t["controls"] == 0.0, tier
        assert t["replication"] == 0.0, tier
        assert t["efficiency"] == 0.0, tier
        assert t["self_normalizable"] == 0.0, tier
        # Deliberate reading of the spec: the literal qc_hygiene formula returns
        # 1.0 with no plates (nothing missed, nothing over-excluded), handing a
        # do-nothing policy a free 0.12. Process credit must require process.
        assert t["qc_hygiene"] == 0.0, tier
        assert res["shaped"] == 0.0, tier
    print("[measured] do-nothing policy, all tiers: every shaped term = 0.0, "
          "shaped total = 0.0")


def test_prior_parrot_earns_no_experimental_credit():
    """The published failure mode: submit the literature verbatim, run nothing.

    It earns exactly zero on every term that requires a plate to have existed.
    It does NOT earn zero on efficiency: the gate is on `endpoint > 0.4`, and
    the spec itself notes that a prior-parrot scores 0.4-0.67 on endpoint, so on
    the seeds where the literature happens to be good enough it clears the gate
    and banks the unspent budget. Measured below, and recorded here rather than
    assumed away -- see the README for the size of the hole and the one-line
    tightening that would close it.

    The gate still does its job: running nothing is never reward-*optimal*,
    because the five process terms it cannot touch are worth 0.37 of shaped.
    """
    rates = {}
    for tier in ("clean", "standard", "hard"):
        endpoints, efficiencies, shapes = [], [], []
        for seed in range(200):
            env = AssayGym(seed, tier)
            brief = env.reset()
            reported = brief["literature_prior"]["previously_reported_hits"]
            env.submit(reported, {g: -1 for g in reported}, 2.0)
            res = score(env)
            t = res["shaped_terms"]
            # Zero on everything that requires a plate.
            assert t["controls"] == 0.0 and t["replication"] == 0.0
            assert t["self_normalizable"] == 0.0 and t["qc_hygiene"] == 0.0
            assert res["diagnostics"]["n_plates"] == 0
            # efficiency is either 0 (gated out) or 1 (nothing spent).
            assert t["efficiency"] in (0.0, 1.0)
            assert (t["efficiency"] == 1.0) == (res["endpoint"] > EFFICIENCY_GATE)
            endpoints.append(res["endpoint"])
            efficiencies.append(t["efficiency"])
            shapes.append(res["shaped"])
        rates[tier] = (float(np.mean(endpoints)), float(np.mean(efficiencies)),
                       float(np.mean(shapes)))

    print("[measured] prior-parrot (zero plates), 200 seeds per tier:")
    for tier, (ep, eff, sh) in rates.items():
        print(f"             {tier:9s} endpoint {ep:.4f}  clears the gate "
              f"{100 * eff:5.1f}% of seeds  shaped {sh:.4f}")

    # The comparison that matters: doing the work dominates, on every tier.
    env = _run_env(0, "clean", n_plates=3)
    env.submit(**_perfect(env.world))
    competent = score(env)["shaped"]
    print(f"[measured] competent campaign on clean: shaped = {competent:.4f} "
          f"vs prior-parrot {rates['clean'][2]:.4f}")
    assert competent > rates["clean"][2] + 0.4


# --------------------------------------------------------------------------
# The efficiency gate — the most important line in the file
# --------------------------------------------------------------------------


def test_efficiency_is_gated_on_endpoint():
    """Below the gate thrift pays nothing; above it, it pays the unspent fraction."""
    w = _world(3, "clean")
    plates = [_plate("P1", _good_layout(w.genes))]

    below = shaped_terms(w, plates, _perfect(w), usd_spent=1041.0,
                         budget_usd=6000.0, endpoint=EFFICIENCY_GATE)
    above = shaped_terms(w, plates, _perfect(w), usd_spent=1041.0,
                         budget_usd=6000.0, endpoint=EFFICIENCY_GATE + 1e-9)
    unspent = 1.0 - 1041.0 / 6000.0
    print(f"[measured] efficiency at endpoint = {EFFICIENCY_GATE}: {below['efficiency']:.6f}; "
          f"at {EFFICIENCY_GATE} + 1e-9: {above['efficiency']:.6f} "
          f"(unspent fraction = {unspent:.6f})")
    assert below["efficiency"] == 0.0            # the gate is strict `>`
    assert above["efficiency"] == pytest.approx(unspent)

    # Spending nothing pays the full fraction, but only once the answer is good.
    assert shaped_terms(w, [], _perfect(w), 0.0, 6000.0, 1.0)["efficiency"] == 1.0
    assert shaped_terms(w, [], _perfect(w), 0.0, 6000.0, 0.0)["efficiency"] == 0.0


def test_one_sign_flip_straddles_the_efficiency_gate():
    """The gate bites on a real submission, not just on a hand-passed number.

    Same plates, same spend, one character of difference in the answer: one
    correct hit with the right sign scores endpoint 0.425 and unlocks
    efficiency; with the wrong sign it scores 0.275 and does not.
    """
    w = _world(3, "clean")
    plates = [_plate("P1", _good_layout(w.genes))]
    gene = w.true_hits[0]
    right = {"hits": [gene], "signs": {gene: w.true_signs[gene]}, "log_ec50": None}
    wrong = {"hits": [gene], "signs": {gene: -w.true_signs[gene]}, "log_ec50": None}

    ep_r = endpoint_terms(w, right)["endpoint"]
    ep_w = endpoint_terms(w, wrong)["endpoint"]
    eff_r = shaped_terms(w, plates, right, 1041.0, 6000.0, ep_r)["efficiency"]
    eff_w = shaped_terms(w, plates, wrong, 1041.0, 6000.0, ep_w)["efficiency"]

    print(f"[measured] one hit, right sign: endpoint = {ep_r:.4f} -> efficiency "
          f"{eff_r:.4f}; wrong sign: endpoint = {ep_w:.4f} -> efficiency {eff_w:.4f}")
    assert ep_r == pytest.approx(0.55 * 0.5 + 0.15)   # 0.425, analytic
    assert ep_w == pytest.approx(0.55 * 0.5)          # 0.275, analytic
    assert eff_w == 0.0 and eff_r > 0.0


def test_running_nothing_is_not_reward_optimal():
    """Without the gate, banking the budget would beat doing the experiment."""
    w = _world(3, "clean")
    layout = _good_layout(w.genes)
    plates = [_plate(f"P{i}", layout) for i in range(1, 4)]

    do_nothing = shaped_terms(w, [], {"hits": [], "signs": {}, "log_ec50": None},
                              0.0, 6000.0, 0.0)
    did_the_work = shaped_terms(w, plates, _perfect(w), 3 * 1041.0, 6000.0, 1.0)
    print(f"[measured] shaped: do-nothing = {do_nothing['shaped']:.4f}, "
          f"three plates + perfect answer = {did_the_work['shaped']:.4f}")
    assert did_the_work["shaped"] > do_nothing["shaped"]
    assert do_nothing["shaped"] == 0.0


# --------------------------------------------------------------------------
# qc_hygiene — two genuinely different code paths
# --------------------------------------------------------------------------


def test_qc_hygiene_with_a_bad_lot_present():
    """The first term is the fraction of bad-lot plates the agent excluded."""
    w = _world(3, "clean")
    w.bad_lots = ["LOT-B"]  # clean never generates one; impose it
    layout = _good_layout(w.genes)

    def hyg(excluded_ids):
        plates = [
            _plate("P1", layout, "LOT-A", "P1" in excluded_ids),
            _plate("P2", layout, "LOT-B", "P2" in excluded_ids),  # the bad lot
            _plate("P3", layout, "LOT-C", "P3" in excluded_ids),
        ]
        return shaped_terms(w, plates, _perfect(w), 3123.0, 6000.0, 1.0)["qc_hygiene"]

    rows = [
        (set(), 0.0, "missed the bad lot entirely"),
        ({"P2"}, 1.0, "caught it, excluded nothing else"),
        ({"P2", "P1"}, 0.5, "caught it but also dropped a good plate"),
        ({"P1", "P2", "P3"}, 0.0, "excluded everything"),
        ({"P1", "P3"}, 0.0, "excluded the two good plates, kept the bad one"),
    ]
    print("[measured] qc_hygiene, bad lot = LOT-B, 3 plates (one per lot):")
    for excluded, expected, why in rows:
        got = hyg(excluded)
        label = ", ".join(sorted(excluded)) or "nothing"
        print(f"             exclude {label:<14} -> {got:.4f}  ({why})")
        assert got == pytest.approx(expected), (excluded, why)


def test_qc_hygiene_with_no_bad_lot_is_a_pure_over_exclusion_penalty():
    """No bad lot: the caught term is 1.0 and only over-excluding can cost you."""
    env = _run_env(seed=5, tier="clean", n_plates=3)
    assert env.world.bad_lots == []  # clean has p_bad_lot = 0

    def hyg():
        return score(env)["shaped_terms"]["qc_hygiene"]

    env.submit(**_perfect(env.world))
    baseline = hyg()
    print(f"[measured] qc_hygiene, no bad lot, 3 good plates, 0 excluded = {baseline:.4f}")
    assert baseline == 1.0

    # Exclude two arbitrarily. The metric must drop.
    env.done = False  # exclusion after submit is refused; this is a scoring test
    env.exclude_plate("P1", "no reason")
    env.exclude_plate("P2", "no reason")
    env.done = True
    dropped = hyg()
    print(f"[measured] qc_hygiene after excluding 2 of 3 good plates = {dropped:.4f} "
          f"(analytic 1 - 2/3 = {1 - 2 / 3:.4f})")
    assert dropped == pytest.approx(1.0 - 2.0 / 3.0)
    assert dropped < baseline

    # Exclude all three and it clips at zero rather than going negative.
    env.done = False
    env.exclude_plate("P3", "no reason")
    env.done = True
    assert hyg() == 0.0


def test_qc_hygiene_when_the_bad_lot_was_never_used():
    """A bad lot exists but no plate ran on it: nothing to catch, so 1.0."""
    w = _world(3, "clean")
    w.bad_lots = ["LOT-C"]
    layout = _good_layout(w.genes)
    plates = [_plate("P1", layout, "LOT-A"), _plate("P2", layout, "LOT-B")]
    got = shaped_terms(w, plates, _perfect(w), 2082.0, 6000.0, 1.0)["qc_hygiene"]
    print(f"[measured] qc_hygiene, bad lot exists but unused = {got:.4f}")
    assert got == 1.0


# --------------------------------------------------------------------------
# controls, replication, self_normalizable
# --------------------------------------------------------------------------


def test_controls_counts_wells_on_non_excluded_plates():
    w = _world(3, "clean")
    full = _good_layout(w.genes, n_ntc=4, n_pos=2)     # exactly at threshold
    thin_ntc = _good_layout(w.genes, n_ntc=3, n_pos=2)
    thin_pos = _good_layout(w.genes, n_ntc=4, n_pos=1)

    def controls(plates):
        return shaped_terms(w, plates, _perfect(w), 0.0, 6000.0, 1.0)["controls"]

    assert controls([_plate("P1", full)]) == 1.0
    assert controls([_plate("P1", thin_ntc)]) == 0.0   # 3 NTC is not enough
    assert controls([_plate("P1", thin_pos)]) == 0.0   # 1 POS is not enough
    assert controls([_plate("P1", full), _plate("P2", thin_ntc)]) == pytest.approx(0.5)

    # Excluded plates leave the denominator.
    assert controls([_plate("P1", full),
                     _plate("P2", thin_ntc, excluded=True)]) == 1.0
    assert controls([]) == 0.0
    print("[measured] controls: >=4 NTC and >=2 POS, over non-excluded plates only")


def test_replication_counts_submitted_hits_on_distinct_plates():
    """Not conditions run, and not wells — distinct non-excluded plates."""
    w = _world(3, "clean")
    a, b, c = w.true_hits[0], w.true_hits[1], w.true_hits[2]
    other = [g for g in w.genes if g not in w.true_hits][0]

    on_two = {"B2": f"KD:{a}", "B3": "NTC"}
    twice_one_plate = {"B4": f"KD:{b}", "B5": f"KD:{b}", "B6": "NTC"}
    elsewhere = {"B7": f"KD:{other}", "B8": f"KD:{other}", "B9": "NTC"}

    plates = [
        _plate("P1", {**on_two, **twice_one_plate, **elsewhere}),
        _plate("P2", {**on_two, **elsewhere}),
    ]
    sub = {"hits": [a, b, c], "signs": {}, "log_ec50": None}

    def rep(plates, submission):
        return shaped_terms(w, plates, submission, 0.0, 6000.0, 1.0)["replication"]

    got = rep(plates, sub)
    print(f"[measured] replication: {a} on 2 plates, {b} twice on 1 plate, "
          f"{c} never measured, {other} on 2 plates but not submitted -> {got:.4f} "
          f"(analytic 1/3 = {1 / 3:.4f})")
    assert got == pytest.approx(1 / 3)

    # Replicating everything and calling the wrong genes earns nothing.
    assert rep(plates, {"hits": [c], "signs": {}, "log_ec50": None}) == 0.0
    # A gene replicated only on excluded plates does not count.
    excluded = [_plate("P1", on_two), _plate("P2", on_two, excluded=True)]
    assert rep(excluded, {"hits": [a], "signs": {}, "log_ec50": None}) == 0.0
    # No submitted hits at all.
    assert rep(plates, {"hits": [], "signs": {}, "log_ec50": None}) == 0.0


def test_self_normalizable_needs_an_ntc_and_a_test_condition():
    w = _world(3, "clean")

    def sn(plates):
        return shaped_terms(w, plates, _perfect(w), 0.0, 6000.0, 1.0)["self_normalizable"]

    gene = w.genes[0]
    assert sn([_plate("P1", {"B2": "NTC", "B3": f"KD:{gene}"})]) == 1.0
    assert sn([_plate("P1", {"B2": "NTC", "B3": "CMPD@100"})]) == 1.0
    assert sn([_plate("P1", {"B2": f"KD:{gene}", "B3": "POS"})]) == 0.0  # no NTC
    assert sn([_plate("P1", {"B2": "NTC", "B3": "POS"})]) == 0.0        # nothing to normalise
    assert sn([_plate("P1", {"B2": "NTC", "B3": f"KD:{gene}"}),
               _plate("P2", {"B2": "POS"})]) == pytest.approx(0.5)
    assert sn([]) == 0.0

    # Counted over EVERY plate: excluding a badly-laid-out plate afterwards does
    # not launder a design that could never have been normalised against itself.
    both = [_plate("P1", {"B2": "NTC", "B3": f"KD:{gene}"}),
            _plate("P2", {"B2": "POS"}, excluded=True)]
    assert sn(both) == pytest.approx(0.5)
    print("[measured] self_normalizable: >=1 NTC plus >=1 test condition, "
          "over every plate including excluded ones")


# --------------------------------------------------------------------------
# 4.4 Diagnostics — reported, never scored
# --------------------------------------------------------------------------


def test_decoy_called_measures_prior_dependence_without_changing_reward():
    """A decoy and a plain null are the same false positive to the reward."""
    w = _world(11, "hard")
    assert w.decoys, "hard tier always springs the trap"
    decoy = w.decoys[0]
    null = [g for g in w.genes
            if g not in w.true_hits and g not in w.decoys and g not in w.gray_zone][0]

    with_decoy = dict(_perfect(w), hits=list(w.true_hits) + [decoy])
    with_null = dict(_perfect(w), hits=list(w.true_hits) + [null])

    a = score_trajectory(w, [], with_decoy, 0.0, 3300.0)
    b = score_trajectory(w, [], with_null, 0.0, 3300.0)

    print(f"[measured] hard seed 11: decoys = {w.decoys}, omitted = {w.omitted}")
    print(f"           +1 decoy   -> endpoint {a['endpoint']:.4f}, "
          f"decoy_called = {a['diagnostics']['decoy_called']}")
    print(f"           +1 null    -> endpoint {b['endpoint']:.4f}, "
          f"decoy_called = {b['diagnostics']['decoy_called']}")
    assert a["endpoint"] == b["endpoint"]
    assert a["shaped"] == b["shaped"]
    assert a["strict_pass"] == b["strict_pass"] == 0.0
    assert a["diagnostics"]["decoy_called"] == 1
    assert b["diagnostics"]["decoy_called"] == 0


def test_diagnostics_fields():
    w = _world(11, "hard")
    assert w.omitted
    layout = _good_layout(w.genes)
    plates = [_plate("P1", layout), _plate("P2", layout, excluded=True)]
    sub = dict(_perfect(w), hits=list(w.true_hits) + list(w.decoys))

    d = diagnostics(w, plates, sub, usd_spent=2082.0)
    n_sub = len(set(sub["hits"]))
    print(f"[measured] diagnostics, hard seed 11: {d}")
    assert d["precision"] == pytest.approx(len(w.true_hits) / n_sub)
    assert d["recall"] == 1.0
    assert d["decoy_called"] == len(w.decoys)
    assert d["omitted_recovered"] == len(w.omitted)  # submitted despite the omission
    assert d["prior_trap"] == 1
    assert d["n_plates"] == 2 and d["n_excluded"] == 1
    assert d["usd_spent"] == 2082.0

    clean = _world(3, "clean")
    assert diagnostics(clean, [], _perfect(clean), 0.0)["prior_trap"] == 0


# --------------------------------------------------------------------------
# Shape and determinism
# --------------------------------------------------------------------------


def test_score_is_deterministic_and_json_shaped():
    env = _run_env(13, "standard", n_plates=2)
    env.exclude_plate("P1", "window collapsed")
    env.submit(**_perfect(env.world, 0.15))
    assert score(env) == score(env)

    res = score(env)
    assert set(res) == {"strict_pass", "endpoint", "shaped", "endpoint_terms",
                        "shaped_terms", "diagnostics"}
    assert set(res["endpoint_terms"]) == set(ENDPOINT_WEIGHTS)
    assert set(res["shaped_terms"]) == set(SHAPED_WEIGHTS)
    for block in ("endpoint_terms", "shaped_terms"):
        for k, v in res[block].items():
            assert isinstance(v, float) and np.isfinite(v), (block, k, v)
            assert 0.0 <= v <= 1.0, (block, k, v)
    for k in ("strict_pass", "endpoint", "shaped"):
        assert 0.0 <= res[k] <= 1.0

    import json
    json.dumps(res)  # must survive the wire for Phase 6

    with pytest.raises(RuntimeError):
        score(AssayGym(1, "clean"))
