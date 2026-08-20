"""Phase 5 acceptance checks for assaygym/policies.py, run_baselines.py, verify.py.

Run: ./.venv/bin/python -m pytest tests/ -q -s   (see CONTRIBUTING.md)

Phase 5 is the gate. The acceptance criterion is the ledger itself: four scripted
policies over 200 seeded episodes per tier, checked against the BUILD_SPEC table
with a +/-0.05 tolerance, monotone in every tier, with `prior_parrot` near zero
and the "call every gene a hit" exploit scoring exactly zero.

Every episode is fully seeded, so these are not sampling estimates that might
flake -- they are fixed numbers. A change in any of them means a change in the
environment, not a bad roll.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assaygym.assay import WELLS, is_edge, quadrant  # noqa: E402
from assaygym.env import AssayGym, plate_cost  # noqa: E402
from assaygym.policies import (  # noqa: E402
    ABLATIONS,
    DOSES,
    INTERIOR,
    POLICIES,
    POLICY_RNG_OFFSET,
    PlateView,
    call_everything_policy,
    competent_doe_policy,
    run_episode,
    run_policy,
)
from assaygym.policies import (  # noqa: E402
    _balanced_layout,
    _fit_log_ec50,
    _flag_contaminated,
    _plate_effects,
)
from assaygym.world import TIERS  # noqa: E402

N = 200
TOL = 0.05

# BUILD_SPEC "Phase 5 acceptance -- the numbers to hit", strict_pass.
TARGETS = {
    "clean":    {"random": 0.000, "prior_parrot": 0.040,
                 "naive_screen": 0.275, "competent_doe": 1.000},
    "standard": {"random": 0.000, "prior_parrot": 0.015,
                 "naive_screen": 0.075, "competent_doe": 0.620},
    "hard":     {"random": 0.000, "prior_parrot": 0.000,
                 "naive_screen": 0.005, "competent_doe": 0.165},
}
ORDER = ["random", "prior_parrot", "naive_screen", "competent_doe"]


def _genes(n):
    return [f"SYN{i + 1:02d}" for i in range(n)]


def _mean_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


@pytest.fixture(scope="module")
def ledger():
    """strict_pass for every (tier, policy) cell. Computed once."""
    out = {}
    for tier in TARGETS:
        out[tier] = {}
        for name in ORDER:
            res = run_policy(POLICIES[name], n=N, tier=tier)
            out[tier][name] = _mean_se([r["strict_pass"] for r in res])
    return out


# --------------------------------------------------------------------------
# 5.1 Shared helpers
# --------------------------------------------------------------------------


def test_interior_and_doses():
    assert len(INTERIOR) == 60 and len(set(INTERIOR)) == 60
    assert not any(is_edge(w) for w in INTERIOR)
    counts = {q: 0 for q in range(4)}
    for w in INTERIOR:
        counts[quadrant(w)] += 1
    assert counts == {0: 15, 1: 15, 2: 15, 3: 15}

    # The dose ladder must bracket the whole EC50 range (10^0.5 to 10^3.5 nM)
    # INCLUDING the top, or the plateau cannot be estimated and every fit is
    # biased low.
    assert min(DOSES) <= 10 ** 0.5 and max(DOSES) >= 10 ** 3.5
    print(f"\n[measured] dose ladder spans log10 "
          f"{np.log10(min(DOSES)):.2f} to {np.log10(max(DOSES)):.2f}; "
          f"true_log_ec50 range is 0.50 to 3.50")


def test_balanced_layout_structure_and_budget():
    rng = np.random.default_rng(0)
    for tier, diff in TIERS.items():
        genes = _genes(diff.n_genes)
        layout = _balanced_layout(genes, reps=2, rng=rng)
        conds = list(layout.values())

        assert all(w in INTERIOR for w in layout)      # never the perimeter
        assert conds.count("POS") == 3
        for g in genes:
            assert conds.count(f"KD:{g}") == 2
        for d in DOSES:
            assert conds.count(f"CMPD@{d}") == 2

        # Two NTC in every quadrant. Controls bunched in one corner leave the
        # agent unable to tell a blown quadrant from a real effect.
        per_q = {q: 0 for q in range(4)}
        for w, c in layout.items():
            if c == "NTC":
                per_q[quadrant(w)] += 1
        assert per_q == {0: 2, 1: 2, 2: 2, 3: 2}

        cost = plate_cost(len(layout))
        print(f"[measured] {tier:9s} balanced plate = {len(layout)} wells, "
              f"${cost:,.0f}; 3 plates = ${3 * cost:,.0f} of ${diff.budget_usd:,.0f} "
              f"and 9 of {diff.budget_days} days")
        assert 3 * cost <= diff.budget_usd, tier   # three plates must fit
        assert 9 <= diff.budget_days, tier

    # Hard is the tight one: exactly the $1,041 plate the tier buys three of.
    hard = _balanced_layout(_genes(12), reps=2, rng=rng)
    assert len(hard) == 51 and plate_cost(51) == 1041.0


def test_balanced_layout_truncates_rather_than_overflowing():
    rng = np.random.default_rng(0)
    layout = _balanced_layout(_genes(60), reps=2, rng=rng)
    assert len(layout) <= len(INTERIOR)
    assert list(layout.values()).count("NTC") == 8


def test_balanced_layout_can_use_the_whole_plate():
    """The perimeter ablation draws from all 96 wells at the same well count."""
    rng = np.random.default_rng(0)
    full = _balanced_layout(_genes(12), reps=2, rng=rng, wells=WELLS)
    assert len(full) == 51                      # same cost as the interior plate
    assert any(is_edge(w) for w in full)        # and now it catches the edge


def test_fit_log_ec50_recovers_a_known_curve():
    """Half-maximal interpolation, both signs, noiseless and noisy."""
    for true_log, hill, delta in [(1.0, 1.0, 0.5), (2.0, 1.5, 0.5),
                                  (3.0, 1.2, -0.5), (2.4, 0.9, -0.4)]:
        ec50 = 10 ** true_log
        effects = {
            float(d): [delta / (1 + (ec50 / d) ** hill)] for d in DOSES
        }
        got = _fit_log_ec50(effects)
        print(f"[measured] noiseless fit: true {true_log:.2f} -> {got:.4f} "
              f"(err {abs(got - true_log):.4f}, hill {hill}, delta {delta:+.1f})")
        assert abs(got - true_log) < 0.25

    # Noisy, replicated: still inside the 0.4 scoring tolerance.
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(200):
        true_log = float(rng.uniform(0.5, 3.5))
        ec50, hill = 10 ** true_log, float(rng.uniform(0.9, 2.0))
        effects = {
            float(d): [0.5 / (1 + (ec50 / d) ** hill) + rng.normal(0, 0.05)
                       for _ in range(6)]
            for d in DOSES
        }
        errs.append(abs(_fit_log_ec50(effects) - true_log))
    within = float(np.mean([e <= 0.4 for e in errs]))
    print(f"[measured] noisy fit (sd 0.05, 6 reps, 200 curves): "
          f"median |err| = {np.median(errs):.4f}, within 0.4 = {within:.3f}")
    assert within > 0.85

    assert _fit_log_ec50({}) is None
    assert _fit_log_ec50({1.0: [0.1], 10.0: [0.2]}) is None      # too few doses
    assert _fit_log_ec50({float(d): [0.0] for d in DOSES}) is None  # flat


def test_flag_contaminated():
    layout = {w: ("NTC" if i % 5 == 0 else "POS")
              for i, w in enumerate(INTERIOR)}
    clean = {w: 1.0 for w in layout}
    assert _flag_contaminated(PlateView("P", "LOT-A", layout, clean)) == set()

    for target in range(4):
        blown = {w: 1.0 + (0.45 if quadrant(w) == target else 0.0) for w in layout}
        assert _flag_contaminated(
            PlateView("P", "LOT-A", layout, blown)) == {target}

    # A real effect (0.20) must not trip the 0.25 floor.
    mild = {w: 1.0 + (0.20 if quadrant(w) == 1 else 0.0) for w in layout}
    assert _flag_contaminated(PlateView("P", "LOT-A", layout, mild)) == set()
    print("[measured] contamination flag: 0.45 offset detected in all 4 "
          "quadrants; a 0.20 real effect is not flagged")


def test_plate_effects_subtracts_the_plates_own_ntc_median():
    layout = {w: ("NTC" if i < 8 else "POS") for i, w in enumerate(INTERIOR)}
    values = {w: (2.0 if layout[w] == "NTC" else 5.0) for w in layout}
    eff = _plate_effects(PlateView("P", "LOT-A", layout, values))
    assert all(eff[w] == 0.0 for w in layout if layout[w] == "NTC")
    assert all(eff[w] == 3.0 for w in layout if layout[w] == "POS")

    dropped = _plate_effects(PlateView("P", "LOT-A", layout, values),
                             drop_quadrants=[1])
    assert all(quadrant(w) != 1 for w in dropped)
    assert len(dropped) < len(eff)


# --------------------------------------------------------------------------
# Three independent rngs
# --------------------------------------------------------------------------


def test_every_policy_faces_the_identical_world_and_noise():
    """Policy randomness must not perturb the world or the measurement noise."""
    worlds = {}
    for name, policy in POLICIES.items():
        env = AssayGym(41, "hard")
        briefing = env.reset()
        policy(env, briefing, np.random.default_rng(41 + POLICY_RNG_OFFSET))
        worlds[name] = (tuple(env.world.true_hits), env.world.true_log_ec50,
                        tuple(sorted(env.world.lot_potency.items())),
                        tuple(env.world.reported_hits))
    assert len(set(worlds.values())) == 1, worlds
    print(f"[measured] all 4 policies at seed 41/hard face true_hits="
          f"{list(worlds['random'][0])}, log_ec50={worlds['random'][1]:.4f}")


def test_competent_doe_spends_the_hard_budget_exactly():
    env = AssayGym(3, "hard")
    briefing = env.reset()
    competent_doe_policy(env, briefing,
                         np.random.default_rng(3 + POLICY_RNG_OFFSET))
    print(f"[measured] competent_doe on hard: {len(env.plates)} plates, "
          f"${env.usd_spent:,.0f} of $3,300, {env.days_used} of 9 days, "
          f"${env.usd_left:,.0f} left")
    assert len(env.plates) == 3
    assert env.usd_spent == 3123.0 and env.days_used == 9
    assert env.usd_left == 177.0 and env.days_left == 0


# --------------------------------------------------------------------------
# verify.py check 1 — determinism
# --------------------------------------------------------------------------


def test_episodes_are_deterministic():
    for tier in ("clean", "standard", "hard"):
        for name, policy in POLICIES.items():
            for seed in (0, 7):
                assert run_episode(policy, seed, tier) == run_episode(
                    policy, seed, tier), (tier, name, seed)
    print("[measured] 24 (tier, policy, seed) combinations replayed identically")


# --------------------------------------------------------------------------
# verify.py check 2 — the degenerate exploit
# --------------------------------------------------------------------------


def test_calling_every_gene_a_hit_does_not_win():
    """The classic way to farm recall. It must score exactly zero."""
    for tier in ("clean", "standard", "hard"):
        res = run_policy(call_everything_policy, n=N, tier=tier)
        sp, _ = _mean_se([r["strict_pass"] for r in res])
        ep, ep_se = _mean_se([r["endpoint"] for r in res])
        recall = np.mean([r["diagnostics"]["recall"] for r in res])
        doe = np.mean([r["endpoint"] for r in run_policy(
            POLICIES["competent_doe"], n=N, tier=tier)])
        print(f"[measured] call-everything {tier:9s}: strict_pass {sp:.3f}, "
              f"endpoint {ep:.3f} +/- {ep_se:.3f}, recall {recall:.3f} "
              f"(competent_doe endpoint {doe:.3f})")
        assert sp == 0.000, tier
        assert recall == 1.0                # recall IS farmed
        assert ep < doe - 0.3               # and it buys nothing


# --------------------------------------------------------------------------
# verify.py check 3 — the ledger. This is the gate.
# --------------------------------------------------------------------------


def test_ledger_matches_the_acceptance_table(ledger):
    print(f"\n[measured] strict_pass ledger, n = {N} per cell:")
    worst = 0.0
    for tier in ("clean", "standard", "hard"):
        cells = "  ".join(
            f"{n}={ledger[tier][n][0]:.3f}+/-{ledger[tier][n][1]:.3f}"
            for n in ORDER)
        print(f"             {tier:9s} {cells}")
        for name in ORDER:
            got = ledger[tier][name][0]
            target = TARGETS[tier][name]
            worst = max(worst, abs(got - target))
            assert abs(got - target) <= TOL, (tier, name, got, target)
    print(f"             worst deviation from the BUILD_SPEC table: {worst:.3f} "
          f"(tolerance {TOL})")


def test_ladder_is_monotone_in_every_tier(ledger):
    """More real experimental design must mean a higher score. Everywhere."""
    for tier in ("clean", "standard", "hard"):
        means = [ledger[tier][n][0] for n in ORDER]
        assert all(means[i] <= means[i + 1] for i in range(len(means) - 1)), (
            tier, dict(zip(ORDER, means)))
    print("[measured] monotone random <= prior_parrot <= naive_screen <= "
          "competent_doe on all three tiers")


def test_prior_parrot_is_near_zero(ledger):
    """If this scores well above 0.05 the trap is not working.

    The two bugs BUILD_SPEC names as the causes are skewed hit signs and a
    narrow EC50 range; both are pinned by tests/test_world.py.
    """
    for tier in ("clean", "standard", "hard"):
        got = ledger[tier]["prior_parrot"][0]
        print(f"[measured] prior_parrot {tier:9s} strict_pass = {got:.3f}")
        assert got <= 0.05, (tier, got)


def test_clean_competent_doe_is_exactly_one(ledger):
    """The most important cell, and the one people skip.

    It proves the task is well-posed: strip the artifacts and a correct policy
    solves it every single time. Without this cell, a low hard-tier score might
    just mean the environment is broken.
    """
    got, se = ledger["clean"]["competent_doe"]
    print(f"[measured] clean/competent_doe = {got:.3f} +/- {se:.3f} over {N} seeds")
    assert got == 1.000


def test_prior_dependence_is_measurable_and_falls_with_competence():
    """decoy_called is the number the environment exists to produce."""
    print("[measured] fraction of episodes calling >=1 decoy:")
    for tier in ("standard", "hard"):
        rates = {}
        for name in ORDER:
            res = run_policy(POLICIES[name], n=N, tier=tier)
            rates[name] = float(np.mean(
                [r["diagnostics"]["decoy_called"] > 0 for r in res]))
        print(f"             {tier:9s} " + "  ".join(
            f"{k}={v:.3f}" for k, v in rates.items()))
        # The trap catches the parrot hardest and the reference policy least.
        assert rates["prior_parrot"] > rates["competent_doe"]
    # Even the best scripted policy still falls for it often -- that is headroom.
    hard = float(np.mean([r["diagnostics"]["decoy_called"] > 0
                          for r in run_policy(POLICIES["competent_doe"], N, "hard")]))
    assert 0.3 < hard < 0.7


# --------------------------------------------------------------------------
# Ablations
# --------------------------------------------------------------------------


def test_ablations_are_wired_and_full_matches_the_reference():
    assert set(ABLATIONS) == {
        "full", "no_lot_exclusion", "no_ntc_normalisation",
        "no_contamination_flag", "one_replicate", "full_plate_with_edges",
        "no_qc_at_all",
    }
    a = run_policy(ABLATIONS["full"], n=40, tier="standard")
    b = run_policy(POLICIES["competent_doe"], n=40, tier="standard")
    assert a == b, "the 'full' ablation must BE the reference policy"


def test_replication_is_the_defence_that_earns_the_most():
    """Dropping to one replicate is the single most costly ablation.

    The full table lives in run_baselines.py --ablate; this pins the one result
    that is far outside the error bars on both tiers, so a regression that
    quietly makes replication worthless would fail here.
    """
    for tier in ("standard", "hard"):
        full = np.array([r["strict_pass"]
                         for r in run_policy(ABLATIONS["full"], N, tier)])
        one = np.array([r["strict_pass"]
                        for r in run_policy(ABLATIONS["one_replicate"], N, tier)])
        delta = full.mean() - one.mean()
        se = float(np.hypot(full.std(ddof=1), one.std(ddof=1)) / np.sqrt(N))
        print(f"[measured] {tier:9s} one_replicate costs {delta:+.3f} +/- {se:.3f} "
              f"({delta / se:.1f} SE): {full.mean():.3f} -> {one.mean():.3f}")
        assert delta > 2 * se, (tier, delta, se)
