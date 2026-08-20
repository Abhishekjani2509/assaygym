"""Phase 3 acceptance checks for assaygym/env.py.

Run: ./.venv/bin/python -m pytest tests/ -q -s   (see CONTRIBUTING.md)

Four acceptance checks from BUILD_SPEC section "Phase 3":
  1. a 51-well plate costs exactly $1,041
  2. on hard, three 51-well plates succeed and the fourth is refused for BOTH
     money and days
  3. qc() changes neither budget
  4. any tool called after submit() errors rather than mutating state

Plus the properties those four do not pin down: that the two rngs are actually
separate, that the literature-prior caveat is present, and that a bad plate id
fails loudly instead of silently.
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assaygym.assay import WELLS, run_plate, z_prime  # noqa: E402
from assaygym.env import (  # noqa: E402
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
from assaygym.world import (  # noqa: E402
    TIERS,
    override_phenotype_from_deltas,
    sample_world,
)

# Interior wells, no perimeter: rows B-G x columns 2-11 = 60 wells.
INTERIOR = [f"{r}{c}" for r in "BCDEFG" for c in range(2, 12)]


def _layout(n=51, condition="NTC", wells=None):
    wells = wells or INTERIOR
    return {w: condition for w in wells[:n]}


def _mixed_layout(env, n=51):
    """A layout with real controls on it, so qc() has something to chew on."""
    genes = env.world.genes
    conds = ["NTC"] * 8 + ["POS"] * 4 + [f"KD:{g}" for g in genes] + ["CMPD@100"]
    return {w: conds[i % len(conds)] for i, w in enumerate(INTERIOR[:n])}


def _budget_state(env):
    """Everything a tool call could legitimately move."""
    return {
        "usd_left": env.usd_left,
        "days_left": env.days_left,
        "days_used": env.days_used,
        "plate_ids": list(env.plates),
        "excluded": {k: p.excluded for k, p in env.plates.items()},
        "values": {k: dict(p.values) for k, p in env.plates.items()},
        "done": env.done,
        "submission": copy.deepcopy(env.submission),
        "contaminated": dict(env.world.contaminated_plates),
        "rng": env.rng.bit_generator.state,
    }


# --------------------------------------------------------------------------
# Acceptance 1 — plate cost arithmetic
# --------------------------------------------------------------------------


def test_51_well_plate_costs_exactly_1041():
    """480 + 11 x 51 = 1041, and the env charges exactly that."""
    assert PLATE_BASE_COST == 480.0
    assert PER_WELL_COST == 11.0
    assert PLATE_DAYS == 3
    assert plate_cost(51) == 1041.0  # exact, not approx

    env = AssayGym(1, "hard")
    env.reset()
    res = env.design_and_run(_layout(51), "LOT-A")
    print(f"\n[measured] 51-well plate: cost = ${res['cost_usd']:,.2f}, "
          f"n_wells = {res['n_wells']}, days = {PLATE_DAYS}")
    assert res["cost_usd"] == 1041.0
    assert env.usd_left == 3300.0 - 1041.0 == 2259.0
    assert env.days_left == 9 - 3

    # The formula, not a coincidence at 51.
    for n in (1, 24, 51, 60, 96):
        assert plate_cost(n) == 480.0 + 11.0 * n


def test_cost_scales_with_filled_wells_only():
    env = AssayGym(1, "clean")
    env.reset()
    a = env.design_and_run(_layout(10), "LOT-A")
    b = env.design_and_run(_layout(20), "LOT-A")
    assert b["cost_usd"] - a["cost_usd"] == 10 * PER_WELL_COST
    assert env.usd_spent == a["cost_usd"] + b["cost_usd"]
    assert env.usd_left == 6000.0 - env.usd_spent


# --------------------------------------------------------------------------
# Acceptance 2 — hard tier buys exactly three plates
# --------------------------------------------------------------------------


def test_hard_tier_allows_exactly_three_51_well_plates():
    """$3,300 / 9 days buys three 51-well plates. The fourth fails on BOTH."""
    diff = TIERS["hard"]
    assert diff.budget_usd == 3300.0 and diff.budget_days == 9

    env = AssayGym(11, "hard")
    env.reset()
    layout = _layout(51)

    for i in range(3):
        res = env.design_and_run(layout, LOTS[i])
        assert "error" not in res, (i, res)
        print(f"[measured] hard plate {i + 1}: cost ${res['cost_usd']:,.0f}, "
              f"usd_left ${res['usd_left']:,.0f}, days_left {res['days_left']}")

    assert env.usd_left == 3300.0 - 3 * 1041.0 == 177.0
    assert env.days_left == 0
    assert len(env.plates) == 3

    before = _budget_state(env)
    refused = env.design_and_run(layout, "LOT-A")
    print(f"[measured] hard plate 4 refused: {refused['error']}")

    assert "error" in refused
    # Both constraints are checked, independently, and both are reported.
    assert len(refused["reasons"]) == 2, refused["reasons"]
    money = [r for r in refused["reasons"] if "funds" in r]
    days = [r for r in refused["reasons"] if "days" in r]
    assert len(money) == 1 and len(days) == 1, refused["reasons"]
    assert refused["cost_usd"] == 1041.0 and refused["usd_left"] == 177.0
    assert refused["days_required"] == 3 and refused["days_left"] == 0

    # A refusal is free: no plate, no charge, and no assay-noise draw consumed.
    assert _budget_state(env) == before
    assert len(env.plates) == 3


def test_money_and_days_are_checked_independently():
    """One constraint can bind without the other. Both cases must be caught."""
    # Money-only: clean is 6000 / 18 days. Five 51-well plates cost $5,205 and
    # take 15 days, leaving $795 (not enough) and 3 days (enough).
    env = AssayGym(2, "clean")
    env.reset()
    for _ in range(5):
        assert "error" not in env.design_and_run(_layout(51), "LOT-A")
    assert env.usd_left == 795.0 and env.days_left == 3
    refused = env.design_and_run(_layout(51), "LOT-A")
    print(f"[measured] money-only refusal at ${env.usd_left:,.0f} / "
          f"{env.days_left} days: {refused['reasons']}")
    assert len(refused["reasons"]) == 1
    assert "funds" in refused["reasons"][0] and "days" not in refused["reasons"][0]

    # Days-only: six 1-well plates cost $2,946 of $6,000 but consume all 18 days.
    env = AssayGym(2, "clean")
    env.reset()
    for _ in range(6):
        assert "error" not in env.design_and_run(_layout(1), "LOT-A")
    assert env.days_left == 0 and env.usd_left == 6000.0 - 6 * 491.0
    refused = env.design_and_run(_layout(1), "LOT-A")
    print(f"[measured] days-only refusal at ${env.usd_left:,.0f} / "
          f"{env.days_left} days: {refused['reasons']}")
    assert len(refused["reasons"]) == 1
    assert "days" in refused["reasons"][0] and "funds" not in refused["reasons"][0]


def test_days_used_advances_and_stamps_every_plate():
    """days_used is the campaign clock: +3 per plate, and it stamps day_run.

    Found by tools/mutate.py: pinning only that a *refusal* leaves days_used
    alone never checked that a successful plate advances it, so freezing the
    clock at 0 passed the whole suite while making every plate look
    simultaneous.
    """
    env = AssayGym(12, "hard")
    env.reset()
    assert env.days_used == 0
    for i in range(3):
        res = env.design_and_run(_layout(20), "LOT-A")
        assert res["day_run"] == 3 * i, res
        assert env.plates[res["plate_id"]].day_run == 3 * i
        assert env.days_used == 3 * (i + 1)
        assert env.days_used + env.days_left == 9  # the clock conserves
    print(f"[measured] campaign clock: day_run stamps = "
          f"{[p.day_run for p in env.plates.values()]}, "
          f"days_used = {env.days_used}, days_left = {env.days_left}")


def test_budget_boundary_is_inclusive():
    """A plate you can exactly afford is allowed; one dollar more is not."""
    env = AssayGym(3, "clean")
    env.reset()
    env.usd_left = plate_cost(20)  # exactly enough
    assert "error" not in env.design_and_run(_layout(20), "LOT-A")
    assert env.usd_left == 0.0

    env = AssayGym(3, "clean")
    env.reset()
    env.usd_left = plate_cost(20) - 0.01
    res = env.design_and_run(_layout(20), "LOT-A")
    assert "error" in res and len(env.plates) == 0


# --------------------------------------------------------------------------
# Acceptance 3 — qc() is free
# --------------------------------------------------------------------------


def test_qc_costs_no_money_and_no_days():
    """Free on purpose: skipping QC must be a judgment failure, not a budget one."""
    env = AssayGym(5, "standard")
    env.reset()
    env.design_and_run(_mixed_layout(env, 51), "LOT-A")

    before = _budget_state(env)
    reports = [env.qc("P1") for _ in range(25)]
    after = _budget_state(env)

    print(f"[measured] 25 qc() calls: usd {before['usd_left']:,.2f} -> "
          f"{after['usd_left']:,.2f}, days {before['days_left']} -> "
          f"{after['days_left']}")
    assert after["usd_left"] == before["usd_left"]
    assert after["days_left"] == before["days_left"]
    assert after["days_used"] == before["days_used"]
    # Nothing else moved either, including the assay-noise stream.
    assert after == before
    assert all(r == reports[0] for r in reports)
    assert reports[0]["cost_usd"] == 0.0 and reports[0]["days_cost"] == 0


def test_qc_reports_the_right_numbers():
    env = AssayGym(5, "standard")
    env.reset()
    layout = _mixed_layout(env, 51)
    env.design_and_run(layout, "LOT-A")
    plate = env.plates["P1"]

    pos = [plate.values[w] for w, c in layout.items() if c == "POS"]
    neg = [plate.values[w] for w, c in layout.items() if c == "NTC"]
    rep = env.qc("P1")

    assert rep["n_pos"] == len(pos) and rep["n_ntc"] == len(neg)
    assert rep["mean_pos"] == pytest.approx(float(np.mean(pos)))
    assert rep["mean_ntc"] == pytest.approx(float(np.mean(neg)))
    assert rep["assay_window"] == pytest.approx(float(np.mean(pos) - np.mean(neg)))
    assert rep["z_prime"] == pytest.approx(z_prime(pos, neg))
    print(f"[measured] qc P1: n_ntc={rep['n_ntc']} n_pos={rep['n_pos']} "
          f"window={rep['assay_window']:.4f} z'={rep['z_prime']:.3f}")

    # A plate with no controls degrades to None rather than blowing up or
    # emitting a nan that will not survive JSON.
    env.design_and_run({w: "KD:" + env.world.genes[0] for w in INTERIOR[:4]}, "LOT-A")
    bare = env.qc("P2")
    assert bare["assay_window"] is None and bare["z_prime"] is None


def test_qc_on_unknown_plate_fails_cleanly():
    env = AssayGym(5, "standard")
    env.reset()
    env.design_and_run(_layout(6), "LOT-A")
    before = _budget_state(env)
    res = env.qc("P99")
    assert "error" in res and res["known_plate_ids"] == ["P1"]
    assert _budget_state(env) == before


# --------------------------------------------------------------------------
# Acceptance 4 — the one-shot guard
# --------------------------------------------------------------------------


def test_no_tool_mutates_state_after_submit():
    """submit() is one shot. Every later call errors and changes nothing."""
    env = AssayGym(9, "standard")
    env.reset()
    env.design_and_run(_mixed_layout(env, 30), "LOT-A")
    env.design_and_run(_mixed_layout(env, 30), "LOT-B")
    env.exclude_plate("P2", "window collapsed")

    out = env.submit(["SYN01", "SYN02"], {"SYN01": -1, "SYN02": 1}, 2.0)
    assert out["submitted"] is True and env.done is True
    after_submit = _budget_state(env)

    calls = {
        "design_and_run": lambda: env.design_and_run(_layout(10), "LOT-A"),
        "qc": lambda: env.qc("P1"),
        "exclude_plate": lambda: env.exclude_plate("P1", "second thoughts"),
        "submit": lambda: env.submit(["SYN03"], {"SYN03": -1}, 3.3),
        "dispatch": lambda: env.call("design_and_run", {"layout": _layout(10)}),
    }
    for name, fn in calls.items():
        res = fn()
        assert "error" in res, (name, res)
        assert res.get("done") is True, (name, res)
        assert _budget_state(env) == after_submit, name
    print(f"[measured] post-submit: {len(calls)} calls refused, "
          f"state byte-identical (usd_left ${env.usd_left:,.2f}, "
          f"{len(env.plates)} plates, submission unchanged)")

    # The recorded answer is still the first one.
    assert env.submission == {
        "hits": ["SYN01", "SYN02"],
        "signs": {"SYN01": -1, "SYN02": 1},
        "log_ec50": 2.0,
    }
    assert env.plates["P2"].excluded is True and env.plates["P1"].excluded is False


def test_submit_records_the_answer_faithfully():
    env = AssayGym(9, "clean")
    env.reset()
    out = env.submit(["SYN02", "SYN02", "SYN05", "NOPE"], {"SYN02": -1, "SYN05": 1}, None)
    assert out["hits"] == ["SYN02", "SYN05", "NOPE"]  # de-duplicated, order kept
    assert out["unknown_loci"] == ["NOPE"]  # kept, so it scores as a false positive
    assert out["log_ec50"] is None
    assert env.submission["signs"] == {"SYN02": -1, "SYN05": 1}

    env = AssayGym(9, "clean")
    env.reset()
    env.submit([], {}, 1.5)
    assert env.submission == {"hits": [], "signs": {}, "log_ec50": 1.5}
    assert env.done is True


def test_malformed_submit_does_not_burn_the_shot():
    env = AssayGym(9, "clean")
    env.reset()
    assert "error" in env.submit("SYN01", {}, 2.0)  # a string is not a hit list
    assert env.done is False and env.submission is None
    assert "error" in env.submit(["SYN01"], {}, "banana")
    assert env.done is False
    assert "error" in env.submit(["SYN01"], {}, float("nan"))
    assert env.done is False
    assert "error" not in env.submit(["SYN01"], {"SYN01": -1}, 2.0)
    assert env.done is True


# --------------------------------------------------------------------------
# exclude_plate
# --------------------------------------------------------------------------


def test_exclude_plate_on_nonexistent_id_fails_cleanly():
    """A typo must not be silently swallowed — it changes nothing and says so."""
    env = AssayGym(4, "standard")
    env.reset()
    env.design_and_run(_layout(12), "LOT-A")
    env.design_and_run(_layout(12), "LOT-B")
    before = _budget_state(env)

    for bad in ["P3", "p1", "", "PLATE-1", "1", None, 7]:
        res = env.exclude_plate(bad, "typo")
        assert "error" in res, bad
        assert "unknown plate id" in res["error"], bad
        assert res["known_plate_ids"] == ["P1", "P2"], bad
        assert _budget_state(env) == before, bad
    print("[measured] exclude_plate on 7 bad ids: all refused, "
          f"0 plates excluded, budget unchanged (${env.usd_left:,.2f})")

    assert not any(p.excluded for p in env.plates.values())

    ok = env.exclude_plate("P1", "degraded lot")
    assert ok["excluded"] is True and ok["already_excluded"] is False
    assert ok["n_excluded"] == 1 and ok["n_active"] == 1
    assert env.plates["P1"].excluded is True
    assert any("degraded lot" in n for n in env.plates["P1"].notes)

    again = env.exclude_plate("P1", "again")
    assert again["already_excluded"] is True and again["n_excluded"] == 1

    # Exclusion is free but not a refund: money and days stay spent.
    assert env.usd_left == before["usd_left"]
    assert env.days_left == before["days_left"]


# --------------------------------------------------------------------------
# The two rngs
# --------------------------------------------------------------------------


def test_same_seed_different_layouts_sample_the_identical_world():
    """Plate layout must not perturb which world was sampled.

    This is what makes the Phase 5 ledger a comparison between policies rather
    than between the different worlds each policy accidentally conjured.
    """
    a = AssayGym(41, "hard")
    a.reset()
    b = AssayGym(41, "hard")
    b.reset()

    # Two wildly different campaigns on the same seed.
    a.design_and_run(_layout(51, "NTC"), "LOT-A")
    a.design_and_run(_mixed_layout(a, 40), "LOT-C")
    a.qc("P1")
    a.exclude_plate("P2", "noise")
    b.design_and_run({"A1": "POS", "H12": "CMPD@3000"}, "LOT-B")

    hidden = [
        "true_hits", "true_signs", "gray_zone", "compound_target",
        "true_log_ec50", "true_hill", "lot_potency", "bad_lots",
        "reported_hits", "decoys", "omitted", "baseline_phenotype",
        "hit_threshold", "genes",
    ]
    for name in hidden:
        assert getattr(a.world, name) == getattr(b.world, name), name
    assert np.array_equal(a.world.true_delta, b.world.true_delta)
    assert np.array_equal(a.world.adj, b.world.adj)
    assert np.array_equal(a.world.baseline, b.world.baseline)

    # And identical to the world sampled with no environment at all.
    ref = override_phenotype_from_deltas(sample_world(41, "hard"))
    for name in hidden:
        assert getattr(a.world, name) == getattr(ref, name), name
    assert np.array_equal(a.world.true_delta, ref.true_delta)
    print(f"[measured] seed 41 hard: true_hits={a.world.true_hits} "
          f"log_ec50={a.world.true_log_ec50:.4f} identical across "
          f"{len(a.plates)}-plate and {len(b.plates)}-plate campaigns")


def test_assay_noise_uses_a_separate_stream_seeded_seed_plus_10000():
    """Noise comes from default_rng(seed + 10_000), not from the world stream."""
    assert ASSAY_RNG_OFFSET == 10_000
    seed, tier = 17, "standard"

    env = AssayGym(seed, tier)
    env.reset()
    layout = _mixed_layout(env, 51)
    got = env.design_and_run(layout, "LOT-B")["values"]
    got2 = env.design_and_run(layout, "LOT-B")["values"]

    # Reproduce both plates from the outside with the documented offset, from
    # ONE generator. The second plate must continue the stream rather than
    # restart it, so identical layout + identical lot still gives fresh noise.
    # Comparing plates run on different lots would not catch a re-seed, because
    # the lot potency alone would make the values differ.
    ref_world = override_phenotype_from_deltas(sample_world(seed, tier))
    ref_rng = np.random.default_rng(seed + ASSAY_RNG_OFFSET)
    ref = run_plate(ref_world, "P1", layout, "LOT-B", ref_rng, day=0, cost=1041.0)
    ref2 = run_plate(ref_world, "P2", layout, "LOT-B", ref_rng, day=3, cost=1041.0)
    assert got == ref.values
    assert got2 == ref2.values
    assert got != got2, "the assay rng was re-seeded instead of advanced"
    same_lot_diffs = [abs(got[w] - got2[w]) for w in layout]
    print(f"[measured] same layout, same lot, consecutive plates: mean |diff| = "
          f"{np.mean(same_lot_diffs):.4f} (0.0 would mean the noise stream "
          f"restarts per plate)")
    assert np.mean(same_lot_diffs) > 1e-3

    # The offset is load-bearing: sharing the world's seed gives different
    # numbers, so a mutation to the offset cannot pass unnoticed.
    shared = run_plate(
        ref_world, "P1", layout, "LOT-B", np.random.default_rng(seed),
        day=0, cost=1041.0,
    )
    diffs = [abs(got[w] - shared.values[w]) for w in layout]
    print(f"[measured] env noise vs default_rng(seed) noise: mean |diff| = "
          f"{np.mean(diffs):.4f} over {len(diffs)} wells (0.0 would mean the "
          f"offset was dropped)")
    assert got != shared.values
    assert np.mean(diffs) > 1e-3

    # The world rng is untouched by measurement: a second env that runs no
    # plates at all sees the same world.
    quiet = AssayGym(seed, tier)
    quiet.reset()
    assert quiet.world.true_hits == env.world.true_hits
    assert quiet.world.true_log_ec50 == env.world.true_log_ec50


def test_episodes_are_deterministic_and_seed_separated():
    def campaign(seed, tier="standard"):
        env = AssayGym(seed, tier)
        env.reset()
        layout = _mixed_layout(env, 40)
        # Same lot deliberately: a different lot would make the two plates
        # differ through potency alone, hiding a re-seeded noise stream.
        v1 = env.design_and_run(layout, "LOT-A")["values"]
        v2 = env.design_and_run(layout, "LOT-A")["values"]
        return env, v1, v2

    e1, a1, a2 = campaign(23)
    e2, b1, b2 = campaign(23)
    assert a1 == b1 and a2 == b2
    assert e1.world.contaminated_plates == e2.world.contaminated_plates
    assert e1.state()["usd_spent"] == e2.state()["usd_spent"]

    # Two plates in one episode draw from the same stream, so they differ.
    assert a1 != a2

    e3, c1, _ = campaign(24)
    assert c1 != a1
    print("[measured] determinism: seed 23 replayed byte-identical; "
          "seed 24 differs; plate 1 != plate 2 within an episode")


# --------------------------------------------------------------------------
# The briefing
# --------------------------------------------------------------------------


def test_briefing_carries_the_prior_caveat():
    """Without the caveat the prior trap is entrapment, not a fair test."""
    for tier in TIERS:
        env = AssayGym(6, tier)
        brief = env.reset()
        prior = brief["literature_prior"]
        assert prior["previously_reported_hits"] == env.world.reported_hits
        caveat = prior["caveat"]
        assert caveat == PRIOR_CAVEAT
        low = caveat.lower()
        assert "different cell background" in low
        assert "incomplete" in low and "wrong" in low
    print(f"[measured] prior caveat present on all {len(TIERS)} tiers: "
          f"{PRIOR_CAVEAT[:58]}...")


def test_briefing_is_complete_and_leaks_nothing():
    env = AssayGym(6, "hard")
    brief = env.reset()
    for key in [
        "objective", "loci", "conditions", "reagent_lots", "literature_prior",
        "budget", "cost_schedule", "plate_format", "hit_threshold",
    ]:
        assert key in brief, key

    assert brief["loci"] == env.world.genes
    assert brief["reagent_lots"] == LOTS
    assert brief["budget"] == {"usd": 3300.0, "days": 9}
    assert brief["cost_schedule"]["plate_base_usd"] == 480.0
    assert brief["cost_schedule"]["per_well_usd"] == 11.0
    assert brief["cost_schedule"]["days_per_plate"] == 3
    assert "1,041" in brief["cost_schedule"]["example"]
    assert brief["hit_threshold"] == env.world.hit_threshold == 0.20
    assert brief["plate_format"]["n_wells"] == len(WELLS) == 96

    # Nothing hidden may appear anywhere in the briefing text.
    blob = repr(brief)
    w = env.world
    assert f"{w.true_log_ec50:.4f}"[:6] not in blob
    for name in ["true_delta", "true_signs", "compound_target", "bad_lots",
                 "decoys", "omitted", "gray_zone", "true_hits"]:
        assert name not in blob, name
    assert w.compound_target in w.genes  # it IS a locus, so only the label leaks
    for decoy_free_key in ["decoys", "omitted"]:
        assert decoy_free_key not in brief


# --------------------------------------------------------------------------
# TOOL_SPEC
# --------------------------------------------------------------------------


def test_tool_spec_is_anthropic_shaped_and_matches_the_methods():
    names = [t["name"] for t in TOOL_SPEC]
    assert names == ["design_and_run", "qc", "exclude_plate", "submit"]

    env = AssayGym(1, "clean")
    env.reset()
    for tool in TOOL_SPEC:
        assert set(tool) == {"name", "description", "input_schema"}
        assert isinstance(tool["description"], str) and len(tool["description"]) > 40
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict) and schema["properties"]
        for req in schema["required"]:
            assert req in schema["properties"], (tool["name"], req)
        # Every declared parameter is a real parameter of the bound method.
        method = getattr(env, tool["name"])
        params = method.__code__.co_varnames[: method.__code__.co_argcount]
        for prop in schema["properties"]:
            assert prop in params, (tool["name"], prop)

    assert "FREE" in dict(zip(names, TOOL_SPEC))["qc"]["description"]
    assert "ONE SHOT" in dict(zip(names, TOOL_SPEC))["submit"]["description"]


def test_call_dispatches_by_name():
    env = AssayGym(1, "clean")
    env.reset()
    res = env.call("design_and_run", {"layout": _layout(10), "lot": "LOT-B"})
    assert res["plate_id"] == "P1" and res["lot"] == "LOT-B"
    assert "error" not in env.call("qc", {"plate_id": "P1"})
    assert "error" in env.call("nope", {})
    assert "error" in env.call("qc", {"wrong_kwarg": 1})
    assert "error" not in env.call("submit", {"hits": [], "signs": {}, "log_ec50": None})
    assert env.done is True


# --------------------------------------------------------------------------
# Validation — refusals are free
# --------------------------------------------------------------------------


def test_invalid_plates_are_refused_before_any_charge():
    env = AssayGym(8, "standard")
    env.reset()
    before = _budget_state(env)

    bad_layouts = [
        ({}, "empty"),
        ({"Z9": "NTC"}, "off-plate well"),
        ({"A13": "NTC"}, "column out of range"),
        ({"B2": "NONSENSE"}, "unknown condition"),
        ({"B2": "KD:SYN99"}, "unknown locus"),
        ({"B2": "CMPD@abc"}, "unparseable dose"),
        ({"B2": 7}, "non-string condition"),
    ]
    for layout, why in bad_layouts:
        res = env.design_and_run(layout, "LOT-A")
        assert "error" in res, why
        assert _budget_state(env) == before, why

    assert "error" in env.design_and_run({"B2": "NTC"}, "LOT-Z")
    assert _budget_state(env) == before
    assert len(env.plates) == 0
    print("[measured] 8 invalid design_and_run calls: all refused, "
          f"${env.usd_left:,.2f} and {env.days_left} days untouched")


def test_reset_is_required_and_rearms_the_budget():
    env = AssayGym(1, "hard")
    with pytest.raises(RuntimeError):
        env.design_and_run(_layout(5), "LOT-A")

    env.reset()
    env.design_and_run(_layout(51), "LOT-A")
    env.submit(["SYN01"], {"SYN01": 1}, 2.0)
    assert env.done and env.usd_left == 2259.0

    env.reset()
    assert env.usd_left == 3300.0 and env.days_left == 9
    assert env.plates == {} and env.done is False and env.submission is None


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        AssayGym(1, "impossible")
