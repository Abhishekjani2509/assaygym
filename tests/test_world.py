"""Phase 1 acceptance checks for assaygym/world.py.

Run: ./.venv/bin/python -m pytest tests/ -q -s   (see CONTRIBUTING.md)"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assaygym.world import (  # noqa: E402
    HIT_THRESHOLD,
    TIERS,
    override_phenotype_from_deltas,
    sample_world,
)

N_SEEDS = 200
SEEDS = list(range(N_SEEDS))


@pytest.fixture(scope="module")
def standard_worlds():
    return [sample_world(s, "standard") for s in SEEDS]


def test_determinism():
    a = sample_world(7, "standard")
    b = sample_world(7, "standard")
    assert a.true_hits == b.true_hits
    assert a.true_signs == b.true_signs
    assert a.true_log_ec50 == b.true_log_ec50
    assert a.true_hill == b.true_hill
    assert a.gray_zone == b.gray_zone
    assert a.reported_hits == b.reported_hits
    assert a.compound_target == b.compound_target
    assert a.lot_potency == b.lot_potency
    assert np.array_equal(a.true_delta, b.true_delta)
    assert np.array_equal(a.adj, b.adj)


def test_effect_structure(standard_worlds):
    """Hits are at or above threshold; gray-zone genes are strictly below."""
    for w in standard_worlds:
        idx = {g: i for i, g in enumerate(w.genes)}
        for g in w.true_hits:
            assert abs(w.true_delta[idx[g]]) >= HIT_THRESHOLD, (w.seed, g)
        for g in w.gray_zone:
            assert abs(w.true_delta[idx[g]]) < HIT_THRESHOLD, (w.seed, g)
        assert len(w.true_hits) == w.diff.n_true_hits
        assert len(w.gray_zone) == w.diff.n_gray
        assert not (set(w.true_hits) & set(w.gray_zone))


def test_effect_structure_identical_across_tiers():
    """Tiers differ in noise/traps/scarcity only, never in effect magnitude."""
    for tier in TIERS:
        mags = []
        for s in SEEDS:
            w = sample_world(s, tier)
            idx = {g: i for i, g in enumerate(w.genes)}
            mags.extend(abs(w.true_delta[idx[g]]) / HIT_THRESHOLD for g in w.true_hits)
        assert min(mags) >= 1.5 - 1e-9, (tier, min(mags))
        assert max(mags) <= 3.0 + 1e-9, (tier, max(mags))


def test_sign_balance(standard_worlds):
    signs = [s for w in standard_worlds for s in w.true_signs.values()]
    neg_frac = sum(1 for s in signs if s < 0) / len(signs)
    print(f"\n[measured] negative-sign fraction = {neg_frac:.4f} over {len(signs)} hits")
    assert abs(neg_frac - 0.5) <= 0.05, neg_frac


def test_ec50_spans_full_range(standard_worlds):
    vals = np.array([w.true_log_ec50 for w in standard_worlds])
    print(
        f"[measured] true_log_ec50: min={vals.min():.3f} max={vals.max():.3f} "
        f"mean={vals.mean():.3f}"
    )
    assert vals.min() >= 0.5 and vals.max() <= 3.5
    assert vals.min() < 0.7 and vals.max() > 3.3
    hill = np.array([w.true_hill for w in standard_worlds])
    assert hill.min() >= 0.9 and hill.max() <= 2.0


def test_blind_ec50_guess_rate_matches_uniform():
    """A fixed blind guess must not land inside tolerance more often than the
    uniform range implies.

    The spec names a too-narrow EC50 range as one of two bugs that inflated the
    prior-parrot policy, so this pins the exploit rate. With the scoring
    tolerance of +/-0.4 log units against uniform(0.5, 3.5), the best fixed
    guess wins on 0.8/3.0 = 26.7% of worlds and no more.
    """
    n = 2000
    vals = np.array([sample_world(s, "standard").true_log_ec50 for s in range(n)])
    rate = float(np.mean(np.abs(vals - 2.0) <= 0.4))
    print(f"[measured] blind log_ec50=2.0 within +/-0.4: {rate:.4f} (analytic 0.2667)")
    assert abs(rate - 2 * 0.4 / 3.0) <= 0.04, rate
    # The draw itself is uniform, not merely bounded.
    assert abs(vals.mean() - 2.0) <= 0.06, vals.mean()
    assert abs(vals.std() - 3.0 / np.sqrt(12)) <= 0.05, vals.std()
    counts, _ = np.histogram(vals, bins=10, range=(0.5, 3.5))
    chi2 = float(((counts - n / 10) ** 2 / (n / 10)).sum())
    assert chi2 < 27.9, (chi2, counts.tolist())  # 9 df, 99.9% critical value


def test_trap_rate_standard(standard_worlds):
    rate = sum(1 for w in standard_worlds if w.decoys) / len(standard_worlds)
    print(f"[measured] standard trap rate (non-empty decoys) = {rate:.4f}")
    assert abs(rate - 0.70) <= 0.06, rate


def test_trap_rate_hard():
    worlds = [sample_world(s, "hard") for s in SEEDS]
    rate = sum(1 for w in worlds if w.decoys) / len(worlds)
    print(f"[measured] hard trap rate (non-empty decoys) = {rate:.4f}")
    assert rate == 1.0


def test_decoys_prefer_gray_zone(standard_worlds):
    from_gray = 0
    total = 0
    for w in standard_worlds:
        for d in w.decoys:
            total += 1
            if d in w.gray_zone:
                from_gray += 1
    frac = from_gray / total if total else 0.0
    print(f"[measured] decoys drawn from gray zone = {frac:.4f} ({from_gray}/{total})")
    # standard has n_gray=2 and n_decoys=2, so the gray zone always suffices.
    assert frac == 1.0


def test_trap_structure(standard_worlds):
    for w in standard_worlds:
        assert not (set(w.decoys) & set(w.true_hits)), w.seed
        assert set(w.omitted) <= set(w.true_hits), w.seed
        assert len(w.omitted) < len(w.true_hits), w.seed  # never omit them all
        assert w.reported_hits == sorted(
            (set(w.true_hits) - set(w.omitted)) | set(w.decoys)
        ), w.seed
        if w.decoys:
            assert len(w.decoys) == w.diff.n_decoys
            assert len(w.omitted) == w.diff.n_omitted


def test_clean_prior_is_truthful():
    for s in SEEDS:
        w = sample_world(s, "clean")
        assert w.reported_hits == sorted(w.true_hits), s
        assert w.decoys == [] and w.omitted == []
        assert w.bad_lots == []


def test_condition_values():
    w = override_phenotype_from_deltas(sample_world(11, "standard"))
    base = w.baseline_phenotype
    idx = {g: i for i, g in enumerate(w.genes)}

    assert w.condition_value("NTC") == pytest.approx(base)
    assert w.condition_value("POS") == pytest.approx(
        base + 1.6 * float(np.max(np.abs(w.true_delta)))
    )
    for g in w.genes:
        assert w.condition_value(f"KD:{g}") == pytest.approx(
            base + w.true_delta[idx[g]]
        )

    # Compound: flat below EC50, saturating above, half-maximal at the EC50.
    target_delta = float(w.true_delta[idx[w.compound_target]])
    ec50 = 10.0 ** w.true_log_ec50
    assert w.condition_value(f"CMPD@{ec50}") == pytest.approx(base + 0.5 * target_delta)
    assert w.condition_value("CMPD@0") == pytest.approx(base)
    assert w.condition_value("CMPD@-5") == pytest.approx(base)
    top = w.condition_value("CMPD@30000") - base
    assert abs(top) > 0.8 * abs(target_delta)

    for bad in ["", "ntc", "KD:SYN99", "KD:", "CMPD@abc", "POSITIVE", "CMPD"]:
        with pytest.raises(ValueError):
            w.condition_value(bad)


def test_lot_potency():
    good, bad = [], []
    for s in SEEDS:
        w = sample_world(s, "standard")
        assert set(w.lot_potency) == {"LOT-A", "LOT-B", "LOT-C"}
        for lot, p in w.lot_potency.items():
            (bad if lot in w.bad_lots else good).append(p)
    bad_rate = sum(1 for s in SEEDS if sample_world(s, "standard").bad_lots) / len(SEEDS)
    print(f"[measured] standard bad-lot rate = {bad_rate:.4f}")
    assert abs(bad_rate - 0.35) <= 0.07
    assert 0.30 <= min(bad) and max(bad) <= 0.55
    assert abs(np.mean(good) - 1.0) < 0.02


def test_compound_target_is_a_true_hit(standard_worlds):
    for w in standard_worlds:
        assert w.compound_target in w.true_hits, w.seed


def test_decoys_fall_back_to_nulls_when_gray_zone_too_small():
    """No shipped tier has a gray zone smaller than n_decoys, so exercise the
    fallback branch with a temporary tier."""
    import dataclasses

    from assaygym import world as world_mod

    tier = dataclasses.replace(
        TIERS["standard"], name="_gray_starved", n_gray=1, n_decoys=3, p_prior_trap=1.0
    )
    world_mod.TIERS["_gray_starved"] = tier
    try:
        worlds = [sample_world(s, "_gray_starved") for s in SEEDS]
    finally:
        del world_mod.TIERS["_gray_starved"]

    gray_used = null_used = 0
    for w in worlds:
        assert len(w.decoys) == 3, w.seed
        assert not (set(w.decoys) & set(w.true_hits)), w.seed
        for d in w.decoys:
            if d in w.gray_zone:
                gray_used += 1
            else:
                null_used += 1
        # the gray zone is exhausted before any null is taken
        assert set(w.gray_zone) <= set(w.decoys), w.seed
    print(
        f"[measured] gray-starved fallback: {gray_used} gray + {null_used} null decoys"
    )
    assert gray_used == len(worlds)  # exactly the one available gray gene each time
    assert null_used == 2 * len(worlds)
