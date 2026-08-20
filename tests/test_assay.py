"""Phase 2 acceptance checks for assaygym/assay.py.

Run: ./.venv/bin/python -m pytest tests/ -q -s   (see CONTRIBUTING.md)
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assaygym.assay import (  # noqa: E402
    COLS,
    ROWS,
    WELLS,
    is_edge,
    quadrant,
    run_plate,
    z_prime,
)
from assaygym.world import (  # noqa: E402
    override_phenotype_from_deltas,
    sample_world,
)

# Interior wells, no perimeter: rows B-G x columns 2-11.
INTERIOR = [f"{r}{c}" for r in "BCDEFG" for c in range(2, 12)]


def _world(seed=3, tier="standard", **overrides):
    """A world with `diff` fields overridden, for isolating one artifact."""
    w = override_phenotype_from_deltas(sample_world(seed, tier))
    if overrides:
        w.diff = dataclasses.replace(w.diff, **overrides)
    return w


def _silent(seed=3, tier="standard", **overrides):
    """A world with every noise source switched off."""
    off = dict(
        well_noise=0.0,
        pipet_cv=0.0,
        batch_sigma=0.0,
        edge_bias=0.0,
        p_contamination=0.0,
    )
    off.update(overrides)
    return _world(seed, tier, **off)


def _by_quadrant(wells):
    out = {q: [] for q in range(4)}
    for well in wells:
        out[quadrant(well)].append(well)
    return out


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_geometry():
    assert len(WELLS) == 96 and len(set(WELLS)) == 96
    assert ROWS == "ABCDEFGH" and COLS == list(range(1, 13))

    assert is_edge("A1") and is_edge("H12") and is_edge("D1") and is_edge("A7")
    assert not is_edge("B2") and not is_edge("G11")
    assert sum(is_edge(w) for w in WELLS) == 36  # 96 - 60 interior
    assert not any(is_edge(w) for w in INTERIOR)

    assert quadrant("A1") == 0 and quadrant("A12") == 1
    assert quadrant("H1") == 2 and quadrant("H12") == 3
    assert quadrant("D6") == 0 and quadrant("E7") == 3
    counts = {q: 0 for q in range(4)}
    for w in WELLS:
        counts[quadrant(w)] += 1
    assert counts == {0: 24, 1: 24, 2: 24, 3: 24}

    for bad in ["", "I1", "A0", "A13", "9", "AA", "A", "B1x"]:
        with pytest.raises(ValueError):
            is_edge(bad)


# --------------------------------------------------------------------------
# Acceptance 1 — zero-noise passthrough
# --------------------------------------------------------------------------


def test_zero_noise_passthrough_is_exact():
    """Every noise parameter 0 and potency 1.0 => obs == condition_value exactly."""
    w = _silent()
    w.lot_potency["LOT-A"] = 1.0
    conditions = ["NTC", "POS", "CMPD@100", "CMPD@3000"] + [
        f"KD:{g}" for g in w.genes
    ]
    layout = {well: conditions[i % len(conditions)] for i, well in enumerate(WELLS)}

    res = run_plate(w, "P1", layout, "LOT-A", np.random.default_rng(0))

    worst = 0.0
    for well, cond in layout.items():
        expected = w.condition_value(cond)
        assert res.values[well] == expected, (well, cond)
        worst = max(worst, abs(res.values[well] - expected))
    print(f"\n[measured] zero-noise passthrough: max |obs - truth| = {worst:.1e} "
          f"over {len(layout)} wells (exact equality asserted)")
    assert worst == 0.0

    # Edge wells included and still exact, since edge_bias is 0 here.
    assert any(is_edge(well) for well in layout)


def test_plate_result_fields():
    w = _silent()
    layout = {well: "NTC" for well in INTERIOR[:10]}
    res = run_plate(w, "P7", layout, "LOT-B", np.random.default_rng(1), day=6, cost=1041.0)
    assert res.plate_id == "P7" and res.lot == "LOT-B"
    assert res.day_run == 6 and res.cost_usd == 1041.0
    assert res.excluded is False
    assert res.layout == layout and set(res.values) == set(layout)
    # layout is copied, not aliased
    layout["B2"] = "POS"
    assert "B2" not in res.layout or res.layout["B2"] == "NTC"


# --------------------------------------------------------------------------
# Acceptance 2 — degraded lot shrinks the assay window (THE headline number)
# --------------------------------------------------------------------------


def _window(res, layout):
    pos = [res.values[w] for w, c in layout.items() if c == "POS"]
    neg = [res.values[w] for w, c in layout.items() if c == "NTC"]
    return float(np.mean(pos) - np.mean(neg))


def _balanced_pos_ntc_layout():
    """POS and NTC spread evenly over all four quadrants, interior only.

    Symmetry matters: if POS and NTC were not matched across quadrants and the
    perimeter, edge bias and contamination would contaminate the window ratio
    with a positional term that has nothing to do with lot potency.
    """
    layout = {}
    for wells in _by_quadrant(INTERIOR).values():
        for well in wells[:4]:
            layout[well] = "POS"
        for well in wells[4:8]:
            layout[well] = "NTC"
    return layout


def test_degraded_lot_shrinks_window_to_40_percent():
    """Potency 0.4 gives ~40% of the good-lot window on the same plate."""
    layout = _balanced_pos_ntc_layout()
    n = 400
    ratios = []
    for seed in range(n):
        # Contamination off so a random blown quadrant cannot corrupt a paired
        # comparison; every other artifact left at standard-tier strength.
        w = _world(seed, "standard", p_contamination=0.0)
        w.lot_potency["GOOD"] = 1.0
        w.lot_potency["BAD"] = 0.4
        # Same rng seed for both runs => identical noise draws, so the only
        # difference between the two plates is the lot potency.
        good = run_plate(w, "G", layout, "GOOD", np.random.default_rng(seed))
        bad = run_plate(w, "B", layout, "BAD", np.random.default_rng(seed))
        ratios.append(_window(bad, layout) / _window(good, layout))

    ratios = np.array(ratios)
    mean, se = ratios.mean(), ratios.std(ddof=1) / np.sqrt(n)
    print(f"[measured] degraded-lot window ratio (potency 0.4/1.0): "
          f"{mean:.4f} +/- {se:.4f} (SE, n={n} paired plates); "
          f"min={ratios.min():.4f} max={ratios.max():.4f}")
    assert abs(mean - 0.4) < 0.02, mean


def test_degraded_lot_ratio_under_full_noise():
    """Same ratio holds with contamination live and unpaired noise draws."""
    layout = _balanced_pos_ntc_layout()
    n = 400
    ratios = []
    for seed in range(n):
        w = _world(seed, "standard")
        w.lot_potency["GOOD"] = 1.0
        w.lot_potency["BAD"] = 0.4
        good = run_plate(w, "G", layout, "GOOD", np.random.default_rng(seed))
        bad = run_plate(w, "B", layout, "BAD", np.random.default_rng(seed + 99_000))
        ratios.append(_window(bad, layout) / _window(good, layout))

    ratios = np.array(ratios)
    mean, se = ratios.mean(), ratios.std(ddof=1) / np.sqrt(n)
    print(f"[measured] degraded-lot window ratio, full noise + contamination: "
          f"{mean:.4f} +/- {se:.4f} (SE, n={n})")
    assert abs(mean - 0.4) < 0.06, mean


def test_lot_scales_effect_not_baseline():
    """The mechanism behind the ratio: NTC is untouched, POS shrinks."""
    layout = _balanced_pos_ntc_layout()
    w = _silent()
    w.lot_potency["GOOD"] = 1.0
    w.lot_potency["BAD"] = 0.4
    good = run_plate(w, "G", layout, "GOOD", np.random.default_rng(0))
    bad = run_plate(w, "B", layout, "BAD", np.random.default_rng(0))

    ntc_good = np.mean([good.values[k] for k, c in layout.items() if c == "NTC"])
    ntc_bad = np.mean([bad.values[k] for k, c in layout.items() if c == "NTC"])
    assert ntc_good == pytest.approx(ntc_bad), "baseline must not move with lot potency"
    assert ntc_bad == pytest.approx(w.baseline_phenotype)

    ratio = _window(bad, layout) / _window(good, layout)
    print(f"[measured] noise-free window ratio = {ratio:.6f} (analytic expectation 0.4)")
    assert ratio == pytest.approx(0.4)


# --------------------------------------------------------------------------
# Acceptance 3 — contaminated quadrant sits ~0.45 above the same conditions
# --------------------------------------------------------------------------


def test_contaminated_quadrant_offset():
    """Contaminated wells average ~0.45 above identical conditions elsewhere."""
    layout = {well: "NTC" for well in INTERIOR}
    n = 400
    deltas = []
    for seed in range(n):
        # Contamination guaranteed, so every plate contributes one measurement.
        w = _world(seed, "standard", p_contamination=1.0)
        res = run_plate(w, "P", layout, "LOT-A", np.random.default_rng(seed))
        q = w.contaminated_plates["P"]
        hot = [res.values[k] for k in layout if quadrant(k) == q]
        cold = [res.values[k] for k in layout if quadrant(k) != q]
        deltas.append(np.mean(hot) - np.mean(cold))

    deltas = np.array(deltas)
    mean, se = deltas.mean(), deltas.std(ddof=1) / np.sqrt(n)
    print(f"[measured] contaminated-quadrant offset = {mean:.4f} +/- {se:.4f} "
          f"(SE, n={n} plates); analytic expectation 0.45")
    assert abs(mean - 0.45) < 0.03, mean


def test_contamination_is_recorded_and_probabilistic():
    """Contamination fires at p_contamination, over enough plates to tell.

    Driven from one continuous rng, which is how the environment will run it —
    a fresh generator per plate over a few hundred sequential small seeds is
    both unrealistic and too noisy to distinguish 0.25 from 0.19.
    """
    layout = {well: "NTC" for well in INTERIOR[:2]}
    n = 20_000
    w = _world(0, "standard")  # p_contamination = 0.25
    rng = np.random.default_rng(0)
    quadrants = []
    for i in range(n):
        run_plate(w, f"P{i}", layout, "LOT-A", rng)
        if f"P{i}" in w.contaminated_plates:
            quadrants.append(w.contaminated_plates[f"P{i}"])

    rate = len(quadrants) / n
    se = np.sqrt(0.25 * 0.75 / n)
    print(f"[measured] contamination rate on standard = {rate:.4f} +/- {se:.4f} "
          f"(SE, n={n} plates); parameter 0.25")
    assert abs(rate - 0.25) < 0.01, rate

    # The blown quadrant is uniform over all four.
    counts = np.bincount(quadrants, minlength=4)
    frac = counts / counts.sum()
    print(f"[measured] blown-quadrant distribution = {np.round(frac, 4).tolist()} "
          f"(analytic expectation 0.25 each)")
    assert set(quadrants) <= {0, 1, 2, 3}
    assert max(abs(frac - 0.25)) < 0.02, frac

    # Never fires when the tier disables it.
    for seed in range(50):
        w = _world(seed, "standard", p_contamination=0.0)
        run_plate(w, "P", layout, "LOT-A", np.random.default_rng(seed))
        assert w.contaminated_plates == {}


# --------------------------------------------------------------------------
# Extra 1 — no hidden positional bias beyond the perimeter
# --------------------------------------------------------------------------


def test_no_positional_bias_between_quadrants():
    """Same condition, different quadrants, no contamination => noise only.

    Guards against a positional term leaking in beyond the intended edge bias:
    interior wells in all four quadrants must be exchangeable.
    """
    layout = {well: "NTC" for well in INTERIOR}  # interior only, so no edge bias
    n = 600
    per_quadrant = {q: [] for q in range(4)}
    for seed in range(n):
        w = _world(seed, "standard", p_contamination=0.0)
        res = run_plate(w, "P", layout, "LOT-A", np.random.default_rng(seed))
        # Remove this plate's own offset so the batch shift does not inflate
        # the between-quadrant comparison.
        centred = np.mean(list(res.values.values()))
        for q, wells in _by_quadrant(INTERIOR).items():
            per_quadrant[q].append(np.mean([res.values[k] for k in wells]) - centred)

    means = {q: float(np.mean(v)) for q, v in per_quadrant.items()}
    ses = {q: float(np.std(v, ddof=1) / np.sqrt(n)) for q, v in per_quadrant.items()}
    spread = max(means.values()) - min(means.values())
    worst_z = max(abs(means[q]) / ses[q] for q in range(4))
    print("[measured] per-quadrant mean deviation (interior, no contamination):")
    for q in range(4):
        print(f"             Q{q}: {means[q]:+.5f} +/- {ses[q]:.5f} (SE)")
    print(f"           spread = {spread:.5f}, worst |z| = {worst_z:.2f}")
    assert spread < 0.01, means
    assert worst_z < 4.0, (means, ses)


def test_edge_bias_is_the_only_positional_term():
    """Perimeter wells sit exactly edge_bias above interior ones, nothing more."""
    w = _silent(edge_bias=0.10)
    layout = {well: "NTC" for well in WELLS}
    res = run_plate(w, "P", layout, "LOT-A", np.random.default_rng(0))
    edge = np.mean([res.values[k] for k in WELLS if is_edge(k)])
    interior = np.mean([res.values[k] for k in WELLS if not is_edge(k)])
    print(f"[measured] edge minus interior (noise-free) = {edge - interior:.6f} "
          f"(parameter edge_bias = 0.10)")
    assert edge - interior == pytest.approx(0.10)


# --------------------------------------------------------------------------
# Extra 2 — batch shift is per-plate, not per-well
# --------------------------------------------------------------------------


def test_batch_shift_is_a_constant_per_plate_offset():
    """Identical layouts, different seeds => a constant offset across all wells.

    With per-well noise switched off, two plates of the same layout may differ
    only by their single per-plate batch draw. Any residual well-to-well
    variation in the difference would mean the batch shift is leaking into
    per-well variance.
    """
    layout = {well: "NTC" if i % 3 else "POS" for i, well in enumerate(INTERIOR)}
    w = _silent(batch_sigma=0.12)  # batch on, every per-well noise source off

    residuals = []
    offsets = []
    for seed in range(200):
        a = run_plate(w, "A", layout, "LOT-A", np.random.default_rng(seed))
        b = run_plate(w, "B", layout, "LOT-A", np.random.default_rng(seed + 5_000))
        diff = np.array([a.values[k] - b.values[k] for k in layout])
        offsets.append(diff.mean())
        residuals.append(diff.std(ddof=1))

    residual = float(np.max(residuals))
    offsets = np.array(offsets)
    print(f"[measured] per-well spread of the plate-to-plate difference = "
          f"{residual:.2e} (exactly constant offset expected)")
    print(f"[measured] sd of the offset itself over 200 plate pairs = "
          f"{offsets.std(ddof=1):.4f} (analytic expectation "
          f"sqrt(2)*batch_sigma = {np.sqrt(2) * 0.12:.4f})")
    assert residual < 1e-12, residual
    assert abs(offsets.std(ddof=1) - np.sqrt(2) * 0.12) < 0.02


def test_batch_variance_does_not_leak_into_well_variance():
    """Under full noise, within-plate scatter must exclude the batch term.

    If the batch shift were drawn per well, the within-plate spread of the
    difference between two plates would pick up an extra sqrt(2)*batch_sigma.
    This checks the observed spread matches the per-well-only prediction and is
    well below the per-well-batch alternative.
    """
    layout = {well: "NTC" for well in INTERIOR}
    w = _world(0, "standard", p_contamination=0.0)
    w.lot_potency["LOT-A"] = 1.0
    base = w.baseline_phenotype

    spreads = []
    for seed in range(300):
        a = run_plate(w, "A", layout, "LOT-A", np.random.default_rng(seed))
        b = run_plate(w, "B", layout, "LOT-A", np.random.default_rng(seed + 7_000))
        diff = np.array([a.values[k] - b.values[k] for k in layout])
        spreads.append(diff.std(ddof=1))
    observed = float(np.mean(spreads))

    per_well = np.sqrt(w.diff.well_noise ** 2 + (base * w.diff.pipet_cv) ** 2)
    predicted = np.sqrt(2) * per_well
    if_per_well_batch = np.sqrt(2) * np.sqrt(per_well ** 2 + w.diff.batch_sigma ** 2)
    print(f"[measured] within-plate spread of plate difference = {observed:.4f}")
    print(f"           analytic, batch per-plate  = {predicted:.4f}  <- expected")
    print(f"           analytic, batch per-well   = {if_per_well_batch:.4f}  <- ruled out")
    assert abs(observed - predicted) < 0.01, (observed, predicted)
    assert observed < if_per_well_batch - 0.05


def test_pipetting_precedes_batch_shift():
    """Pipetting must scale the signal only, never the plate's batch offset.

    Correct order gives ``obs = value*(1+e) + batch``, so within-plate scatter
    is ``value * pipet_cv`` and is independent of that plate's batch draw. The
    inverted order gives ``obs = (value + batch)*(1+e)``, whose scatter is
    ``(value + batch) * pipet_cv`` and therefore tracks the plate's own offset.

    Both orders produce the same plate mean, so the mean cannot distinguish
    them; the correlation between a plate's mean and its internal spread can.
    Without this check the inverted order passes every other test in this file.
    """
    # Batch inflated well beyond tier strength to separate the two orders
    # decisively; well noise off so pipetting is the only scatter.
    w = _silent(pipet_cv=0.05, batch_sigma=0.60)
    w.lot_potency["LOT-A"] = 1.0
    base = w.baseline_phenotype
    layout = {well: "NTC" for well in INTERIOR}

    rng = np.random.default_rng(0)
    means, sds = [], []
    for _ in range(400):
        res = run_plate(w, "P", layout, "LOT-A", rng)
        vals = np.array(list(res.values.values()))
        means.append(vals.mean())
        sds.append(vals.std(ddof=1))
    means, sds = np.array(means), np.array(sds)

    corr = float(np.corrcoef(means, sds)[0, 1])
    print(f"[measured] corr(plate mean, within-plate spread) = {corr:+.4f} "
          f"(analytic expectation ~0 for correct order, ~+1 if inverted)")
    print(f"[measured] within-plate spread = {sds.mean():.4f} +/- {sds.std(ddof=1):.4f}; "
          f"analytic base*pipet_cv = {base * 0.05:.4f}")
    assert abs(corr) < 0.2, corr
    # Spread is set by the signal alone, not by the plate offset.
    assert abs(sds.mean() - base * 0.05) < 0.005, sds.mean()
    assert sds.std(ddof=1) < 0.1 * sds.mean()


def test_pipetting_is_multiplicative_before_batch():
    """Pipetting scales the value; batch translates it. Order is checkable.

    With only pipetting live, the spread of observations must be proportional
    to the value being measured — a POS well scatters more than an NTC well.
    An additive error, or one applied after the batch shift, would not.
    """
    w = _silent(pipet_cv=0.05)
    w.lot_potency["LOT-A"] = 1.0
    ntc_wells = INTERIOR[:30]
    pos_wells = INTERIOR[30:60]
    layout = {**{k: "NTC" for k in ntc_wells}, **{k: "POS" for k in pos_wells}}

    ntc_vals, pos_vals = [], []
    for seed in range(400):
        res = run_plate(w, "P", layout, "LOT-A", np.random.default_rng(seed))
        ntc_vals.extend(res.values[k] for k in ntc_wells)
        pos_vals.extend(res.values[k] for k in pos_wells)

    ntc_sd = float(np.std(ntc_vals, ddof=1))
    pos_sd = float(np.std(pos_vals, ddof=1))
    ntc_mean = float(np.mean(ntc_vals))
    pos_mean = float(np.mean(pos_vals))
    print(f"[measured] pipetting scatter: NTC sd={ntc_sd:.4f} (mean {ntc_mean:.3f}), "
          f"POS sd={pos_sd:.4f} (mean {pos_mean:.3f})")
    print(f"           sd/mean: NTC={ntc_sd / ntc_mean:.4f} POS={pos_sd / pos_mean:.4f} "
          f"(analytic expectation pipet_cv = 0.05 for both)")
    assert pos_sd > ntc_sd  # scatter grows with the value => multiplicative
    assert abs(ntc_sd / ntc_mean - 0.05) < 0.006
    assert abs(pos_sd / pos_mean - 0.05) < 0.006


# --------------------------------------------------------------------------
# z_prime
# --------------------------------------------------------------------------


def test_z_prime():
    rng = np.random.default_rng(0)
    pos = rng.normal(2.0, 0.05, 40)
    neg = rng.normal(1.0, 0.05, 40)
    expected = 1 - 3 * (pos.std(ddof=1) + neg.std(ddof=1)) / abs(pos.mean() - neg.mean())
    got = z_prime(pos, neg)
    assert got == pytest.approx(expected)
    print(f"[measured] z_prime on a clean separation = {got:.4f}")
    assert got > 0.5  # a usable assay

    # Degrades as the window collapses.
    tight = z_prime(rng.normal(1.05, 0.05, 40), rng.normal(1.0, 0.05, 40))
    assert tight < got

    assert np.isnan(z_prime([1.0], [0.0, 0.1]))
    assert np.isnan(z_prime([1.0, 1.1], [0.0]))
    assert np.isnan(z_prime([], []))
    assert z_prime([1.0, 1.0], [1.0, 1.0]) == float("-inf")


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_run_plate_is_deterministic():
    w1 = _world(5)
    w2 = _world(5)
    layout = {well: "NTC" if i % 2 else "POS" for i, well in enumerate(INTERIOR)}
    a = run_plate(w1, "P", layout, "LOT-A", np.random.default_rng(42))
    b = run_plate(w2, "P", layout, "LOT-A", np.random.default_rng(42))
    assert a.values == b.values
    assert w1.contaminated_plates == w2.contaminated_plates


def test_run_plate_validation():
    w = _world()
    with pytest.raises(ValueError):
        run_plate(w, "P", {"Z9": "NTC"}, "LOT-A", np.random.default_rng(0))
    with pytest.raises(ValueError):
        run_plate(w, "P", {"B2": "NTC"}, "LOT-Z", np.random.default_rng(0))
    with pytest.raises(ValueError):
        run_plate(w, "P", {"B2": "NONSENSE"}, "LOT-A", np.random.default_rng(0))

    raw = sample_world(1, "standard")  # condition_value not installed
    with pytest.raises(ValueError):
        run_plate(raw, "P", {"B2": "NTC"}, "LOT-A", np.random.default_rng(0))
