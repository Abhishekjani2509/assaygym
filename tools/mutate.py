#!/usr/bin/env python3
"""Mutation testing for AssayGym.

A test suite that has never failed has not been shown to test anything. This
script breaks the source on purpose, one edit at a time, and checks that the
suite notices. A mutant the suite does not catch is a hole in the suite, not a
bug in the mutant.

It has already earned its place twice:

  * Phase 2 -- moving pipetting error after the batch shift survived the whole
    suite. The two orders produce identical plate means, so nothing in the tests
    could separate them. `test_pipetting_precedes_batch_shift` closed the gap by
    measuring the correlation between a plate's mean and its internal spread.
  * Phase 3 -- re-creating the assay rng inside `design_and_run`, so every plate
    in an episode gets identical noise, survived. The determinism test compared
    plates run on *different reagent lots*, and the potency difference alone
    made the values differ, masking the repeated noise. The test was passing for
    the wrong reason.

Both were bugs a careful reader had already looked straight at. Neither showed
up any other way.

Usage
-----
    ./.venv/bin/python tools/mutate.py                # every target
    ./.venv/bin/python tools/mutate.py rewards env    # named targets
    ./.venv/bin/python tools/mutate.py --list         # show the catalogue
    ./.venv/bin/python tools/mutate.py rewards -k test_rewards.py

Exit code is 0 only when every mutant behaved as the catalogue predicts:
`expect="killed"` mutants must break the suite, and `expect="survives"` control
mutants (no-op edits) must not. A control mutant that gets "killed" means the
suite is flaky, which invalidates every other result in the run.

Adding a mutant
---------------
Append a `Mutant` to `CATALOGUE[target]`. `edits` is a tuple of
`(old, new)` string replacements applied together; every `old` must appear
exactly once in the file or the mutant is reported as PATCH-FAILED rather than
silently skipped. Anchor on code, not on comment prose, so a reworded comment
does not quietly disable a mutant.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable


@dataclass(frozen=True)
class Mutant:
    group: str
    name: str
    edits: Tuple[Tuple[str, str], ...]
    expect: str = "killed"  # "killed" or "survives" (no-op control)


def M(group: str, name: str, old: str, new: str, expect: str = "killed") -> Mutant:
    """One-edit mutant. The common case."""
    return Mutant(group, name, ((old, new),), expect)


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

CATALOGUE: Dict[str, List[Mutant]] = {}

# --- Phase 1: world.py -----------------------------------------------------
CATALOGUE["world"] = [
    M("effect structure", "hit signs forced negative (blind guess scores 100%)",
      "hit_signs = rng.choice(np.array([-1.0, 1.0]), size=len(hit_idx))",
      "hit_signs = -np.ones(len(hit_idx))"),
    M("effect structure", "hit magnitudes pushed below threshold",
      "_HIT_MULT_RANGE = (1.5, 3.0)", "_HIT_MULT_RANGE = (0.5, 0.9)"),
    M("effect structure", "gray zone pushed above threshold",
      "_GRAY_MULT_RANGE = (0.60, 0.92)", "_GRAY_MULT_RANGE = (1.05, 1.40)"),
    M("compound", "EC50 range narrowed (inflates a blind guess)",
      "_LOG_EC50_RANGE = (0.5, 3.5)", "_LOG_EC50_RANGE = (1.8, 2.2)"),
    M("prior trap", "decoys drawn from nulls instead of the gray zone",
      "n_from_gray = min(diff.n_decoys, len(gray_zone))", "n_from_gray = 0"),
    # Equivalent at every shipped tier: n_omitted (0/1/2) is always below
    # n_true_hits (3/3/4), so min(n_omitted, n-1) == min(n_omitted, n) and the
    # `- 1` guard is unreachable. It is defensive code for a tier config that
    # does not exist, so no test can kill this and none should be written to.
    # Kept in the catalogue to record that fact rather than rediscover it.
    M("prior trap", "omission may remove every genuine hit (equivalent at shipped tiers)",
      "n_omit = min(diff.n_omitted, len(true_hits) - 1)",
      "n_omit = min(diff.n_omitted, len(true_hits))", expect="survives"),
    M("control", "no-op: comment reworded", "# Absolute, in readout units.",
      "# Absolute value, in readout units.", expect="survives"),
]

# --- Phase 2: assay.py -----------------------------------------------------
CATALOGUE["assay"] = [
    M("artifact order", "lot potency multiplies the raw value, not the effect",
      "obs = base + effect * lot_potency", "obs = (base + effect) * lot_potency"),
    Mutant("artifact order", "pipetting applied after the batch shift", (
        ("obs *= 1.0 + rng.normal(0.0, diff.pipet_cv)", "pass"),
        ("obs += batch_shift",
         "obs += batch_shift\n        obs *= 1.0 + rng.normal(0.0, diff.pipet_cv)"),
    )),
    M("artifact order", "pipetting error additive instead of multiplicative",
      "obs *= 1.0 + rng.normal(0.0, diff.pipet_cv)",
      "obs += rng.normal(0.0, diff.pipet_cv)"),
    M("per-plate draws", "batch shift drawn per well instead of per plate",
      "obs += batch_shift", "obs += float(rng.normal(0.0, diff.batch_sigma))"),
    M("per-plate draws", "contaminated quadrant re-rolled per well",
      "if contaminated_quadrant is not None and quadrant(well) == contaminated_quadrant:",
      "if contaminated_quadrant is not None and rng.random() < 0.25:"),
    M("geometry", "edge test inverted (interior wells get the bias)",
      "        if is_edge(well):", "        if not is_edge(well):"),
    M("control", "no-op: comment reworded", "# --- drawn ONCE per plate",
      "# --- drawn once per plate", expect="survives"),
]

# --- Phase 3: env.py -------------------------------------------------------
CATALOGUE["env"] = [
    # budget arithmetic
    M("budget", "per-well cost 11.0 -> 10.0",
      "PER_WELL_COST = 11.0", "PER_WELL_COST = 10.0"),
    M("budget", "plate base cost 480.0 -> 500.0",
      "PLATE_BASE_COST = 480.0", "PLATE_BASE_COST = 500.0"),
    M("budget", "days per plate 3 -> 2", "PLATE_DAYS = 3", "PLATE_DAYS = 2"),
    M("budget", "cost formula off by one well",
      "return PLATE_BASE_COST + PER_WELL_COST * n_wells",
      "return PLATE_BASE_COST + PER_WELL_COST * (n_wells - 1)"),
    M("budget", "affordability > -> >= (rejects a plate you can exactly afford)",
      "        if cost > self.usd_left:", "        if cost >= self.usd_left:"),
    M("budget", "affordability > -> < (accepts anything)",
      "        if cost > self.usd_left:", "        if cost < self.usd_left:"),
    M("budget", "money never decremented",
      "        self.usd_left -= cost", "        self.usd_left -= 0.0"),
    M("budget", "days never decremented",
      "        self.days_left -= PLATE_DAYS", "        self.days_left -= 0"),
    M("budget", "days_used never advanced",
      "        self.days_used += PLATE_DAYS", "        self.days_used += 0"),
    M("budget", "days constraint dropped (money checked, days not)",
      "        if PLATE_DAYS > self.days_left:",
      "        if False and PLATE_DAYS > self.days_left:"),
    M("budget", "money constraint dropped (days checked, money not)",
      "        if cost > self.usd_left:",
      "        if False and cost > self.usd_left:"),
    M("budget", "charged before validation (an invalid plate still bills)",
      "        n_wells = len(layout)\n        cost = plate_cost(n_wells)",
      "        n_wells = len(layout)\n        cost = plate_cost(n_wells)\n"
      "        self.usd_left -= cost"),
    M("budget", "qc charges money",
      '            "cost_usd": 0.0,\n            "days_cost": 0,',
      '            "cost_usd": 0.0,\n            "days_cost": 0,\n'
      '            **{"_": [setattr(self, "usd_left", self.usd_left - 1.0)]},'),
    M("budget", "qc burns a day",
      '        pos = [plate.values[w] for w, c in plate.layout.items() if c == "POS"]',
      '        self.days_left -= 1\n'
      '        pos = [plate.values[w] for w, c in plate.layout.items() if c == "POS"]'),
    # the one-shot guard
    M("guard", "design_and_run guard removed",
      '        if self.done:\n            return self._post_submit_error("design_and_run")',
      '        if False:\n            return self._post_submit_error("design_and_run")'),
    M("guard", "qc guard removed",
      '        if self.done:\n            return self._post_submit_error("qc")',
      '        if False:\n            return self._post_submit_error("qc")'),
    M("guard", "exclude_plate guard removed",
      '        if self.done:\n            return self._post_submit_error("exclude_plate")',
      '        if False:\n            return self._post_submit_error("exclude_plate")'),
    M("guard", "submit guard removed (a second submission overwrites the first)",
      '        if self.done:\n            return self._post_submit_error("submit")',
      '        if False:\n            return self._post_submit_error("submit")'),
    M("guard", "submit never sets done",
      '        self.done = True\n\n        return {\n            "submitted": True,',
      '        self.done = False\n\n        return {\n            "submitted": True,'),
    M("guard", "guard errors but mutates the budget anyway",
      '        """The one-shot guard. Returns an error and mutates nothing."""\n'
      '        return _err(',
      '        """The one-shot guard. Returns an error and mutates nothing."""\n'
      '        self.usd_left -= 1.0\n        return _err('),
    # the two rngs
    M("rng", "assay rng seeded `seed` instead of `seed + 10_000`",
      "        self.rng = np.random.default_rng(self.seed + ASSAY_RNG_OFFSET)",
      "        self.rng = np.random.default_rng(self.seed)"),
    M("rng", "offset 10_000 -> 1 (adjacent seeds collide)",
      "ASSAY_RNG_OFFSET = 10_000", "ASSAY_RNG_OFFSET = 1"),
    M("rng", "assay rng re-created per plate (every plate gets identical noise)",
      "        result = run_plate(\n            world,",
      "        self.rng = np.random.default_rng(self.seed + ASSAY_RNG_OFFSET)\n"
      "        result = run_plate(\n            world,"),
    M("rng", "assay rng unseeded (nondeterministic episodes)",
      "        self.rng = np.random.default_rng(self.seed + ASSAY_RNG_OFFSET)",
      "        self.rng = np.random.default_rng()"),
    M("rng", "world seeded from the assay stream",
      "        self.world = override_phenotype_from_deltas(sample_world(self.seed, self.tier))",
      "        self.world = override_phenotype_from_deltas(\n"
      "            sample_world(self.seed + ASSAY_RNG_OFFSET, self.tier))"),
    # briefing and error paths
    M("briefing", "prior caveat removed from the briefing",
      '"caveat": PRIOR_CAVEAT,', '"caveat": "",'),
    M("briefing", 'caveat drops the "different cell background" clause',
      '    "These previously reported hits come from a different cell background than "\n'
      '    "the one you are screening. The list may be incomplete or wrong: it can "',
      '    "These previously reported hits are from an earlier screen. "\n'
      '    "The list may be incomplete or wrong: it can "'),
    M("errors", "exclude_plate silently accepts an unknown id",
      '        if plate is None:\n            return _err(\n'
      '                f"unknown plate id {plate_id!r}; nothing excluded",\n'
      '                known_plate_ids=list(self.plates),\n            )',
      '        if plate is None:\n'
      '            return {"plate_id": plate_id, "excluded": True,\n'
      '                    "already_excluded": False, "reason": str(reason),\n'
      '                    "n_excluded": 0, "n_active": 0,\n'
      '                    "usd_left": self.usd_left, "days_left": self.days_left}'),
    M("errors", "submit silently drops unknown loci (hides false positives)",
      "        unknown = [h for h in clean_hits if h not in world.genes]",
      "        unknown = [h for h in clean_hits if h not in world.genes]\n"
      "        clean_hits = [h for h in clean_hits if h in world.genes]"),
    M("control", "no-op: comment reworded", "# --- 3.1 Economy",
      "# --- 3.1 The economy", expect="survives"),
]

# --- Phase 4: rewards.py ---------------------------------------------------
CATALOGUE["rewards"] = [
    # the efficiency gate -- the single most important line in the file
    M("efficiency gate", "gate removed entirely (running nothing banks the budget)",
      "    if endpoint <= EFFICIENCY_GATE:\n        return 0.0",
      "    if False:\n        return 0.0"),
    M("efficiency gate", "gate threshold 0.40 -> 0.0",
      "EFFICIENCY_GATE = 0.40", "EFFICIENCY_GATE = 0.0"),
    M("efficiency gate", "gate <= -> < (pays at exactly the threshold)",
      "    if endpoint <= EFFICIENCY_GATE:", "    if endpoint < EFFICIENCY_GATE:"),
    M("efficiency gate", "n_plates condition removed (zero-plate policy banks the budget)",
      "    if n_plates <= 0:\n        return 0.0", "    if False:\n        return 0.0"),
    M("efficiency gate", "n_plates <= 0 -> < 0 (never fires)",
      "    if n_plates <= 0:", "    if n_plates < 0:"),
    M("efficiency gate", "efficiency measures spent, not unspent",
      "    return _clip01(1.0 - usd_spent / budget_usd)",
      "    return _clip01(usd_spent / budget_usd)"),
    # endpoint composition
    M("endpoint", "hit_f1 and ec50 weights swapped",
      '"hit_f1": 0.55,\n    "sign_acc": 0.15,\n    "ec50": 0.30,',
      '"hit_f1": 0.30,\n    "sign_acc": 0.15,\n    "ec50": 0.55,'),
    M("endpoint", "hit_f1 and sign_acc weights swapped",
      '"hit_f1": 0.55,\n    "sign_acc": 0.15,',
      '"hit_f1": 0.15,\n    "sign_acc": 0.55,'),
    M("endpoint", "precision used where F1 is specified",
      "    hit_f1 = (\n        2.0 * precision * recall / (precision + recall)\n"
      "        if (precision + recall) > 0\n        else 0.0\n    )",
      "    hit_f1 = precision"),
    M("endpoint", "recall used where F1 is specified",
      "    hit_f1 = (\n        2.0 * precision * recall / (precision + recall)\n"
      "        if (precision + recall) > 0\n        else 0.0\n    )",
      "    hit_f1 = recall"),
    M("endpoint", "sign_acc vacuously 1.0 when nothing was identified",
      "    if n_tp == 0:\n        sign_acc = 0.0",
      "    if n_tp == 0:\n        sign_acc = 1.0"),
    M("endpoint", "sign_acc computed over all submitted hits, not true positives",
      "        correct = sum(\n            1 for g in true_pos "
      "if signs.get(g, 0) == world.true_signs.get(g, 0)\n        )\n"
      "        sign_acc = correct / n_tp",
      "        correct = sum(\n            1 for g in submitted "
      "if signs.get(g, 0) == world.true_signs.get(g, 0)\n        )\n"
      "        sign_acc = correct / len(submitted)"),
    M("endpoint", "a missing EC50 scores full credit instead of zero",
      "    if log_ec50 is None:\n        ec50 = 0.0",
      "    if log_ec50 is None:\n        ec50 = 1.0"),
    M("endpoint", "EC50 tolerance 0.40 -> 1.0",
      "EC50_TOLERANCE = 0.40", "EC50_TOLERANCE = 1.00"),
    # strict_pass
    M("strict_pass", "exact set equality relaxed to superset (farms recall)",
      "    if set(hits) != set(world.true_hits):",
      "    if not set(hits) >= set(world.true_hits):"),
    M("strict_pass", "exact set equality relaxed to any overlap",
      "    if set(hits) != set(world.true_hits):",
      "    if not (set(hits) & set(world.true_hits)):"),
    M("strict_pass", "signs no longer checked",
      "        if signs.get(gene, 0) != world.true_signs.get(gene, 0):\n            return 0.0",
      "        if False:\n            return 0.0"),
    M("strict_pass", "EC50 no longer checked",
      "    if abs(log_ec50 - world.true_log_ec50) > EC50_TOLERANCE + _EC50_EPS:\n"
      "        return 0.0",
      "    if False:\n        return 0.0"),
    M("strict_pass", "a missing EC50 still passes",
      "    if log_ec50 is None:\n        return 0.0",
      "    if log_ec50 is None:\n        return 1.0"),
    M("strict_pass", "EC50 slack widened from 1e-9 to 0.01",
      "_EC50_EPS = 1e-9", "_EC50_EPS = 1e-2"),
    # qc_hygiene
    M("qc_hygiene", "sign inverted (rewards over-excluding)",
      "    return _clip01(caught - wrongly)", "    return _clip01(wrongly - caught)"),
    M("qc_hygiene", "over-exclusion penalty dropped",
      "    return _clip01(caught - wrongly)", "    return _clip01(caught)"),
    M("qc_hygiene", "no-bad-lot branch returns 0.0 instead of 1.0",
      "    else:\n        caught = 1.0", "    else:\n        caught = 0.0"),
    M("qc_hygiene", "bad-lot branch credits plates that were NOT excluded",
      "        caught = sum(1 for p in bad_plates if p.excluded) / len(bad_plates)",
      "        caught = sum(1 for p in bad_plates if not p.excluded) / len(bad_plates)"),
    M("qc_hygiene", "zero plates scores 1.0 (free credit for doing nothing)",
      "    if not plates:\n        return 0.0\n\n    bad_lots = set(world.bad_lots)",
      "    if not plates:\n        return 1.0\n\n    bad_lots = set(world.bad_lots)"),
    # process terms
    M("controls", "threshold 4 NTC -> 1 NTC", "MIN_NTC_FOR_CONTROLS = 4",
      "MIN_NTC_FOR_CONTROLS = 1"),
    M("controls", "threshold 2 POS -> 0 POS", "MIN_POS_FOR_CONTROLS = 2",
      "MIN_POS_FOR_CONTROLS = 0"),
    M("controls", "excluded plates counted as controlled",
      "    active = [p for p in plates if not p.excluded]\n    if not active:\n"
      "        return 0.0",
      "    active = list(plates)\n    if not active:\n        return 0.0"),
    M("replication", "counts every measured condition, not submitted hits",
      "    for gene in set(hits):\n        condition = f\"KD:{gene}\"",
      "    for gene in {c[3:] for p in plates for c in p.layout.values() "
      "if c.startswith('KD:')}:\n        condition = f\"KD:{gene}\""),
    M("replication", "counts wells rather than distinct plates",
      "        n_plates = sum(1 for p in active if condition in p.layout.values())",
      "        n_plates = sum(list(p.layout.values()).count(condition) for p in active)"),
    M("replication", "one plate is enough", "MIN_PLATES_FOR_REPLICATION = 2",
      "MIN_PLATES_FOR_REPLICATION = 1"),
    M("replication", "excluded plates still count",
      "    active = [p for p in plates if not p.excluded]\n    replicated = 0",
      "    active = list(plates)\n    replicated = 0"),
    M("self_normalizable", "NTC alone is enough, no test condition needed",
      '        if "NTC" in conds and any(_is_test_condition(c) for c in conds):',
      '        if "NTC" in conds:'),
    M("self_normalizable", "POS counts as a test condition",
      '    return condition.startswith("KD:") or condition.startswith("CMPD@")',
      '    return condition != "NTC"'),
    # shaped composition
    M("shaped", "endpoint weight 0.55 -> 0.95, process terms marginalised",
      '"endpoint": 0.55,\n    "controls": 0.10,',
      '"endpoint": 0.95,\n    "controls": 0.10,'),
    M("shaped", "qc_hygiene and efficiency weights swapped",
      '"qc_hygiene": 0.12,\n    "efficiency": 0.08,',
      '"qc_hygiene": 0.08,\n    "efficiency": 0.12,'),
    # diagnostics must not leak into reward
    M("diagnostics", "decoy_called subtracted from the endpoint",
      '        "endpoint": ep["endpoint"],',
      '        "endpoint": ep["endpoint"] - 0.05 * len(\n'
      '            set(_submission_parts(submission)[0]) & set(world.decoys)),'),
    M("diagnostics", "precision reported where recall is specified",
      '        "recall": float(n_tp / len(truth)) if truth else 0.0,',
      '        "recall": float(n_tp / len(submitted)) if submitted else 0.0,'),
    M("diagnostics", "decoy_called counts all false positives, not just decoys",
      '        "decoy_called": int(len(submitted & set(world.decoys))),',
      '        "decoy_called": int(len(submitted - truth)),'),
    M("control", "no-op: comment reworded", "# --- 4.1 endpoint",
      "# --- 4.1 The endpoint", expect="survives"),
]

# --- Phase 6: llm_harness.py -----------------------------------------------
CATALOGUE["llm_harness"] = [
    M("loop", "max_turns 24 -> 1", "MAX_TURNS = 24", "MAX_TURNS = 1"),
    M("loop", "turn cap removed (a never-submitting model loops forever)",
      "    while turns < max_turns and not env.done:",
      "    while turns < max_turns * 3 and not env.done:"),
    M("loop", "pause_turn breaks instead of resuming",
      '        if stop_reason == "pause_turn":\n            continue',
      '        if stop_reason == "pause_turn":\n            break'),
    M("loop", "no tool call breaks instead of nudging",
      '            messages.append({"role": "user", "content": NUDGE})',
      '            break  # MUTANT'),
    M("scoring", "forced empty submission removed (episode vanishes unscored)",
      "        env.submit([], {}, None)\n        forced = True",
      "        forced = True"),
    M("tool results", "parallel tool_results split across separate messages",
      '        messages.append({"role": "user", "content": results})',
      '        for _r in results:\n'
      '            messages.append({"role": "user", "content": [_r]})'),
    M("tool results", "errors not flagged with is_error",
      '                "is_error": is_error,\n            })',
      '                "is_error": False,\n            })'),
    M("tool results", "tool_use_id not echoed back",
      '                "tool_use_id": _block_attr(use, "id", ""),',
      '                "tool_use_id": "",'),
    M("tool results", "failed tool calls dropped instead of reported",
      "            results.append({",
      "            if is_error:\n                continue\n            results.append({"),
    M("briefing", "briefing summarised instead of passed through verbatim",
      "        f\"{json.dumps(briefing, indent=2)}\\n\\n\"",
      "        f\"Loci: {briefing['loci']}\\n\\n\""),
    M("briefing", "tool list replaced with an empty list",
      '            "tools": TOOL_SPEC,', '            "tools": [],'),
    M("reporting", "harness settings not recorded (the variable goes unreported)",
      '            "model": model, "max_turns": max_turns, "max_tokens": max_tokens,',
      '            "model": "unknown", "max_turns": 0, "max_tokens": 0,'),
    M("reporting", "non-dict tool input passed through instead of coerced",
      "            arguments = raw_input if isinstance(raw_input, dict) else {}",
      "            arguments = raw_input"),
    M("cost", "usage not accumulated across turns (spend cap goes blind)",
      "        _accumulate_usage(usage_total, response)", "        pass"),
    M("cost", "cache reads billed at full input price",
      "        + 0.1 * usage.get(\"cache_read_input_tokens\", 0)",
      "        + 1.0 * usage.get(\"cache_read_input_tokens\", 0)"),
    M("cost", "unpriced model silently guesses a price instead of 0.0",
      "    if price is None:\n        return 0.0",
      "    if price is None:\n        price = (5.0, 25.0)"),
    M("cost", "non-int usage values accumulated (a null becomes a crash)",
      "        if isinstance(value, int):", "        if value is not None:"),
    M("control", "no-op: comment reworded", "# Claude Opus 5. Overridable",
      "# Claude Opus 5 -- overridable", expect="survives"),
]

# --- Phase 6: vf_adapter.py -------------------------------------------------
CATALOGUE["vf_adapter"] = [
    M("dataset", "ground truth leaked into the answer field",
      '            "answer": "",',
      '            "answer": str(env.world.true_hits),'),
    M("dataset", "seed dropped from info (grading cannot re-derive the world)",
      '                "seed": seed,', '                "seed": 0,'),
    M("dataset", "every row built from seed0 instead of one per seed",
      "        seed = seed0 + i", "        seed = seed0"),
    M("dataset", "n_episodes ignored",
      "    for i in range(n_episodes):", "    for i in range(1):"),
    M("fallback", "ImportError propagates instead of falling back",
      '    except ImportError:\n        return _fallback(tier, n_episodes, seed0, '
      '"verifiers is not installed")',
      "    except ImportError:\n        raise"),
    M("fallback", "version mismatch raises instead of degrading",
      "    except (AttributeError, TypeError) as exc:",
      "    except (AttributeError,) as exc:"),
    M("fallback", "fallback drops the usable callables",
      '        "make_env": make_env,\n        "score_episode": score_episode,',
      '        "make_env": None,\n        "score_episode": None,'),
    M("rubric", "primary metric switched from strict_pass to endpoint",
      '    "primary": "strict_pass",', '    "primary": "endpoint",'),
    M("rubric", "diagnostics folded into the optimised metrics",
      '    "metrics": ["strict_pass", "endpoint", "shaped"],',
      '    "metrics": ["strict_pass", "endpoint", "shaped", "decoy_called"],'),
    M("control", "no-op: docstring reworded",
      "**One dataset row per seed, and the row never contains the answer.**",
      "**One dataset row per seed; the row never contains the answer.**",
      expect="survives"),
]

TARGET_FILES = {
    "world": "assaygym/world.py",
    "assay": "assaygym/assay.py",
    "env": "assaygym/env.py",
    "rewards": "assaygym/rewards.py",
    "llm_harness": "assaygym/llm_harness.py",
    "vf_adapter": "assaygym/vf_adapter.py",
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _pytest(tests: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "pytest", tests, "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    )


def _failed_tests(output: str) -> List[str]:
    names = sorted({
        line.split("::")[1].split(" ")[0]
        for line in output.splitlines()
        if line.startswith("FAILED") and "::" in line
    })
    return names or ["<collection/import error>"]


def run_target(target: str, tests: str) -> List[Tuple[Mutant, str, str]]:
    """Apply each mutant for `target` in turn. Always restores the file."""
    path = ROOT / TARGET_FILES[target]
    if not path.exists():
        return [(m, "NO-TARGET", f"{TARGET_FILES[target]} does not exist")
                for m in CATALOGUE.get(target, [])]

    original = path.read_text()
    results: List[Tuple[Mutant, str, str]] = []
    try:
        for mut in CATALOGUE.get(target, []):
            source = original
            bad_anchor = None
            for old, new in mut.edits:
                if source.count(old) != 1:
                    bad_anchor = f"anchor appears {source.count(old)}x: {old[:48]!r}"
                    break
                source = source.replace(old, new, 1)
            if bad_anchor:
                results.append((mut, "PATCH-FAILED", bad_anchor))
                continue

            path.write_text(source)
            proc = _pytest(tests)
            path.write_text(original)  # restore before anything else can fail

            if proc.returncode == 0:
                results.append((mut, "SURVIVED", "suite still green"))
            else:
                failed = _failed_tests(proc.stdout + proc.stderr)
                detail = f"{len(failed)} test(s): " + ", ".join(failed[:3])
                if len(failed) > 3:
                    detail += " ..."
                results.append((mut, "KILLED", detail))
    finally:
        path.write_text(original)
    return results


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Break the source on purpose; confirm the suite notices.")
    ap.add_argument("targets", nargs="*", choices=sorted(TARGET_FILES) + [],
                    help="modules to mutate (default: all with a catalogue)")
    ap.add_argument("-k", "--tests", default="tests/",
                    help="pytest target to run for each mutant (default: tests/)")
    ap.add_argument("--list", action="store_true",
                    help="print the catalogue and exit without running anything")
    args = ap.parse_args(argv)

    targets = args.targets or [t for t in TARGET_FILES if CATALOGUE.get(t)]

    if args.list:
        for target in targets:
            muts = CATALOGUE.get(target, [])
            print(f"\n{target}  ({TARGET_FILES[target]}, {len(muts)} mutants)")
            for mut in muts:
                flag = "" if mut.expect == "killed" else "  [control, must survive]"
                print(f"  [{mut.group}] {mut.name}{flag}")
        return 0

    # A baseline green run. Without it, "KILLED" is meaningless -- the suite
    # might have been red before we touched anything.
    baseline = _pytest(args.tests)
    if baseline.returncode != 0:
        print("BASELINE FAILED: the suite is not green before mutation.")
        print("Nothing was mutated. Fix the suite first.\n")
        print(baseline.stdout[-2000:])
        return 2
    print(f"baseline: {args.tests} green\n")

    all_results: List[Tuple[str, Mutant, str, str]] = []
    for target in targets:
        results = run_target(target, args.tests)
        if not results:
            continue
        width = max(len(m.name) for m, _, _ in results)
        print(f"=== {target}  ({TARGET_FILES[target]}) " + "=" * max(0, 46 - len(target)))
        group = None
        for mut, status, detail in results:
            if mut.group != group:
                print(f"  -- {mut.group}")
                group = mut.group
            print(f"    {status:12s} {mut.name:<{width}}  {detail}")
            all_results.append((target, mut, status, detail))
        print()

    expected_kill = [r for r in all_results if r[1].expect == "killed"]
    controls = [r for r in all_results if r[1].expect == "survives"]
    killed = [r for r in expected_kill if r[2] == "KILLED"]
    bad_controls = [r for r in controls if r[2] != "SURVIVED"]
    problems = [r for r in expected_kill if r[2] != "KILLED"] + bad_controls

    print(f"{len(killed)}/{len(expected_kill)} mutants killed; "
          f"{len(controls) - len(bad_controls)}/{len(controls)} controls survived "
          f"as expected.")
    if problems:
        print("\nPROBLEMS:")
        for target, mut, status, detail in problems:
            why = ("a control mutant was killed -- the suite is flaky"
                   if mut.expect == "survives" else "not caught by any test")
            print(f"  [{status}] {target}: {mut.name}\n      {why} ({detail})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
