# Contributing / running AssayGym locally

This project's central claim is that its reported numbers are reproducible by
anyone. That only holds if the environment is part of the repo, so the setup
below is the supported path — please don't report numbers produced by an ad-hoc
interpreter.

Requires Python 3.11+ (developed and verified on 3.13.7). `numpy` is the only
runtime dependency; `pytest` is used for the acceptance suites.

## Setup

From a clean clone:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Running the tests

```bash
./.venv/bin/python -m pytest tests/ -q
```

Add `-s` to see the measured values (sign balance, trap rate, bad-lot rate) that
the acceptance checks assert on — they are printed, not just asserted, because
the exact figures are what get reported:

```bash
./.venv/bin/python -m pytest tests/ -q -s
```

One-liner from a clean clone:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt && ./.venv/bin/python -m pytest tests/ -q -s
```

If you'd rather activate the venv (`source .venv/bin/activate`), the commands
become plain `pip` / `pytest`. The explicit `./.venv/bin/...` form is used above
so the commands are correct whether or not the venv is active.

## The Phase 5 gate

Phase 5 is the phase that makes the project credible, and it is a gate: if the
baseline ladder is not monotone, the bug is in the world model or the scoring,
not in the policy.

```bash
./.venv/bin/python verify.py                       # the three gate checks
./.venv/bin/python run_baselines.py --n 200 --json results.json
./.venv/bin/python run_baselines.py --n 1000 --ablate --tiers standard hard
```

`verify.py` checks determinism, that the "call every gene a hit" exploit scores
exactly zero, and that the ledger matches the BUILD_SPEC acceptance table within
±0.05 while staying monotone in every tier. It exits non-zero if any of the
three fails. The same checks run inside `tests/test_policies.py`, which is where
they belong for CI; `verify.py` is the human-readable version.

**Never tune a policy to hit a ledger number.** Every constant in
`policies.py` is either specified by BUILD_SPEC or derived from the assay
(`CONTAM_FLOOR = 0.25` sits between a real effect at 0.20 and contamination at
0.45). If a cell misses, the cause is upstream. `naive_screen` is the one place
a construction choice was made from evidence rather than from the text, and the
reasoning plus both candidate ledgers are in its docstring.

Episodes are fully seeded, so ledger cells are fixed numbers rather than
sampling estimates. A cell that moves means the environment moved.

## Mutation testing

The test suites are checked by breaking the source on purpose:

```bash
./.venv/bin/python tools/mutate.py            # every target (slow: ~4 min)
./.venv/bin/python tools/mutate.py rewards    # one module
./.venv/bin/python tools/mutate.py --list     # the catalogue, without running
```

Each mutant runs the whole suite, so a full sweep is 80-odd suite runs. Since
Phase 5 the suite includes the baseline ledger, which dominates the time. Narrow
it with `-k tests/test_rewards.py` while iterating, then do one full sweep before
committing.

`tools/mutate.py` applies each catalogued edit to one file, runs the suite,
restores the file, and reports whether the suite noticed. Exit code is 0 only
when every mutant behaves as catalogued: `expect="killed"` mutants must break
the suite, and no-op **control** mutants must not. A control that gets killed
means the suite is flaky, which invalidates the whole run. The script refuses to
start if the suite is not green first, because otherwise "killed" means nothing.

**This is not optional tooling.** It has caught four bugs that a passing suite
and a careful read both missed:

| phase | bug | why the suite missed it |
|---|---|---|
| 2 | pipetting applied after the batch shift | both orders give identical plate means |
| 3 | assay rng re-created per plate | the test compared plates on *different lots*, so potency alone made them differ |
| 3 | `days_used` frozen at 0 | only "a refusal changes nothing" was pinned, never "a plate advances the clock" |
| 4 | `sign_acc` denominator: true positives vs everything submitted | every sign test had either zero true positives or an exactly-correct hit set, and in both the two sets coincide |

Every one was a test passing for the wrong reason. When a mutant survives, the
fix is a new test that names the signature the mutant destroys — never deleting
the mutant.

A mutant that is **equivalent** (it cannot change behaviour for any reachable
input) is marked `expect="survives"` with a comment explaining why, so the fact
is recorded once instead of rediscovered every run.

### Adding a mutant

When you add a rule that matters, add the mutant that breaks it. Append a
`Mutant` to `CATALOGUE[target]` in `tools/mutate.py`; `edits` is a tuple of
`(old, new)` replacements applied together. Every `old` must appear exactly once
or the mutant reports PATCH-FAILED rather than being silently skipped. Anchor on
code, not on comment prose, so a reworded comment does not quietly disable a
mutant.

## Conventions that the tests enforce

- **All randomness flows from a single `np.random.default_rng(seed)`.** Same
  seed must always produce the same world and the same score. Adding an
  unseeded `np.random` call anywhere will break determinism checks.
- **Difficulty tiers differ only in noise, traps and scarcity** — never in true
  effect size. `tests/test_world.py::test_effect_structure_identical_across_tiers`
  enforces this. See the `Difficulty` docstring for why.
- **Dependencies are pinned exactly** in `requirements.txt`. If you bump one,
  re-run the full suite and the Phase 5 baseline ladder, since the numbers are
  rng-ordering sensitive.
- **World generation and assay noise use two separate rngs** —
  `default_rng(seed)` and `default_rng(seed + 10_000)`. `tests/test_env.py`
  pins the offset and the stream continuity. Collapsing them would make the
  sampled world a function of the agent's plate layouts, and the Phase 5 ledger
  would stop being a comparison between policies.
- **`qc()` is free and must stay free.** Skipping QC has to be a judgment
  failure we can score, not a budget constraint we imposed.
- **`efficiency` is gated on `endpoint > 0.4`,** and the gate is a strict `>`.
  Ungated, banking the budget beats running the experiment. `tools/mutate.py`
  carries four mutants on that one branch.
- **The harness is a reported variable, not a hidden one.** Harness choice moves
  bio-agent pass rates by several points on identical tasks. Every setting that
  could move a score is a named argument recorded in `result.harness`, the API
  client is injected, and scoring is reached only through
  `assaygym.rewards.score`. If you change the harness, two runs stay comparable
  by diffing their `harness` blocks.
- **Ground truth never travels with a dataset row.** `vf_adapter.build_dataset`
  emits `answer: ""` and the grader re-derives the world from `info["seed"]`.
  There is a mutant on this.
- **Diagnostics are never summed into reward.** `decoy_called` is the direct
  measurement of prior-dependence; it has to stay reportable without the reward
  having been tuned against it. Both totals are asserted to reproduce exactly
  from their declared weight tables, so a leak shows up as a mismatch.

## Phase status

| phase | file | tests |
|---|---|---|
| 1 | `assaygym/world.py` | `tests/test_world.py` (15 checks, passing) |
| 2 | `assaygym/assay.py` | `tests/test_assay.py` (17 checks, passing) |
| 3 | `assaygym/env.py` | `tests/test_env.py` (23 checks, passing) |
| 4 | `assaygym/rewards.py` | `tests/test_rewards.py` (24 checks, passing) |
| 5 | `assaygym/policies.py`, `run_baselines.py`, `verify.py` | `tests/test_policies.py` (18 checks, passing) |
| 6 | `assaygym/llm_harness.py`, `vf_adapter.py` | `tests/test_interfaces.py` (17 checks, passing) |

All six phases are built. Each had an acceptance check that passed before the
next began.

`numpy` is the only runtime dependency for phases 1-5. Phase 6's `llm_harness`
additionally needs `anthropic` **only to make a real API call** — the SDK is
imported lazily, so the module imports, tests and scores without it:

```bash
./.venv/bin/pip install anthropic     # optional
```

## Definition of done for a phase

A phase is not finished when the code runs. It is finished when:

1. The phase's acceptance checks are implemented as tests and **pass**.
2. The full suite still passes (`./.venv/bin/python -m pytest tests/ -q`).
2b. `tools/mutate.py` runs clean, with mutants added for whatever the phase
   made load-bearing.
3. **`README.md` is updated** — the build-status table, a dated entry in the
   progress log, and any measured numbers the phase produced. The README is a
   living document that must describe only what actually exists and has been
   measured; targets and roadmap items stay clearly labelled as unverified.

Point 3 is not bookkeeping. A README that overstates what is built is the
fastest way to make the project's central reproducibility claim untrue.
