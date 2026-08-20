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

## Phase status

| phase | file | tests |
|---|---|---|
| 1 | `assaygym/world.py` | `tests/test_world.py` (15 checks, passing) |
| 2 | `assaygym/assay.py` | `tests/test_assay.py` (17 checks, passing) |
| 3 | `assaygym/env.py` | not started |

Each phase has an acceptance check that must pass before the next phase begins.

## Definition of done for a phase

A phase is not finished when the code runs. It is finished when:

1. The phase's acceptance checks are implemented as tests and **pass**.
2. The full suite still passes (`./.venv/bin/python -m pytest tests/ -q`).
3. **`README.md` is updated** — the build-status table, a dated entry in the
   progress log, and any measured numbers the phase produced. The README is a
   living document that must describe only what actually exists and has been
   measured; targets and roadmap items stay clearly labelled as unverified.

Point 3 is not bookkeeping. A README that overstates what is built is the
fastest way to make the project's central reproducibility claim untrue.
