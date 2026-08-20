# AssayGym

A synthetic RL environment and eval for **wet-lab experimental design**. An agent
runs a gene-knockdown screening campaign under a fixed budget: it buys 96-well
plates, decides what goes in every well, fights six realistic sources of assay
artifact, and submits one answer. Grading is arithmetic, because the ground
truth was written down before the agent existed.

> **Status: Phases 1-2 of 6 complete.** The world model (hidden ground truth
> plus the prior trap) and the observation model (six assay artifacts) are built
> and tested, with 32 passing checks. The environment loop, scoring, baselines
> and LLM harness are **not built yet**.
>
> This README is a living document: it describes only what actually exists and
> has been measured, and it is updated at the end of every phase. See the
> [Progress log](#progress-log) for what landed when, and
> [What is not proven yet](#what-is-not-proven-yet) before citing anything here.

---

## The problem this exists to solve

To train or evaluate an AI on scientific reasoning you need to grade its answers
automatically, thousands of times. In biology the ground truth is normally a
physical experiment, so you can't. Curated benchmarks work around this with a
fixed set of human-annotated questions, which buys correctness at the cost of
being finite, expensive, and eventually memorised.

**The move: generate the world yourself.** Sample a hidden truth first, derive
the agent's observations from it second. Because the truth precedes the agent,
grading is a comparison rather than a judgement call. And because the world is
generated rather than curated, you get unlimited tasks at zero annotation cost,
and difficulty becomes a dial instead of a property of whichever questions
someone happened to write down.

### The one original mechanic: the prior trap

Published work shows agents will run every correct analysis step and then answer
from memorised priors instead of from the data in front of them. So the
environment hands the agent a "previously reported hits" list that is **wrong on
purpose**: it adds plausible false hits and omits real ones.

An agent that answers from memory fails. This converts "is it grounded in *this*
experiment" from a hoped-for property into a scored quantity.

The environment always ships the prior with an explicit caveat that it comes
from a different cell background and may be incomplete or wrong. Without the
caveat the trap is entrapment; with it, trusting the prior is a choice the agent
made and the score is fair.

---

## Quickstart

Requires Python 3.11+ (developed on 3.13.7). `numpy` is the only runtime
dependency.

```bash
git clone https://github.com/Abhishekjani2509/assaygym.git
cd assaygym
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pytest tests/ -q -s
```

Expected: `32 passed`. The `-s` flag prints the measured values the checks
assert on, since those figures are the point.

Dependencies are pinned exactly (`numpy==2.5.2`, `pytest==9.1.1`). The project's
whole claim is that its numbers are reproducible, so the environment is part of
the artifact. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Sampling a world

```python
from assaygym import sample_world, override_phenotype_from_deltas

w = override_phenotype_from_deltas(sample_world(7, "standard"))

w.true_hits       # ['SYN03', 'SYN07', 'SYN09']   <- the answer
w.true_signs      # {'SYN03': 1, 'SYN07': -1, 'SYN09': -1}
w.gray_zone       # ['SYN06', 'SYN10']            <- real but sub-threshold
w.reported_hits   # ['SYN03', 'SYN06', 'SYN09', 'SYN10']  <- what the agent is told

w.decoys          # ['SYN06', 'SYN10']  false hits added to the prior
w.omitted         # ['SYN07']           a real hit hidden from the prior
```

Read that last block carefully, because it is the entire design in one example.
The agent is told four genes are hits. Two of them (`SYN06`, `SYN10`) are
gray-zone genes with real but sub-threshold effects, and calling them is wrong.
One real hit (`SYN07`) is missing from the prior entirely, and it happens to be
the compound's target. An agent that trusts the list scores two false positives
and one false negative. An agent that runs the experiment properly can recover
all three.

Noise-free ground truth for any condition:

```python
w.condition_value("NTC")        # 1.1694  negative control = baseline
w.condition_value("POS")        # 2.0700  positive control, well above any real effect
w.condition_value("KD:SYN03")   # 1.4808  knockdown of a true hit
w.condition_value("CMPD@300")   # 0.9823  compound at 300 nM, on the dose-response curve
```

`sample_world(seed, tier)` is deterministic: the same seed always yields the
same world.

---

## What Phase 1 gives you

`assaygym/world.py` is the sealed envelope. Everything in it is hidden from the
agent forever; the agent only ever sees noisy measurements derived from it.

### Difficulty tiers

| field | clean | standard | hard |
|---|---|---|---|
| `n_genes` | 8 | 10 | 12 |
| `n_true_hits` | 3 | 3 | 4 |
| `n_gray` | 0 | 2 | 3 |
| `well_noise` | 0.02 | 0.07 | 0.13 |
| `pipet_cv` | 0.02 | 0.05 | 0.09 |
| `batch_sigma` | 0.02 | 0.12 | 0.20 |
| `edge_bias` | 0.0 | 0.10 | 0.14 |
| `p_bad_lot` | 0.0 | 0.35 | 0.55 |
| `p_contamination` | 0.0 | 0.25 | 0.40 |
| `p_prior_trap` | 0.0 | 0.70 | 1.0 |
| `n_decoys` | 0 | 2 | 2 |
| `n_omitted` | 0 | 1 | 2 |
| `budget_usd` | 6000 | 4500 | 3300 |
| `budget_days` | 18 | 12 | 9 |

The hard-tier budget is deliberate: a plate costs ~$1,041 and takes 3 days, so
$3,300 and 9 days buys **exactly three plates**. Plate layout becomes a real
allocation decision — every well spent on a replicate is a well not spent on a
dose point.

### Effect structure

`HIT_THRESHOLD = 0.20`, absolute and identical across every tier. Genes are
partitioned into three groups:

- **true hits** — `|delta|` is 1.5x to 3.0x threshold. Unambiguously real.
- **gray zone** — `|delta|` is 0.60x to 0.92x threshold. Real but sub-threshold:
  the things a small underpowered study would have called significant. This is
  the most important design element after the trap, and it is what makes a
  single well per condition insufficient.
- **nulls** — the rest, `normal(0, 0.25 x threshold)`.

---

## Four design decisions that are load-bearing

Each of these guards a specific failure mode that makes the environment stop
measuring anything. Each is pinned by a test, with the measured value.

### 1. Hit signs are drawn uniformly from {-1, +1}

In an early build signs skewed negative, and a policy that blindly guessed
"every knockdown lowers the readout" scored 77% on direction **without running a
single plate**. Randomising makes that guess worth exactly 50%.

Measured: negative-sign fraction **0.4667** over 600 hits (200 standard seeds),
converging to **0.4966** over 60,000 draws. Per-hit-slot fractions are 0.5004 /
0.4966 / 0.4928, so there is no structural bias — the low 200-seed value is a
localized fluctuation, and it runs in the harmless direction anyway (a blind
"everything goes down" guess scores *below* chance on those seeds).

### 2. `true_log_ec50` spans the full testable range

`uniform(0.5, 3.5)` — 3 nM to ~3 µM. In an early build the range was narrow
enough that a fixed blind guess landed inside scoring tolerance about a third of
the time, inflating the do-nothing policy.

With a scoring tolerance of ±0.4 log units against a range of width 3.0, the
best possible fixed blind guess wins on `0.8/3.0 = 26.7%` of worlds and no more.

Measured at 200,000 seeds: **0.2672 / 0.2678 / 0.2679** for clean / standard /
hard, all within 1.2σ of analytic. The draw is uniform, not merely bounded — KS
statistic 0.00174 against a 95% critical value of 0.00304, decile chi-square
10.29 on 9 df against a critical 16.92.

### 3. Tiers differ only in noise, traps and scarcity — never in effect size

Difficulty increases by adding noise, traps and scarcity. It **never** shrinks
true effect sizes; the effect structure is byte-identical across all three
tiers.

The reason: a score gap between tiers must be a statement about the agent's
experimental design, not about the signal having been quietly made undetectable.
Without this rule, a low hard-tier score is ambiguous between "the agent designs
badly" and "the task is impossible," and the tier ladder measures nothing. The
signal is always there to be found; what varies is how much design skill it
takes to find it.

Enforced by `test_effect_structure_identical_across_tiers`.

### 4. Decoys come from the gray zone first

A plausible false published result is not a gene with an obviously zero effect.
It is one sitting at ~0.7x threshold. Gray-zone decoys are the things that would
genuinely appear in a literature, which is what makes the prior tempting rather
than obviously broken.

Decoys fall back to null genes only when the gray zone is too small to supply
them. Measured: **272/272** decoys drawn from the gray zone on standard. No
shipped tier can exhaust its gray zone, so a test registers a deliberately
gray-starved tier to cover the fallback path.

---

## Measured Phase 1 numbers

200 seeds per tier unless noted, reproducible with the quickstart command.

| check | target | measured |
|---|---|---|
| negative-sign fraction | 0.50 ± 0.05 | **0.4667** (600 hits) |
| prior-trap rate, standard | ~0.70 | **0.680** |
| prior-trap rate, hard | 1.0 | **1.000** |
| decoys drawn from gray zone | gray first | **1.000** (272/272) |
| bad-lot rate, standard | 0.35 | **0.360** |
| blind EC50 guess inside tolerance | 0.267 | **0.2725** (2,000 seeds) |
| `true_log_ec50` range | uniform(0.5, 3.5) | min 0.506, max 3.480, mean 2.008 |

Determinism holds: the same seed twice produces identical `true_hits`,
`true_signs`, `true_log_ec50`, `true_delta` and network.

## Measured Phase 2 numbers

All figures below are measured from real runs. Where an analytic expectation
exists it is named as such alongside, never in place of a measurement.

| check | analytic expectation | measured |
|---|---|---|
| degraded-lot window ratio (potency 0.4) | 0.400 | **0.3984 ± 0.0012** (SE, n=400 paired plates) |
| same, full noise + contamination live | 0.400 | **0.4015 ± 0.0022** (SE, n=400) |
| same, noise-free | 0.400 | **0.400000** (exact) |
| contaminated-quadrant offset | 0.45 | **0.4509 ± 0.0023** (SE, n=400 plates) |
| contamination rate, standard | 0.25 | **0.2477 ± 0.0031** (SE, n=20,000 plates) |
| zero-noise passthrough error | 0 | **0.0** exactly, all 96 wells |
| edge minus interior, noise-free | 0.10 | **0.100000** |
| batch: per-well spread of plate difference | 0 | **1.7e-16** |
| batch: sd of the offset itself | 0.1697 | **0.1700** (n=200 plate pairs) |
| pipetting scatter, sd/mean | 0.05 both | **0.0500** (NTC), **0.0499** (POS) |

Two structural checks worth calling out. Within-plate scatter between two plates
measures **0.1051** against **0.1058** predicted if the batch shift is per-plate,
and **0.2000** if it were per-well — so the batch term provably does not leak
into per-well variance. And the correlation between a plate's mean and its
internal spread is **-0.0267**, where the inverted pipetting/batch order would
drive it to roughly +1.

---

## Build status

| phase | file | what it does | status |
|---|---|---|---|
| 1 | `assaygym/world.py` | hidden ground truth + the prior trap | **done**, 15 checks passing |
| 2 | `assaygym/assay.py` | observation model — the six artifacts | **done**, 17 checks passing |
| 3 | `assaygym/env.py` | episode loop, tool API, budget | not started |
| 4 | `assaygym/rewards.py` | scoring | not started |
| 5 | `assaygym/policies.py` | four scripted baselines | not started |
| 6 | `assaygym/llm_harness.py`, `vf_adapter.py` | Anthropic tool-use + verifiers adapter | not started |

Each phase has an acceptance check that must pass before the next one starts.
Phase 5 is the gate: if the baseline ladder is not monotone, the bug is in the
world model or the scoring, not in the policy.

### Progress log

**2026-08-19 — Phase 1: `world.py`, the sealed envelope.**
Hidden ground-truth generator. `Difficulty` tiers (clean / standard / hard), the
`World` dataclass, `sample_world(seed, tier)`, and
`override_phenotype_from_deltas` installing a noise-free `condition_value`
closure for `NTC`, `POS`, `KD:<locus>` and `CMPD@<dose>`. Effects are imposed
rather than emergent, so task difficulty is known exactly instead of being an
accident of the sampled network. Prior trap implemented with gray-zone-first
decoys and hit omission. 15 acceptance checks, all passing; measured figures in
[Measured Phase 1 numbers](#measured-phase-1-numbers). Environment pinned
in-repo and verified from a clean clone.

**2026-08-20 — Phase 2: `assay.py`, the dirty window.**
The observation model. `run_plate` applies the six artifacts in a fixed order,
plus geometry helpers (`is_edge`, `quadrant`) and `z_prime`. 17 acceptance
checks. Measured: a degraded lot (potency 0.4) shrinks the assay window to
**0.3984 ± 0.0012** of the same plate on a good lot; contaminated quadrants sit
**0.4509 ± 0.0023** above identical conditions elsewhere; zero-noise passthrough
is bit-exact. Verified by mutation testing that the suite catches all six
plausible ordering bugs — see [The six artifacts](#the-six-artifacts).

*Next: Phase 3, `env.py` — the episode loop, tool API and budget.*

### The six artifacts

Each exists to punish one specific missing experimental skill. An artifact that
punishes nothing is just noise, and noise alone makes the task harder without
making it more diagnostic.

| # | artifact | punishes |
|---|---|---|
| 1 | reagent lot potency | never running positive controls |
| 2 | pipetting error (multiplicative) | trusting a single well instead of replicating |
| 3 | batch shift (once per plate) | comparing across plates without an on-plate control |
| 4 | edge bias | filling the plate from A1 |
| 5 | contamination (one quadrant) | bunching all controls in one corner |
| 6 | measurement noise | reading one well as truth |

**The application order is load-bearing.** Lot potency multiplies the *effect
above baseline*, not the raw value, so a degraded lot shrinks the assay window
rather than translating the plate — the collapsed positive-control window is the
only signal an agent has that anything is wrong. Pipetting error is
multiplicative and lands *before* the additive batch shift, which keeps a
per-plate offset separable from per-well scatter; inverted, it would scale the
batch offset too and couple the two.

Both errors would leave the observations superficially plausible while
destroying what the agent can detect, so the suite pins the order down
explicitly rather than trusting it. Confirmed by mutation testing: making the
lot multiply the raw value, moving pipetting after the batch shift, drawing the
batch shift or the contaminated quadrant per well, making pipetting additive, or
inverting the edge test each break at least one test, while a no-op control
change correctly breaks none.

---

## What is not proven yet

Phase 5 is the phase that would make this credible: run scripted baseline
policies and check that the reward actually separates competence from noise.
**It has not been run.** The intended ladder is a design target, not a result:

| tier | random | prior_parrot | naive_screen | competent_doe |
|---|---|---|---|---|
| clean | 0.000 | 0.040 | 0.275 | **1.000** |
| standard | 0.000 | 0.015 | 0.075 | **0.620** |
| hard | 0.000 | 0.000 | 0.005 | **0.165** |

Treat every cell as unverified until `run_baselines.py` exists and prints them.
The two numbers that will matter most: `clean/competent_doe = 1.000` would prove
the task is *well-posed* (strip the artifacts and a correct policy solves it
every time, so difficulty elsewhere comes from noise and traps rather than an
ambiguous objective), and `prior_parrot` near zero would prove the trap works.
If `prior_parrot` scores well above 0.05, something is wrong.

---

## Limits, honestly

- **The biology is a caricature.** Linear propagation through a DAG over
  invented loci (`SYN01`, `SYN02`, …), not a model of any real regulatory
  network. Gene names are synthetic on purpose and always will be. This tests
  experimental *reasoning* under noise and budget — it is not biological
  realism, and it cannot be validated against a real screen as-is.
- **The artifact parameters are priors, not fitted values.** Batch variance,
  edge bias, plate cost, gray-zone width: all chosen to be plausible, none
  calibrated against real plate data. Calibrating them is the first thing a
  domain expert should be asked for.
- **No gene-gene interactions, no time course, no inventory or sample-tracking
  layer.**
- **Phases 2-6 do not exist yet**, so nothing here has been shown end-to-end.

---

## Repo map

```
assaygym/
  __init__.py
  world.py           Phase 1 - hidden ground truth + the prior trap
  assay.py           Phase 2 - observation model, the six artifacts
tests/
  test_world.py      Phase 1 acceptance suite (15 checks)
  test_assay.py      Phase 2 acceptance suite (17 checks)
requirements.txt     pinned: numpy 2.5.2, pytest 9.1.1
CONTRIBUTING.md      setup, test commands, invariants the suite enforces
```

## Conventions

- All randomness flows from a single `np.random.default_rng(seed)`. Same seed,
  same score, always. This is tested.
- Loci are synthetic (`SYN01`, `SYN02`, …). Never real gene names.
- Plates are 96-well: rows `A`-`H`, columns `1`-`12`, wells like `"C7"`.
