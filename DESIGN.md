# AssayGym — design notes

Why this environment is shaped the way it is, what the numbers mean, and what it
cannot do.

**Contents**

1. [The landscape, and three structural limits](#1-the-landscape-and-three-structural-limits)
2. [The design move](#2-the-design-move-sample-truth-first)
3. [The prior trap](#3-the-prior-trap)
4. [Scoring: three numbers, kept apart](#4-scoring-three-numbers-kept-apart)
5. [The baseline ladder](#5-the-baseline-ladder)
6. [Does each artifact earn its place?](#6-does-each-artifact-earn-its-place)
7. [Validating the instruments: mutation testing](#7-validating-the-instruments-mutation-testing)
8. [A clinical variant](#8-a-clinical-variant)
9. [Limits, honestly](#9-limits-honestly)

---

## 1. The landscape, and three structural limits

To train or evaluate a model on scientific reasoning you have to grade its
answers automatically, thousands of times. In biology the ground truth is
normally a physical experiment, so you can't.

The field's answer has been curated benchmarks, and they are genuinely good
work. **LAB-Bench** covers literature QA, protocol reasoning and figure
interpretation over expert-written items. **BixBench** poses open-ended
bioinformatics analyses over real notebooks. **BioML-bench** scores end-to-end ML
pipelines on biological data. **AstaBench** pushes on agentic scientific
workflows with a shared harness. **Latch's scBench-Long** family targets
long-horizon single-cell analysis.

Every one of them shares three limits, and the limits are structural rather than
failures of effort:

1. **They are finite.** A fixed item set is a fixed item set. Once a model has
   seen it — in pretraining, in a paper, in a leaked eval — the number stops
   measuring capability and starts measuring recall.
2. **They are expensive.** Every item costs expert time. That caps the size, and
   it caps how fast the benchmark can be refreshed once it saturates.
3. **Difficulty is a property of whoever wrote the questions.** You cannot turn a
   dial. You cannot ask "how much harder does this get when batch variance
   doubles", because batch variance is not a parameter of a curated item — it is
   baked into whatever data happened to be collected.

The third is the one that matters most for RL. Training needs a difficulty
gradient, and a curated set gives you a single point.

---

## 2. The design move: sample truth first

**Generate the world yourself. Sample a hidden truth first, derive the agent's
observations from it second.**

Because the truth was written down before the agent existed, grading is
arithmetic rather than judgement. Because the world is generated rather than
curated, you get unlimited tasks at zero annotation cost — and difficulty becomes
a dial.

The scenario: an agent runs a gene-knockdown screening campaign under a fixed
budget. It must identify which knockdowns move a reporter readout (the hit set),
the direction of each, and the EC50 of a compound. It buys 96-well plates,
decides what goes in every well, and submits once.

### Effects are imposed, not emergent

A network is sampled to give a plausible baseline, but the effect structure is
then **imposed directly** — three groups, cut from a random permutation:

| group | \|delta\| | what it is |
|---|---|---|
| true hits | 1.5–3.0 × threshold | unambiguously real |
| **gray zone** | 0.60–0.92 × threshold | real but sub-threshold |
| nulls | `normal(0, 0.25 × threshold)` | noise |

`HIT_THRESHOLD = 0.20`, absolute, identical on every tier.

Letting effects emerge from the network would have been more elegant and would
have meant not knowing how hard the task is. Imposing them means the difficulty
is a known quantity rather than an accident of the sampled graph.

**The gray zone is the most important design element after the trap.** These are
genes with real but sub-threshold effects — exactly what a small underpowered
study would have called significant. They are what makes a single well per
condition insufficient, and they are where the reference policy's errors
concentrate.

### Difficulty adds noise, traps and scarcity — never smaller effects

Effect structure is **identical across `clean`, `standard` and `hard`**. What
changes is well noise, pipetting CV, batch variance, edge bias, bad-lot and
contamination probability, the prior trap, and the budget.

This rule is load-bearing. If hard-tier effects were quietly smaller, a low
hard-tier score would be ambiguous between *"the agent designs badly"* and *"the
task is impossible"*, and the tier ladder would measure nothing. With the rule,
the signal is always there to be found; what varies is how much design skill it
takes to find it.

### Six artifacts, each punishing one missing skill

An artifact that punishes nothing is just noise, and noise alone makes a task
harder without making it more diagnostic.

| # | artifact | punishes |
|---|---|---|
| 1 | reagent lot potency | never running positive controls |
| 2 | pipetting error (multiplicative) | trusting a single well instead of replicating |
| 3 | batch shift (once per plate) | comparing plates without an on-plate control |
| 4 | edge bias | filling the plate from A1 |
| 5 | contamination (one quadrant) | bunching all controls in one corner |
| 6 | measurement noise | reading one well as truth |

The order of application is not free to rearrange. Lot potency multiplies the
*effect above baseline* rather than the raw value, so a degraded lot shrinks the
assay window instead of translating the plate — the collapsed positive-control
window is the only signal that anything is wrong. Pipetting error is
multiplicative and lands *before* the additive batch shift, which keeps a
per-plate offset separable from per-well scatter.

Both orderings leave the observations superficially plausible while destroying
what the agent can detect. Section 7 is about how that was caught.

### The budget is a real allocation decision

A plate costs `$480 + $11/well` and takes 3 days. The reference policy's balanced
layout on `hard` is 8 NTC + 3 POS + 24 KD + 16 dose wells = **51 wells = $1,041**,
so `hard`'s `$3,300` and 9 days buys **exactly three plates** — `$3,123` and 9 of
9 days. Every well spent on a replicate is a well not spent on a dose point.

---

## 3. The prior trap

Published work shows agents will run every correct analysis step and then answer
from memorised priors instead of from the data in front of them. So the
environment hands the agent a "previously reported hits" list that is **wrong on
purpose**: it adds plausible false hits (decoys) and removes real ones
(omissions). The trap punishes a trusting agent twice — false positives from the
decoys, false negatives from the omissions.

This converts *"is the answer grounded in this experiment"* from a hoped-for
property into a scored quantity.

### Decoys come from the gray zone first

This detail is worth defending. A plausible false published result is **not** a
gene with obviously zero effect — it is one sitting at 0.7× threshold, the kind
of thing a real literature actually contains because an underpowered study
called it significant. Decoys are therefore drawn from the gray zone first,
falling back to nulls only when the gray zone is too small.

A decoy drawn from the nulls would be trivially rejected by any policy that ran a
plate, and the trap would measure nothing.

### The caveat is not optional

The briefing states in so many words that the prior comes from **a different cell
background and may be incomplete or wrong**. Without that sentence the trap is
entrapment, and a score built on entrapment measures nothing. With it, trusting
the prior is a choice the agent made, and holding it responsible is fair.

`decoy_called` — how many decoys the agent repeated back — is the direct
measurement of prior-dependence and the number this environment exists to
produce. It is computed and **never summed into reward**, so it can be reported
without the reward having been tuned against it.

---

## 4. Scoring: three numbers, kept apart

| number | shape | for |
|---|---|---|
| **`strict_pass`** | all or nothing | **the headline** |
| `endpoint` | sparse, verifiable | `hit_f1` 0.55 + `sign_acc` 0.15 + `ec50` 0.30 |
| `shaped` | dense, mechanical | `endpoint` 0.55 + five process terms |

**Lead with `strict_pass`.** `endpoint` gives partial credit generously enough
that a policy which parrots the literature and runs **zero plates** measures
**0.660** on `clean`. Quote that and the environment looks far weaker than it is.

`hit_f1` is F1 rather than precision or recall: precision alone is farmed by
submitting one confident gene, recall alone by submitting every gene.
`sign_acc` is computed over **correctly-identified hits only** and is 0.0 — never
a vacuous 1.0 — when none were identified, because vacuous truth there would pay
an agent for submitting nothing.

### Every process term is a checkable fact

| term | weight | the fact |
|---|---|---|
| `controls` | 0.10 | fraction of non-excluded plates with ≥4 NTC and ≥2 POS |
| `replication` | 0.10 | fraction of *submitted* hits measured on ≥2 distinct non-excluded plates |
| `self_normalizable` | 0.05 | fraction of plates with ≥1 NTC alongside ≥1 test condition |
| `qc_hygiene` | 0.12 | bad-lot plates excluded, minus good plates wrongly excluded |
| `efficiency` | 0.08 | budget unspent, gated on `endpoint > 0.4` **and** `n_plates > 0` |

Count the control wells. Check the plate ids. Compare the lots against the ones
the world degraded. **A persuasive transcript can flatter an LLM judge, but it
cannot retroactively put control wells on a plate that was already run.** That is
the whole argument for this over rubric-graded shaping.

### The efficiency gate needed two conditions, and that was measured

The spec gates `efficiency` on `endpoint > 0.4` so that banking the budget does
not beat running the experiment. Measured, that gate alone is insufficient: the
spec itself notes a prior-parrot scores 0.4–0.67 on `endpoint`, so a zero-plate
policy clears it on **100.0%** of `clean` seeds, **59.5%** of `standard` and
**37.0%** of `hard`. Adding `n_plates > 0` closes it and costs a real experiment
nothing.

One place the spec was genuinely silent — `qc_hygiene` with **zero plates run** —
is resolved to 0.0 rather than the literal formula's 1.0, on the same principle:
process credit has to require process. A policy that ran plates and simply had no
bad lot to catch still gets `caught = 1.0`, so the metric still reduces to a pure
over-exclusion penalty.

---

## 5. The baseline ladder

**This is the section that makes the rest credible.** Before letting any model
near the environment: does the reward separate competence from noise? Four
scripted policies, the same 200 seeds per tier.

### `strict_pass`, n = 200 per cell, mean ± standard error

| tier | random | prior_parrot | naive_screen | competent_doe |
|---|---|---|---|---|
| clean | 0.000 ± 0.000 | 0.020 ± 0.010 | 0.280 ± 0.032 | **1.000 ± 0.000** |
| standard | 0.000 ± 0.000 | 0.010 ± 0.007 | 0.080 ± 0.019 | **0.640 ± 0.034** |
| hard | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.020 ± 0.010 | **0.175 ± 0.027** |

All 12 cells are inside the ±0.05 acceptance tolerance; worst deviation
**0.020**. Episodes are fully seeded, so these are fixed numbers rather than
sampling estimates — a cell that moves means the environment moved.

### How to read these

- **Monotone in every tier.** More real experimental design → higher score. This
  is the property that has to hold, and it holds on all three rows.

- **`clean/competent_doe = 1.000` is the load-bearing cell, and the one people
  skip.** It proves the task is **well-posed**: strip the artifacts and a correct
  policy solves it every single time, over 200 seeds, with zero variance. So
  difficulty on the other tiers comes from noise and traps — not from an
  ambiguous objective, not from an unfair scoring rule, not from a bug.
  **Without this cell, a low hard-tier score is uninterpretable**: it could
  equally mean the environment is broken. Every other number here depends on it.

- **`hard/competent_doe = 0.175` is good news.** An environment its own reference
  policy saturates is useless for RL — there would be nothing left to learn.
  Section 6 names the specific unexploited strategy.

- **The trap works.** `prior_parrot` scores 0.020 / 0.010 / 0.000 despite being
  handed most of the right answer. It still calls at least one decoy on
  **100%** of `hard` episodes.

- **`endpoint` and `shaped` are reported, not led with.** `shaped` is
  deliberately *not* monotone between `random` and `prior_parrot` on `standard`
  and `hard` (0.290 vs 0.280; 0.299 vs 0.214). That is correct: `shaped` pays for
  process, `random` ran a plate, `prior_parrot` ran none.

### The degenerate exploit

Calling every gene a hit — the classic way to farm recall — scores
`strict_pass` **0.000** on all three tiers with recall exactly 1.000 and
`endpoint` 0.324–0.373, against `competent_doe`'s 0.763–0.956. F1 caps the
payoff and exact set equality does the rest.

### Prior-dependence, fraction of episodes calling ≥1 decoy

| tier | random | prior_parrot | naive_screen | competent_doe |
|---|---|---|---|---|
| standard | 0.460 | 0.680 | 0.355 | 0.220 |
| hard | 0.665 | **1.000** | 0.610 | **0.470** |

The reference policy still falls for the trap on 47% of `hard` episodes.

---

## 6. Does each artifact earn its place?

The ladder shows the reward is monotone in competence. It does **not** show that
each artifact is doing work. So the reference policy was run again with one
design step disabled at a time, **n = 1000 per cell**.

*(Run first at n = 200, where the standard error on a delta is ±0.048 and cannot
resolve a 0.04 effect. Four rows read as "costs nothing" that do not. The n = 200
version of this table would have been wrong.)*

### Cost of removing one defence, `strict_pass` (positive = removing it hurts)

| defence removed | standard (full = 0.628) | hard (full = 0.158) |
|---|---|---|
| one replicate instead of two | **+0.147 ± 0.022** (6.7 SE) | **+0.072 ± 0.015** (4.8 SE) |
| full plate incl. perimeter | **+0.095 ± 0.022** (4.3 SE) | **+0.039 ± 0.015** (2.6 SE) |
| no per-plate NTC normalisation | **+0.076 ± 0.022** (3.5 SE) | **+0.033 ± 0.016** (2.1 SE) |
| no contaminated-quadrant flag | +0.044 ± 0.022 (2.0 SE) | +0.008 ± 0.016 (0.5 SE) |
| no lot comparison / exclusion | **−0.074 ± 0.021** | **−0.056 ± 0.017** |
| no QC of any kind | −0.020 ± 0.021 | −0.038 ± 0.017 |

Three defences clearly earn their place on both tiers. **Two rows are negative,
and they are saying different things.** Both are reported rather than hidden,
because an ablation where everything degrades tells you only that you built what
you designed.

### Lot exclusion is *redundant*, not broken

The detector works:

| | standard | hard |
|---|---|---|
| worlds with a degraded lot | 0.381 | 0.562 |
| bad-lot plates correctly excluded | **0.895** | **0.801** |
| good plates wrongly excluded | 0.003 | 0.030 |
| mean usable plates after exclusion | 2.656 | 2.520 |

The problem is downstream. `competent_doe` pools three plates and takes the
**median** per condition, and a median over six measurements already survives two
of them being shrunk by a bad lot. So excluding buys no accuracy and costs a
third of the data.

*(That the median is the mechanism is an **interpretation** of the measurements
above — the accuracy figures and the plate counts are measured; the causal story
linking them is not separately tested.)*

Exclusion still pays on `shaped`, via `qc_hygiene`: 0.873 vs 0.835 on standard,
0.774 vs 0.724 on hard. **The two rewards disagree about lot exclusion and both
are right.** `shaped` credits the process — you noticed the collapsed window and
dropped the plate. `strict_pass` reports that in this particular analysis it did
not change the answer. That tension is real and worth stating rather than
papering over: process reward and outcome reward are measuring different things,
which is the entire reason for keeping them as separate numbers.

### Contamination detection is *under-powered* on hard — and that is identified headroom

A different failure. Here the defence itself degrades:

| | standard | hard |
|---|---|---|
| contaminated plates seen | 451 | 716 |
| correctly detected | **0.863** | **0.508** |
| spurious flags (of 1349 / 1084 clean plates) | 2 | 29 |

Two NTC per quadrant against `hard`'s noise (`well_noise` 0.13, `pipet_cv` 0.09)
gives a quadrant mean with a standard error near **0.11**, so the 0.25 detection
floor sits barely two standard errors from the 0.45 contamination offset. The
artifact is not weak — the policy's layout cannot resolve it.

**The fix is more control wells, and more control wells cost wells.** That is
precisely the allocation trade-off the budget exists to force. So
`competent_doe` is leaving real score on the table through a layout choice it
never reconsiders, and no scripted policy here explores it.

This is a stronger claim than "there is headroom" in the abstract: the specific
unexploited strategy can be named, the mechanism quantified, and the cost of the
trade priced in wells. It is the most concrete thing in this document about what
an RL policy could learn that the reference policy does not know.

---

## 7. Validating the instruments: mutation testing

A test suite that has never failed has not been shown to test anything. So the
source is broken on purpose, one edit at a time, and the suite is checked for
whether it notices — `tools/mutate.py`, ~100 catalogued mutants across six
modules, including no-op **control** mutants that must *survive* (a control that
gets killed means the suite is flaky, which invalidates the whole run) and a
green-baseline precondition (without it, "killed" means nothing).

**It found three bugs that passing test suites and careful reading both missed.**
Each was a test passing for the wrong reason.

1. **Pipetting applied after the batch shift (Phase 2).** Both orders produce
   identical plate means, so nothing in the suite could separate them. Closed by
   measuring the correlation between a plate's mean and its internal spread —
   ~0 for the correct order, ~+1 for the inverted one.

2. **The assay rng re-created per plate (Phase 3),** giving every plate in an
   episode identical noise. The determinism test compared plates run on
   **different reagent lots**, and the potency difference alone made the values
   differ — masking the repeated noise entirely. Closed by reproducing two
   consecutive plates from a single external generator and comparing *same-lot*
   plates, which pins stream continuity rather than mere inequality.

3. **`days_used` frozen at 0 (Phase 3),** so every plate reported `day_run = 0`
   and a campaign looked simultaneous. The suite pinned only that a *refusal*
   leaves the clock alone; nothing asserted that a successful plate advances it.

**The third one corrected a number that had already been reported.** The Phase 3
write-up claimed 29/29 mutants caught. It was 28/29 — the harness had been
session-scoped at the time, and the survivor was only found once it was promoted
into the repo and re-run across every phase. The claim was wrong, it was
corrected in the README and in the commit history, and the correction is left
visible here rather than quietly fixed.

Two further gaps were found the same way while building later phases:
`sign_acc` computed over everything submitted rather than over
correctly-identified hits (every existing sign test had either zero true
positives or an exactly-correct hit set, and in both those cases the two sets
coincide), and the harness's forced empty submission (deleting the `submit()`
call left the score unchanged, because the scorer already reads a missing
submission as an empty one).

One catalogued mutant is marked **equivalent** and expected to survive: the
prior trap's `min(n_omitted, n_hits - 1)` guard is unreachable at every shipped
tier, since `n_omitted` (0/1/2) is always below `n_true_hits` (3/3/4). No test
can kill it and none should be written to. Recording that once beats
rediscovering it every run.

This is the strongest evidence in the project that the instruments were
**validated rather than trusted**. The ladder in section 5 is only worth
anything if the code that produced it does what it claims, and "the tests pass"
is not evidence of that — it is evidence that the tests pass.

---

## 8. A clinical variant

The same construction transfers to any domain where the truth can be sampled
before the observations are derived from it. A clinical sketch:

- **Hidden truth**: a patient's underlying condition, drawn with a known
  prevalence, plus per-test sensitivity and specificity.
- **Observations**: test results generated from that truth through the known
  error rates, with cost and turnaround time per test.
- **Budget**: money and days, as here.
- **The prior trap**: a "typical presentation" summary that is wrong in a
  clinically plausible direction — the base-rate-neglect analogue of a
  gray-zone decoy.
- **Scoring**: exact diagnosis (`strict_pass`), partial credit for a ranked
  differential (`endpoint`), and process terms that are checkable facts about the
  trajectory — did the agent order a confirmatory test before committing, did it
  order tests whose results could not change the decision.

The transferable part is not the biology. It is the four properties: truth
precedes observation, difficulty is a parameter rather than a property of the
item set, process terms are facts about the trajectory rather than opinions about
it, and the whole thing is gated on a scripted-baseline ladder before any model
is allowed near it.

---

## 9. Limits, honestly

- **The biology is a caricature.** Linear propagation through a DAG over invented
  loci (`SYN01`, `SYN02`, …), not a model of any real regulatory network. Gene
  names are synthetic on purpose and always will be. This tests experimental
  *reasoning* under noise and budget — it is not biological realism, and it
  **cannot be validated against a real screen as-is**.

- **The artifact parameters are priors, not fitted values.** Batch variance, edge
  bias, plate cost, gray-zone width, contamination offset: all chosen to be
  plausible, **none calibrated against real plate data**. Calibrating them
  against real screening data is the first thing a domain expert should be asked
  for, and until that happens the tier labels are internal comparisons rather
  than statements about real difficulty.

- **No gene-gene interactions, no time course, no inventory or sample-tracking
  layer.** No epistasis: effects are additive and imposed per gene, so a policy
  never has to reason about one knockdown changing another's phenotype. No
  temporal dimension at all — a plate is a single readout, not a trajectory. No
  consumable tracking, no sample mix-ups, no instrument drift across weeks.

- **No language model has been run against this yet.** Phases 1–5 prove the
  environment is well-posed and that the reward separates *scripted* competence
  from noise. Whether it separates *model* competence is untested. The harness
  in `llm_harness.py` is verified against stub clients only — it has never made
  a network call.

- **The `verifiers` integration is unverified against a real install.** The
  adapter's fallback path is fully tested; the wrap path is guarded so a version
  mismatch degrades visibly to the fallback dict rather than raising, but it has
  not been run against the actual training stack.

- **The reference policy is not a ceiling.** Section 6 names where it leaves
  score on the table. Treat `hard/competent_doe = 0.175` as a scripted baseline.

Writing this section before it was needed is the difference between an engineer
who knows what he built and someone who thinks he did biology.
