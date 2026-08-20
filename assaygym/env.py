"""AssayGym Phase 3 — the lab bench.

The episode loop. This module is the only thing the agent ever touches: it owns
the budget, hands out the briefing, exposes four tools, and refuses anything it
cannot pay for.

Two rngs, deliberately
----------------------
``reset()`` samples the world from ``default_rng(seed)`` and then builds a
**second, independent** generator ``default_rng(seed + 10_000)`` for assay
noise. World generation is finished before the first plate is ever run, and it
never touches the assay stream, so *which world was sampled* is a function of
the seed alone — not of how many plates the agent ran, how many wells it filled,
or what it put in them.

That independence is what makes the Phase 5 ledger a fair comparison. Four
policies at seed 41 face the identical hidden truth, and the only thing that
differs between their scores is what they did about it. Sharing one stream would
make the world itself a function of the policy's plate layouts, and a score gap
between policies would no longer mean anything.

Free QC
-------
``qc()`` costs no money and no days. This is a design decision, not an
oversight: if an agent skips quality control, that must be a **judgment
failure** we can score, not a budget constraint we imposed on it. Charging for
QC would confound "did not think to check the assay window" with "could not
afford to".
"""

from __future__ import annotations

from collections.abc import Mapping as _MappingABC, Sequence as _SequenceABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from assaygym.assay import WELLS, PlateResult, run_plate, z_prime
from assaygym.world import TIERS, World, override_phenotype_from_deltas, sample_world

__all__ = [
    "PLATE_BASE_COST",
    "PER_WELL_COST",
    "PLATE_DAYS",
    "LOTS",
    "ASSAY_RNG_OFFSET",
    "PRIOR_CAVEAT",
    "TOOL_SPEC",
    "AssayGym",
]


# --- 3.1 Economy -----------------------------------------------------------
PLATE_BASE_COST = 480.0
PER_WELL_COST = 11.0
PLATE_DAYS = 3
LOTS = ["LOT-A", "LOT-B", "LOT-C"]

# Assay-noise rng seed offset. Large enough that adjacent world seeds cannot
# collide with adjacent noise streams.
ASSAY_RNG_OFFSET = 10_000

_WELL_SET = frozenset(WELLS)

# The caveat is load-bearing. The literature prior is wrong on purpose — it adds
# plausible decoys and omits genuine hits. Handing that to an agent without
# saying so would be entrapment, and a score built on entrapment measures
# nothing. Stated plainly, trusting the prior becomes a choice the agent made,
# and holding it responsible for that choice is fair.
PRIOR_CAVEAT = (
    "These previously reported hits come from a different cell background than "
    "the one you are screening. The list may be incomplete or wrong: it can "
    "omit genuine hits and can name genes that do not replicate here. Treat it "
    "as a lead to test, not as an answer. Your submission is graded against "
    "this experiment, not against the literature."
)

_OBJECTIVE = (
    "You are running a gene-knockdown screening campaign under a fixed budget. "
    "Identify (1) which knockdowns move the reporter readout by at least "
    "hit_threshold in absolute value -- the hit set; (2) the direction of each "
    "hit, -1 if the knockdown lowers the readout and +1 if it raises it; and "
    "(3) log10(EC50) in nM for the compound, which acts through exactly one of "
    "the true hits. Buy 96-well plates, decide what goes in every well, then "
    "call submit exactly once. Everything you measure is noisy; the readout you "
    "get is not the truth."
)


def plate_cost(n_wells: int) -> float:
    """Cost of a plate with ``n_wells`` filled wells."""
    return PLATE_BASE_COST + PER_WELL_COST * n_wells


# --- 3.4 Tool definitions, Anthropic tool-use format -----------------------
# One list drives both the LLM harness (Phase 6) and the verifiers adapter
# (Phase 6). If these ever disagree, the harness and the adapter are running
# different environments and no number from either is comparable.
TOOL_SPEC: List[Dict[str, Any]] = [
    {
        "name": "design_and_run",
        "description": (
            "Design one 96-well plate and run it. `layout` maps well -> "
            "condition, e.g. {\"B2\": \"NTC\", \"B3\": \"POS\", \"C4\": "
            "\"KD:SYN03\", \"C5\": \"CMPD@100\"}. Wells are A1-H12. Conditions "
            "are \"NTC\" (negative control), \"POS\" (positive control), "
            "\"KD:<locus>\" for a knockdown, and \"CMPD@<dose>\" for the "
            f"compound at a dose in nM. Costs ${PLATE_BASE_COST:.0f} per plate "
            f"plus ${PER_WELL_COST:.0f} per filled well, and takes "
            f"{PLATE_DAYS} days. The call is refused, and nothing is spent, if "
            "you cannot afford either the money or the days. Empty wells are "
            "free and simply return no measurement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layout": {
                    "type": "object",
                    "description": (
                        "Map of well id (A1-H12) to condition string. Every "
                        "filled well is billed."
                    ),
                    "additionalProperties": {"type": "string"},
                },
                "lot": {
                    "type": "string",
                    "enum": list(LOTS),
                    "description": (
                        "Reagent lot to run this plate on. Lots are not "
                        "guaranteed to be equivalent."
                    ),
                },
            },
            "required": ["layout"],
        },
    },
    {
        "name": "qc",
        "description": (
            "Quality-control summary for one plate you have already run: "
            "control counts, control means, the assay window "
            "(mean(POS) - mean(NTC)), and Z-prime. FREE -- costs no money and "
            "no days, and you may call it as often as you like."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plate_id": {
                    "type": "string",
                    "description": "Id of a plate returned by design_and_run.",
                }
            },
            "required": ["plate_id"],
        },
    },
    {
        "name": "exclude_plate",
        "description": (
            "Drop a plate from your analysis, with a reason. Excluded plates "
            "are not refunded and the days are not returned; exclusion only "
            "affects which data your submission is judged to have rested on."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plate_id": {
                    "type": "string",
                    "description": "Id of a plate returned by design_and_run.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this plate is being dropped.",
                },
            },
            "required": ["plate_id", "reason"],
        },
    },
    {
        "name": "submit",
        "description": (
            "Submit your final answer. ONE SHOT: this ends the episode and "
            "every later tool call is refused. Submit the hit set, the "
            "direction of each hit, and log10(EC50) in nM."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hits": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Loci you believe are genuine hits, e.g. "
                        "[\"SYN02\", \"SYN07\"]. May be empty."
                    ),
                },
                "signs": {
                    "type": "object",
                    "description": (
                        "Direction per submitted hit: locus -> -1 (knockdown "
                        "lowers the readout) or +1 (raises it)."
                    ),
                    "additionalProperties": {"type": "integer"},
                },
                "log_ec50": {
                    "type": ["number", "null"],
                    "description": (
                        "log10(EC50) in nM for the compound. Use null if you "
                        "did not measure it; null scores zero on that term."
                    ),
                },
            },
            "required": ["hits", "signs", "log_ec50"],
        },
    },
]


def _err(message: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"error": message}
    out.update(extra)
    return out


def _finite_or_none(x: float) -> Optional[float]:
    """JSON-safe: nan/inf become None so tool results always serialise."""
    return float(x) if np.isfinite(x) else None


@dataclass
class AssayGym:
    """One episode: one hidden world, one budget, one submission.

    Every tool returns a plain dict. Failures come back as ``{"error": ...}``
    rather than as exceptions, because the same methods are called by an LLM
    over the tool-use API, where a refusal is information the model should get
    to act on rather than a crash.
    """

    seed: int
    tier: str = "standard"

    world: Optional[World] = field(default=None, init=False)
    rng: Optional[np.random.Generator] = field(default=None, init=False)
    plates: Dict[str, PlateResult] = field(default_factory=dict, init=False)
    usd_left: float = field(default=0.0, init=False)
    days_left: int = field(default=0, init=False)
    days_used: int = field(default=0, init=False)
    done: bool = field(default=False, init=False)
    submission: Optional[Dict[str, Any]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(
                f"unknown tier {self.tier!r}; expected one of {sorted(TIERS)}"
            )

    # --- 3.2 reset ---------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Sample the world, arm the budget, and return the briefing."""
        # rng #1: world generation, seeded with `seed` and fully consumed here.
        self.world = override_phenotype_from_deltas(sample_world(self.seed, self.tier))
        # rng #2: assay noise, an independent stream. See the module docstring —
        # this separation is what keeps the sampled world a function of the seed
        # alone rather than of the agent's plate layouts.
        self.rng = np.random.default_rng(self.seed + ASSAY_RNG_OFFSET)

        diff = self.world.diff
        self.plates = {}
        self.usd_left = float(diff.budget_usd)
        self.days_left = int(diff.budget_days)
        self.days_used = 0
        self.done = False
        self.submission = None
        return self.briefing()

    def _require_world(self) -> World:
        if self.world is None or self.rng is None:
            raise RuntimeError("call reset() before using the environment")
        return self.world

    # --- 3.3 the briefing --------------------------------------------------

    def briefing(self) -> Dict[str, Any]:
        """Everything the agent is told, and nothing else."""
        world = self._require_world()
        diff = world.diff
        example_wells = 51
        return {
            "objective": _OBJECTIVE,
            "tier": diff.name,
            "loci": list(world.genes),
            "conditions": {
                "NTC": "Negative control. No knockdown; the plate's own baseline.",
                "POS": "Positive control. A large, known-direction response.",
                "KD:<locus>": (
                    "Knock down one locus, e.g. \"KD:"
                    f"{world.genes[0]}\". Knockdown is never total."
                ),
                "CMPD@<dose>": (
                    "The compound at <dose> nM, e.g. \"CMPD@100\". The compound "
                    "acts through exactly one of the true hits, which is not "
                    "disclosed."
                ),
            },
            "reagent_lots": list(LOTS),
            "literature_prior": {
                "previously_reported_hits": list(world.reported_hits),
                "caveat": PRIOR_CAVEAT,
            },
            "budget": {
                "usd": float(diff.budget_usd),
                "days": int(diff.budget_days),
            },
            "cost_schedule": {
                "plate_base_usd": PLATE_BASE_COST,
                "per_well_usd": PER_WELL_COST,
                "days_per_plate": PLATE_DAYS,
                "formula": "usd = plate_base_usd + per_well_usd * n_filled_wells",
                "example": (
                    f"a {example_wells}-well plate costs "
                    f"${plate_cost(example_wells):,.0f} and takes {PLATE_DAYS} days"
                ),
            },
            "plate_format": {
                "rows": "A-H",
                "columns": "1-12",
                "n_wells": len(WELLS),
                "well_id_example": "C7",
            },
            "hit_threshold": float(world.hit_threshold),
            "tools": [t["name"] for t in TOOL_SPEC],
        }

    # --- 3.4 the four tools ------------------------------------------------

    def design_and_run(
        self, layout: Mapping[str, str], lot: str = "LOT-A"
    ) -> Dict[str, Any]:
        """Validate, charge, run. Any refusal leaves the budget untouched."""
        world = self._require_world()
        if self.done:
            return self._post_submit_error("design_and_run")

        if not isinstance(layout, _MappingABC):
            return _err("layout must be an object mapping well -> condition")
        if not layout:
            return _err("layout is empty; a plate must have at least one filled well")
        if lot not in LOTS:
            return _err(f"unknown lot {lot!r}", valid_lots=list(LOTS))

        # Validate every well and condition BEFORE any budget moves, so a
        # malformed plate is never billed.
        for well, condition in layout.items():
            if well not in _WELL_SET:
                return _err(f"well {well!r} is not on a 96-well plate (A1-H12)")
            if not isinstance(condition, str):
                return _err(f"condition for well {well!r} must be a string")
            try:
                world.condition_value(condition)
            except ValueError as exc:
                return _err(f"well {well!r}: {exc}")

        n_wells = len(layout)
        cost = plate_cost(n_wells)

        # Money and days are checked independently: a plate can be affordable in
        # dollars and unaffordable in days, or the reverse. Reporting both tells
        # the agent which constraint actually bound.
        reasons: List[str] = []
        if cost > self.usd_left:
            reasons.append(
                f"insufficient funds: plate costs ${cost:,.2f}, "
                f"${self.usd_left:,.2f} remaining"
            )
        if PLATE_DAYS > self.days_left:
            reasons.append(
                f"insufficient days: plate takes {PLATE_DAYS} days, "
                f"{self.days_left} remaining"
            )
        if reasons:
            return _err(
                "over budget: " + "; ".join(reasons),
                reasons=reasons,
                cost_usd=cost,
                days_required=PLATE_DAYS,
                usd_left=self.usd_left,
                days_left=self.days_left,
            )

        plate_id = f"P{len(self.plates) + 1}"
        day_run = self.days_used  # day the plate goes on the machine, 0-indexed
        self.usd_left -= cost
        self.days_left -= PLATE_DAYS
        self.days_used += PLATE_DAYS

        result = run_plate(
            world,
            plate_id=plate_id,
            layout=layout,
            lot=lot,
            rng=self.rng,
            day=day_run,
            cost=cost,
        )
        self.plates[plate_id] = result

        return {
            "plate_id": plate_id,
            "lot": lot,
            "values": dict(result.values),
            "n_wells": n_wells,
            "cost_usd": cost,
            "day_run": day_run,
            "usd_left": self.usd_left,
            "days_left": self.days_left,
        }

    def qc(self, plate_id: str) -> Dict[str, Any]:
        """Control summary for one plate. Free: no money, no days. See module docstring."""
        self._require_world()
        if self.done:
            return self._post_submit_error("qc")

        plate = self.plates.get(plate_id)
        if plate is None:
            return _err(
                f"unknown plate id {plate_id!r}",
                known_plate_ids=list(self.plates),
            )

        pos = [plate.values[w] for w, c in plate.layout.items() if c == "POS"]
        neg = [plate.values[w] for w, c in plate.layout.items() if c == "NTC"]
        window = (
            float(np.mean(pos) - np.mean(neg)) if (pos and neg) else None
        )

        return {
            "plate_id": plate_id,
            "lot": plate.lot,
            "excluded": plate.excluded,
            "n_wells": len(plate.layout),
            "n_pos": len(pos),
            "n_ntc": len(neg),
            "mean_pos": float(np.mean(pos)) if pos else None,
            "mean_ntc": float(np.mean(neg)) if neg else None,
            "assay_window": window,
            "z_prime": _finite_or_none(z_prime(pos, neg)),
            "cost_usd": 0.0,
            "days_cost": 0,
            "usd_left": self.usd_left,
            "days_left": self.days_left,
        }

    def exclude_plate(self, plate_id: str, reason: str = "") -> Dict[str, Any]:
        """Drop a plate from analysis. Unknown ids fail loudly and change nothing."""
        self._require_world()
        if self.done:
            return self._post_submit_error("exclude_plate")

        plate = self.plates.get(plate_id)
        if plate is None:
            return _err(
                f"unknown plate id {plate_id!r}; nothing excluded",
                known_plate_ids=list(self.plates),
            )

        already = plate.excluded
        plate.excluded = True
        plate.notes.append(f"excluded: {reason}" if reason else "excluded")
        return {
            "plate_id": plate_id,
            "excluded": True,
            "already_excluded": already,
            "reason": str(reason),
            "n_excluded": sum(p.excluded for p in self.plates.values()),
            "n_active": sum(not p.excluded for p in self.plates.values()),
            "usd_left": self.usd_left,
            "days_left": self.days_left,
        }

    def submit(
        self,
        hits: Sequence[str],
        signs: Optional[Mapping[str, int]] = None,
        log_ec50: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One shot. Records the answer and ends the episode."""
        world = self._require_world()
        if self.done:
            return self._post_submit_error("submit")

        if hits is None:
            hits = []
        if isinstance(hits, str) or not isinstance(hits, _SequenceABC):
            return _err("hits must be a list of locus names")
        if signs is None:
            signs = {}
        if not isinstance(signs, _MappingABC):
            return _err("signs must be an object mapping locus -> -1 or +1")

        # Unknown loci are kept, not dropped: a submission naming a gene that
        # does not exist is a false positive and is scored as one.
        clean_hits: List[str] = []
        for h in hits:
            h = str(h)
            if h not in clean_hits:
                clean_hits.append(h)
        unknown = [h for h in clean_hits if h not in world.genes]

        clean_signs: Dict[str, int] = {}
        for gene, sign in signs.items():
            try:
                clean_signs[str(gene)] = 1 if float(sign) >= 0 else -1
            except (TypeError, ValueError):
                return _err(f"sign for {gene!r} must be -1 or +1, got {sign!r}")

        if log_ec50 is not None:
            try:
                log_ec50 = float(log_ec50)
            except (TypeError, ValueError):
                return _err(f"log_ec50 must be a number or null, got {log_ec50!r}")
            if not np.isfinite(log_ec50):
                return _err("log_ec50 must be finite or null")

        self.submission = {
            "hits": clean_hits,
            "signs": clean_signs,
            "log_ec50": log_ec50,
        }
        self.done = True

        return {
            "submitted": True,
            "hits": list(clean_hits),
            "signs": dict(clean_signs),
            "log_ec50": log_ec50,
            "unknown_loci": unknown,
            "n_plates": len(self.plates),
            "n_excluded": sum(p.excluded for p in self.plates.values()),
            "usd_spent": self.usd_spent,
            "usd_left": self.usd_left,
            "days_left": self.days_left,
        }

    # --- helpers -----------------------------------------------------------

    def _post_submit_error(self, tool: str) -> Dict[str, Any]:
        """The one-shot guard. Returns an error and mutates nothing."""
        return _err(
            f"episode is over: submit() has already been called, so {tool}() "
            "is refused and no state changed",
            done=True,
            usd_left=self.usd_left,
            days_left=self.days_left,
        )

    @property
    def usd_spent(self) -> float:
        return float(sum(p.cost_usd for p in self.plates.values()))

    def call(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        """Dispatch a TOOL_SPEC tool by name. Used by the Phase 6 harness."""
        tools = {
            "design_and_run": self.design_and_run,
            "qc": self.qc,
            "exclude_plate": self.exclude_plate,
            "submit": self.submit,
        }
        fn = tools.get(name)
        if fn is None:
            return _err(f"unknown tool {name!r}", valid_tools=sorted(tools))
        try:
            return fn(**dict(arguments))
        except TypeError as exc:
            return _err(f"bad arguments for {name}: {exc}")

    def state(self) -> Dict[str, Any]:
        """Trajectory summary. Phase 4 scores from this plus the hidden world."""
        return {
            "seed": self.seed,
            "tier": self.tier,
            "plates": list(self.plates.values()),
            "n_plates": len(self.plates),
            "usd_spent": self.usd_spent,
            "usd_left": self.usd_left,
            "days_left": self.days_left,
            "days_used": self.days_used,
            "done": self.done,
            "submission": self.submission,
        }
