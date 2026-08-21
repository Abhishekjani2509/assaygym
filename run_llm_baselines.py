#!/usr/bin/env python3
"""Run real models against AssayGym and print their ledger next to the scripted one.

    # 1. smoke test: one episode, confirm the schemas work end to end
    ./.venv/bin/python run_llm_baselines.py --smoke

    # 2. the sweep, with a hard spend cap
    ./.venv/bin/python run_llm_baselines.py \
        --models claude-sonnet-5 claude-haiku-4-5 \
        --n 10 --max-usd 12 --json llm_results.json

Requires `ANTHROPIC_API_KEY` (or an `ant auth login` profile) and
`pip install anthropic`.

**The spend cap is checked before every episode and is not advisory.** Episode
cost grows roughly quadratically in turns, because the whole transcript is
resent each turn and a plate result is ~50 floats, so a model that flails
through all 24 turns costs far more than one that runs three plates and
submits. The cap stops the sweep cleanly and reports partial results rather
than silently continuing.

Read the output next to `run_baselines.py`, not instead of it. `competent_doe`
is a hand-written oracle that already knows which defences to apply -- it is a
ceiling, not a fair opponent. The gap between it and a model is the interesting
number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from typing import Any, Dict, List

import numpy as np

from assaygym.llm_harness import PRICE_PER_MTOK, run_episode

TIERS = ["clean", "standard", "hard"]

# The scripted ladder from run_baselines.py, n=200. Printed alongside so a model
# number is never read without its ceiling.
SCRIPTED = {
    "clean": {"prior_parrot": 0.020, "naive_screen": 0.280, "competent_doe": 1.000},
    "standard": {"prior_parrot": 0.010, "naive_screen": 0.080, "competent_doe": 0.640},
    "hard": {"prior_parrot": 0.000, "naive_screen": 0.020, "competent_doe": 0.175},
}


def _mean_se(values: List[float]) -> tuple:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def smoke(model: str, tier: str, seed: int, verbose: bool) -> int:
    """One episode, printed in detail. Confirms the schemas and the layout format."""
    print(f"=== smoke test: {model} on {tier}, seed {seed} ===\n")
    events: List[str] = []

    def on_event(kind: str, payload: Dict[str, Any]) -> None:
        if kind == "tool":
            flag = "ERROR" if payload["is_error"] else "ok"
            print(f"  turn {payload['turn']:>2}  {payload['name']:<16} {flag}")
        elif kind in ("nudge", "forced_submission", "refusal"):
            print(f"  turn {payload.get('turn')}  <{kind}>")
        events.append(kind)

    t0 = time.time()
    try:
        res = run_episode(seed, tier, model=model, on_event=on_event)
    except Exception as exc:                                   # noqa: BLE001
        message = str(exc)
        print(f"\n  REQUEST FAILED: {exc.__class__.__name__}\n  {message}\n")
        # The one schema shape validated offline against ToolParam but not
        # against the API's own validator.
        if "input_schema" in message or "tools" in message or "schema" in message:
            print("  This looks like a tool-schema rejection. The most likely\n"
                  "  culprit is the union type on submit.log_ec50:\n"
                  "      \"type\": [\"number\", \"null\"]\n"
                  "  If so, the fix is to make it a plain \"number\" and drop it\n"
                  "  from `required`, so omitting it means 'not measured'.\n"
                  "  assaygym/env.py, TOOL_SPEC, submit.input_schema.")
        elif "credit" in message.lower() or "billing" in message.lower():
            print("  Billing/credit issue, not a code issue.")
        return 1
    elapsed = time.time() - t0

    print(f"\n  turns            {res.turns} / 24")
    print(f"  submitted        {res.submitted} (forced={res.forced_submission})")
    print(f"  stop_reason      {res.stop_reason}")
    print(f"  tool calls       {len(res.tool_calls)} ({res.n_tool_errors} errors)")
    print(f"  plates run       {res.score['diagnostics']['n_plates']}")
    print(f"  usd spent (env)  ${res.score['diagnostics']['usd_spent']:,.0f}")
    print(f"  strict_pass      {res.score['strict_pass']}")
    print(f"  endpoint         {res.score['endpoint']:.4f}")
    print(f"  shaped           {res.score['shaped']:.4f}")
    print(f"  decoy_called     {res.score['diagnostics']['decoy_called']}")
    print(f"  usage            {res.usage}")
    print(f"  API cost         ${res.cost_usd:.4f}   wall clock {elapsed:.1f}s")

    if res.n_tool_errors:
        print("\n  --- tool errors (these are what a schema problem looks like) ---")
        for call in res.tool_calls:
            if call["is_error"]:
                print(f"    turn {call['turn']} {call['name']}: {call['error']}")

    if verbose:
        print("\n  --- submission ---")
        print(f"    {res.env.submission}")
        print(f"    truth: hits={res.env.world.true_hits} "
              f"signs={res.env.world.true_signs} "
              f"log_ec50={res.env.world.true_log_ec50:.3f}")

    layout_ok = res.score["diagnostics"]["n_plates"] > 0
    print(f"\n  VERDICT: schemas accepted={not res.n_tool_errors or layout_ok}, "
          f"model emitted a usable plate layout={layout_ok}")
    return 0 if layout_ok and res.submitted else 1


def sweep(models: List[str], tiers: List[str], n: int, seed0: int,
          max_usd: float, out_path: str | None) -> int:
    rows: Dict[str, Dict[str, Any]] = {}
    spent = 0.0
    stopped = False
    episodes: List[Dict[str, Any]] = []

    for model in models:
        rows[model] = {}
        for tier in tiers:
            cells: List[float] = []
            endpoints: List[float] = []
            shapes: List[float] = []
            decoys: List[float] = []
            turns: List[int] = []
            forced = 0
            errors = 0
            for i in range(n):
                if spent >= max_usd:
                    print(f"\n!! spend cap ${max_usd:.2f} reached "
                          f"(${spent:.2f}); stopping.", file=sys.stderr)
                    stopped = True
                    break
                seed = seed0 + i
                try:
                    res = run_episode(seed, tier, model=model)
                except Exception:                       # noqa: BLE001
                    errors += 1
                    print(f"  !! {model}/{tier} seed {seed} raised:",
                          file=sys.stderr)
                    traceback.print_exc()
                    continue
                spent += res.cost_usd
                cells.append(res.score["strict_pass"])
                endpoints.append(res.score["endpoint"])
                shapes.append(res.score["shaped"])
                decoys.append(float(res.score["diagnostics"]["decoy_called"] > 0))
                turns.append(res.turns)
                forced += int(res.forced_submission)
                episodes.append({**res.to_dict(), "model": model})
                print(f"  {model} {tier} seed {seed}: "
                      f"strict={res.score['strict_pass']:.0f} "
                      f"endpoint={res.score['endpoint']:.3f} "
                      f"turns={res.turns} ${res.cost_usd:.3f} "
                      f"(total ${spent:.2f})", file=sys.stderr)
            m, se = _mean_se(cells)
            rows[model][tier] = {
                "n": len(cells), "strict_pass": m, "strict_pass_se": se,
                "endpoint": _mean_se(endpoints)[0], "shaped": _mean_se(shapes)[0],
                "decoy_rate": _mean_se(decoys)[0],
                "mean_turns": _mean_se([float(t) for t in turns])[0],
                "forced_submissions": forced, "errors": errors,
            }
            if stopped:
                break
        if stopped:
            break

    print(f"\n=== strict_pass, n = {n} per cell (mean +/- SE) ===")
    print("NOTE: at n=10 the standard error is ~0.15. These anchor the ladder; "
          "they do not resolve small differences.\n")
    width = 24
    print(f"{'tier':<10}" + "".join(f"{m:>{width}}" for m in models)
          + f"{'competent_doe (n=200)':>26}")
    for tier in tiers:
        cells = []
        for model in models:
            r = rows.get(model, {}).get(tier)
            if not r or r["n"] == 0:
                cells.append("-")
            else:
                se = "" if np.isnan(r["strict_pass_se"]) else f" +/- {r['strict_pass_se']:.3f}"
                cells.append(f"{r['strict_pass']:.3f}{se} (n={r['n']})")
        print(f"{tier:<10}" + "".join(f"{c:>{width}}" for c in cells)
              + f"{SCRIPTED[tier]['competent_doe']:>26.3f}")

    print(f"\n=== endpoint / shaped / decoy rate / mean turns ===")
    for model in models:
        for tier in tiers:
            r = rows.get(model, {}).get(tier)
            if not r or r["n"] == 0:
                continue
            print(f"  {model:<18} {tier:<9} endpoint {r['endpoint']:.3f}  "
                  f"shaped {r['shaped']:.3f}  decoy_rate {r['decoy_rate']:.3f}  "
                  f"turns {r['mean_turns']:.1f}  forced {r['forced_submissions']}"
                  f"{'  errors ' + str(r['errors']) if r['errors'] else ''}")

    print(f"\ntotal API spend: ${spent:.2f} of ${max_usd:.2f} cap"
          + ("  (STOPPED EARLY)" if stopped else ""))

    if out_path:
        with open(out_path, "w") as fh:
            json.dump({"models": models, "tiers": tiers, "n": n, "seed0": seed0,
                       "usd_spent": spent, "stopped_early": stopped,
                       "rows": rows, "scripted_reference": SCRIPTED,
                       "episodes": episodes}, fh, indent=2)
        print(f"wrote {out_path}")
    return 0


def _have_credentials() -> bool:
    """A key, a token, or an `ant auth login` profile on disk."""
    import os
    import pathlib as _pathlib

    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (_pathlib.Path.home() / ".config" / "anthropic").exists()


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="run a single episode in detail and exit")
    ap.add_argument("--models", nargs="*", default=["claude-sonnet-5"],
                    help=f"priced: {sorted(PRICE_PER_MTOK)}")
    ap.add_argument("--model", default="claude-sonnet-5", help="smoke-test model")
    ap.add_argument("--tiers", nargs="*", default=TIERS, choices=TIERS)
    ap.add_argument("--tier", default="clean", help="smoke-test tier")
    ap.add_argument("--n", type=int, default=10, help="episodes per cell")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="smoke-test seed")
    ap.add_argument("--max-usd", type=float, default=10.0,
                    help="hard spend cap; the sweep stops when reached")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not _have_credentials():
        print(
            "No Anthropic credentials found.\n\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...      # then re-run\n"
            "  # or, if you use the CLI:  ant auth login\n\n"
            "The rest of the project needs neither a key nor the SDK; only this\n"
            "script and assaygym.llm_harness do.",
            file=sys.stderr,
        )
        return 2

    if args.smoke:
        return smoke(args.model, args.tier, args.seed, args.verbose)
    return sweep(args.models, args.tiers, args.n, args.seed0,
                 args.max_usd, args.json)


if __name__ == "__main__":
    sys.exit(main())
