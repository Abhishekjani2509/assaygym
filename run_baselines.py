#!/usr/bin/env python3
"""Run every scripted policy over N seeded episodes per tier and print the ledger.

    ./.venv/bin/python run_baselines.py --n 200 --json results.json
    ./.venv/bin/python run_baselines.py --n 200 --ablate

The ledger is the artifact. `strict_pass` is the headline column; `endpoint` and
`shaped` are reported alongside because a reader should be able to see how much
partial credit the sparse reward is handing out.

`--ablate` runs the competent_doe reference policy with one design step disabled
at a time. It answers a harder question than the ladder does: not "is the score
monotone in competence" but "does each artifact in the environment actually earn
its place". An ablation that costs nothing means the artifact it defends against
is too weak to matter, or the defence was never being used.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict, List

import numpy as np

from assaygym.policies import ABLATIONS, POLICIES, run_policy

TIERS = ["clean", "standard", "hard"]
METRICS = ["strict_pass", "endpoint", "shaped"]


def _stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean and standard error for every reported quantity."""
    out: Dict[str, Any] = {"n": len(results)}
    for key in METRICS:
        v = np.array([r[key] for r in results], dtype=float)
        out[key] = float(v.mean())
        out[f"{key}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    for key in ("decoy_called", "omitted_recovered", "n_plates", "usd_spent",
                "precision", "recall", "prior_trap"):
        v = np.array([r["diagnostics"][key] for r in results], dtype=float)
        out[key] = float(v.mean())
    # The direct measurement of prior-dependence: how often ANY decoy was called.
    dc = np.array([r["diagnostics"]["decoy_called"] for r in results], dtype=float)
    out["decoy_rate"] = float((dc > 0).mean())
    return out


def _table(rows: Dict[str, Dict[str, Dict[str, Any]]], names: List[str],
           metric: str = "strict_pass") -> str:
    width = max(len(n) for n in names) + 2
    head = f"{'tier':<10}" + "".join(f"{n:>{width + 8}}" for n in names)
    lines = [head, "-" * len(head)]
    for tier in TIERS:
        if tier not in rows:
            continue
        cells = []
        for name in names:
            s = rows[tier].get(name)
            cells.append("-" if s is None
                         else f"{s[metric]:.3f} +/- {s[metric + '_se']:.3f}")
        lines.append(f"{tier:<10}" + "".join(f"{c:>{width + 8}}" for c in cells))
    return "\n".join(lines)


def run_suite(policies: Dict[str, Callable[..., None]], n: int,
              tiers: List[str], seed0: int = 0) -> Dict[str, Dict[str, Dict[str, Any]]]:
    rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for tier in tiers:
        rows[tier] = {}
        for name, policy in policies.items():
            rows[tier][name] = _stats(run_policy(policy, n=n, tier=tier, seed0=seed0))
            print(f"  ran {tier}/{name}", file=sys.stderr)
    return rows


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200, help="episodes per cell")
    ap.add_argument("--tiers", nargs="*", default=TIERS, choices=TIERS)
    ap.add_argument("--seed0", type=int, default=0, help="first seed")
    ap.add_argument("--json", type=str, default=None, help="write results here")
    ap.add_argument("--ablate", action="store_true",
                    help="ablate competent_doe instead of running the baselines")
    args = ap.parse_args(argv)

    policies = ABLATIONS if args.ablate else POLICIES
    rows = run_suite(policies, args.n, args.tiers, args.seed0)
    names = list(policies)

    label = "competent_doe ablations" if args.ablate else "baseline ladder"
    print(f"\n=== {label}: strict_pass, n = {args.n} per cell "
          f"(mean +/- standard error) ===")
    print(_table(rows, names))
    for metric in ("endpoint", "shaped"):
        print(f"\n=== {label}: {metric} ===")
        print(_table(rows, names, metric))

    if args.ablate:
        print("\n=== cost of each ablation (strict_pass, full minus ablated) ===")
        width = max(len(n) for n in names) + 2
        for tier in args.tiers:
            full = rows[tier]["full"]["strict_pass"]
            print(f"  {tier}  (full = {full:.3f})")
            for name in names:
                if name == "full":
                    continue
                s = rows[tier][name]
                delta = full - s["strict_pass"]
                se = float(np.hypot(rows[tier]["full"]["strict_pass_se"],
                                    s["strict_pass_se"]))
                flag = "  <-- costs nothing" if delta <= se else ""
                print(f"    {name:<{width}} {s['strict_pass']:.3f}  "
                      f"delta {delta:+.3f} +/- {se:.3f}{flag}")
    else:
        print("\n=== prior-dependence: fraction of episodes calling >=1 decoy ===")
        width = max(len(n) for n in names) + 2
        for tier in args.tiers:
            cells = "  ".join(
                f"{n}={rows[tier][n]['decoy_rate']:.3f}" for n in names)
            print(f"  {tier:<10} {cells}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"n": args.n, "seed0": args.seed0, "ablation": args.ablate,
                       "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
