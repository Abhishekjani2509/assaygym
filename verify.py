#!/usr/bin/env python3
"""Phase 5 gate: three checks that must pass before anything else is believed.

    ./.venv/bin/python verify.py            # all three, n = 200
    ./.venv/bin/python verify.py --n 50     # faster, wider error bars

1. **Determinism.** The same seed and policy twice gives an identical score
   dict. Without this every number below is a coin flip and the ledger means
   nothing.

2. **The degenerate exploit.** Submit *every* gene as a hit. This is the classic
   way to farm recall, and it has to lose: `strict_pass` exactly 0.000, and an
   `endpoint` no better than a policy that actually did the work.

3. **The ledger.** Mean +/- standard error over N seeds per cell, checked
   against the BUILD_SPEC acceptance table with a +/-0.05 tolerance, plus the
   two structural properties that matter more than any individual cell:
   monotonicity in every tier, and `prior_parrot` staying near zero.

Exit code is 0 only if all three pass.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

import numpy as np

from assaygym.policies import (
    POLICIES,
    call_everything_policy,
    run_episode,
    run_policy,
)

TIERS = ["clean", "standard", "hard"]
ORDER = ["random", "prior_parrot", "naive_screen", "competent_doe"]

# BUILD_SPEC "Phase 5 acceptance -- the numbers to hit", strict_pass.
TARGETS: Dict[str, Dict[str, float]] = {
    "clean":    {"random": 0.000, "prior_parrot": 0.040,
                 "naive_screen": 0.275, "competent_doe": 1.000},
    "standard": {"random": 0.000, "prior_parrot": 0.015,
                 "naive_screen": 0.075, "competent_doe": 0.620},
    "hard":     {"random": 0.000, "prior_parrot": 0.000,
                 "naive_screen": 0.005, "competent_doe": 0.165},
}
TOLERANCE = 0.05
PRIOR_PARROT_CEILING = 0.05


def _ok(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def check_determinism(n: int = 12) -> bool:
    """Same seed, same policy, twice -- byte-identical score dicts."""
    print("\n[1] Determinism")
    failures = 0
    checked = 0
    for tier in TIERS:
        for name, policy in POLICIES.items():
            for seed in range(n):
                a = run_episode(policy, seed, tier)
                b = run_episode(policy, seed, tier)
                checked += 1
                if a != b:
                    failures += 1
                    print(f"    MISMATCH {tier}/{name} seed {seed}")
    print(f"    {checked} episodes replayed, {failures} mismatches")

    # And across policies: the same seed must present the same hidden world.
    from assaygym.env import AssayGym
    worlds = []
    for _ in range(2):
        env = AssayGym(41, "hard")
        env.reset()
        worlds.append((tuple(env.world.true_hits), env.world.true_log_ec50))
    same_world = worlds[0] == worlds[1]
    print(f"    same seed -> same world: {same_world}")
    print(f"    {_ok(failures == 0 and same_world)}")
    return failures == 0 and same_world


def check_degenerate_exploit(n: int) -> bool:
    """Calling every gene a hit must not win."""
    print("\n[2] Degenerate exploit: call every gene a hit")
    ok = True
    for tier in TIERS:
        res = run_policy(call_everything_policy, n=n, tier=tier)
        sp = np.array([r["strict_pass"] for r in res])
        ep = np.array([r["endpoint"] for r in res])
        recall = np.mean([r["diagnostics"]["recall"] for r in res])
        prec = np.mean([r["diagnostics"]["precision"] for r in res])
        doe = np.array([r["endpoint"] for r in run_policy(
            POLICIES["competent_doe"], n=n, tier=tier)])
        beats = ep.mean() >= doe.mean()
        print(f"    {tier:<9} strict_pass {sp.mean():.3f}  endpoint "
              f"{ep.mean():.3f} +/- {ep.std(ddof=1) / np.sqrt(n):.3f}  "
              f"(recall {recall:.3f}, precision {prec:.3f}; "
              f"competent_doe endpoint {doe.mean():.3f})")
        if sp.mean() != 0.0 or beats:
            ok = False
    print(f"    {_ok(ok)}")
    return ok


def check_ledger(n: int) -> bool:
    """The acceptance table, plus monotonicity and the prior-parrot ceiling."""
    print(f"\n[3] The ledger: strict_pass, n = {n} per cell")
    header = f"    {'tier':<10}" + "".join(f"{k:>24}" for k in ORDER)
    print(header)
    ok = True
    for tier in TIERS:
        means, cells = [], []
        for name in ORDER:
            res = run_policy(POLICIES[name], n=n, tier=tier)
            sp = np.array([r["strict_pass"] for r in res])
            m, se = float(sp.mean()), float(sp.std(ddof=1) / np.sqrt(n))
            means.append(m)
            target = TARGETS[tier][name]
            within = abs(m - target) <= TOLERANCE
            ok = ok and within
            cells.append(f"{m:.3f}+/-{se:.3f}{'' if within else ' !'}")
        print(f"    {tier:<10}" + "".join(f"{c:>24}" for c in cells))
        print(f"    {'  target':<10}" + "".join(
            f"{TARGETS[tier][k]:>24.3f}" for k in ORDER))

        monotone = all(means[i] <= means[i + 1] + 1e-12 for i in range(len(means) - 1))
        parrot_ok = means[ORDER.index("prior_parrot")] <= PRIOR_PARROT_CEILING
        print(f"    {'':<10}monotone: {monotone}   prior_parrot <= "
              f"{PRIOR_PARROT_CEILING}: {parrot_ok}")
        ok = ok and monotone and parrot_ok
    print(f"    {_ok(ok)}")
    return ok


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200, help="episodes per cell")
    args = ap.parse_args(argv)

    results = [
        ("determinism", check_determinism()),
        ("degenerate exploit", check_degenerate_exploit(args.n)),
        ("ledger", check_ledger(args.n)),
    ]
    print("\n" + "=" * 60)
    for name, passed in results:
        print(f"  {_ok(passed):<5} {name}")
    all_ok = all(p for _, p in results)
    print(f"\n{'ALL CHECKS PASS' if all_ok else 'FAILURES ABOVE'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
