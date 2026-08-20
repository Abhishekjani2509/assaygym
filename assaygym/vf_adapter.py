"""AssayGym Phase 6 — the verifiers-spec adapter.

Environments and evals are the same object: a dataset, a harness and a scoring
rule. Building to the verifiers spec means this artifact is both at once, is
pip-installable and versioned, and drops into a trainer without anything being
rewritten.

``verifiers`` is imported inside a ``try/except ImportError`` and the module
falls back to returning a plain dict, so it stays importable, testable and
usable with no training stack present. That is not a nicety: the fallback dict
carries the dataset, the tool spec, the rubric description and live
``make_env`` / ``score_episode`` callables, so a trainer that does not use
verifiers still has everything it needs.

**One dataset row per seed, and the row never contains the answer.** The hidden
world is a deterministic function of the seed, so a grader re-derives ground
truth from ``info["seed"]`` at scoring time. Putting the answer in the row would
put it in the prompt's blast radius, and the whole design rests on the agent
never seeing it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from assaygym.env import TOOL_SPEC, AssayGym
from assaygym.rewards import ENDPOINT_WEIGHTS, SHAPED_WEIGHTS, score
from assaygym.world import TIERS

__all__ = [
    "REWARD_SPEC",
    "build_dataset",
    "make_env",
    "score_episode",
    "rollout",
    "load_environment",
]


# What a trainer should optimise, and what it should merely watch. strict_pass
# is the headline; the diagnostics are reported and never summed into reward.
REWARD_SPEC: Dict[str, Any] = {
    "primary": "strict_pass",
    "metrics": ["strict_pass", "endpoint", "shaped"],
    "shaping_metric": "shaped",
    "endpoint_weights": dict(ENDPOINT_WEIGHTS),
    "shaped_weights": dict(SHAPED_WEIGHTS),
    "diagnostics": [
        "precision", "recall", "decoy_called", "omitted_recovered",
        "prior_trap", "n_plates", "n_excluded", "usd_spent",
    ],
    "notes": (
        "strict_pass is the headline number. endpoint gives partial credit "
        "generously enough that a policy which parrots the literature prior "
        "and runs zero plates scores 0.66 on clean. Diagnostics are computed "
        "and never summed into reward; decoy_called is the direct measurement "
        "of prior-dependence."
    ),
}


def make_env(seed: int, tier: str = "standard") -> AssayGym:
    """A reset environment for one seed. The dataset row's other half."""
    env = AssayGym(seed, tier)
    env.reset()
    return env


def score_episode(env: AssayGym) -> Dict[str, Any]:
    """Score a finished episode. Thin passthrough, so scoring stays in one place."""
    return score(env)


def build_dataset(
    tier: str = "standard", n_episodes: int = 100, seed0: int = 0
) -> List[Dict[str, Any]]:
    """One row per seed. Rows carry the briefing and the seed, never the truth."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")
    if n_episodes < 0:
        raise ValueError("n_episodes must be non-negative")

    rows: List[Dict[str, Any]] = []
    for i in range(n_episodes):
        seed = seed0 + i
        env = AssayGym(seed, tier)
        briefing = env.reset()
        rows.append({
            "question": briefing["objective"],
            # Deliberately empty. Ground truth is re-derived from the seed at
            # scoring time; it must never travel with the prompt.
            "answer": "",
            "task": f"assaygym-{tier}",
            "info": {
                "seed": seed,
                "tier": tier,
                "briefing": briefing,
                "budget_usd": briefing["budget"]["usd"],
                "budget_days": briefing["budget"]["days"],
                "hit_threshold": briefing["hit_threshold"],
                "n_loci": len(briefing["loci"]),
            },
        })
    return rows


def rollout(
    seed: int, tier: str = "standard", client: Any = None, **harness_kwargs: Any
) -> Dict[str, Any]:
    """Run and score one episode through the Anthropic harness.

    Imported lazily so this module does not drag in the harness (and its lazy
    SDK import) for callers who only want the dataset.
    """
    from assaygym.llm_harness import run_episode

    result = run_episode(seed, tier, client=client, **harness_kwargs)
    return {**result.to_dict(), "result": result}


def _fallback(
    tier: str, n_episodes: int, seed0: int, reason: str
) -> Dict[str, Any]:
    """The no-verifiers return: a plain dict a trainer can still use directly."""
    return {
        "name": f"assaygym-{tier}",
        "tier": tier,
        "n_episodes": n_episodes,
        "seed0": seed0,
        "dataset": build_dataset(tier, n_episodes, seed0),
        "tools": TOOL_SPEC,
        "rubric": REWARD_SPEC,
        "make_env": make_env,
        "score_episode": score_episode,
        "rollout": rollout,
        "verifiers_available": False,
        "fallback_reason": reason,
    }


def load_environment(
    tier: str = "standard",
    n_episodes: int = 100,
    seed0: int = 0,
    **kwargs: Any,
) -> Any:
    """The verifiers entry point.

    Returns a ``verifiers`` environment when the training stack is installed,
    and otherwise the plain dict from :func:`_fallback`. The fallback path is
    the one covered by tests, because it is the one that can be tested without
    the training stack; the wrap path is guarded so that a version mismatch
    degrades to the dict **visibly** — ``fallback_reason`` says what happened —
    rather than raising or silently returning something different.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")

    try:
        import verifiers as vf
    except ImportError:
        return _fallback(tier, n_episodes, seed0, "verifiers is not installed")

    dataset = build_dataset(tier, n_episodes, seed0)
    tool_env = getattr(vf, "ToolEnv", None)
    if tool_env is None:
        return _fallback(
            tier, n_episodes, seed0,
            "the installed verifiers has no ToolEnv; using the plain dict",
        )
    try:
        return tool_env(
            dataset=dataset,
            tools=TOOL_SPEC,
            rubric=REWARD_SPEC,
            **kwargs,
        )
    except (AttributeError, TypeError) as exc:
        return _fallback(
            tier, n_episodes, seed0,
            f"verifiers.ToolEnv rejected the arguments ({exc.__class__.__name__}: "
            f"{exc}); using the plain dict",
        )
