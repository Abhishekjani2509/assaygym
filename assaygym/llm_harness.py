"""AssayGym Phase 6 — the Anthropic tool-use harness.

Drives :class:`~assaygym.env.AssayGym` over the Claude Messages API: send the
briefing, execute whatever ``tool_use`` blocks come back, return ``tool_result``
blocks, repeat until the model submits or ``max_turns`` is reached.

**Deliberately thin, and that is a design requirement rather than laziness.**
Harness choice moves bio-agent pass rates by several points on identical tasks,
which makes it a variable in the result. A variable has to be reported, and a
variable can only be reported if it is separable — so everything here is
swappable without touching a line of scoring:

* the model, ``max_turns`` and the system prompt are parameters
* the tool list comes from :data:`assaygym.env.TOOL_SPEC`, the same list that
  drives the verifiers adapter
* the API client is **injected**, so this module has no network dependency and
  no import dependency on ``anthropic``; the SDK is imported lazily inside
  :func:`default_client` and only when nobody passed a client
* scoring is :func:`assaygym.rewards.score`, called on the finished env and
  given nothing but the env

Anything the harness does that could move a score is a named argument with a
default recorded in the returned transcript, so two runs can be compared by
diffing their ``harness`` blocks.

A model that never submits is forced to an empty submission rather than dropped,
so a failed episode still lands on the ledger with a real zero instead of
silently shrinking the denominator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from assaygym.env import TOOL_SPEC, AssayGym
from assaygym.rewards import score

__all__ = [
    "DEFAULT_MODEL",
    "MAX_TURNS",
    "MAX_TOKENS",
    "SYSTEM_PROMPT",
    "HarnessResult",
    "default_client",
    "format_briefing",
    "run_episode",
]


# Claude Opus 5. Overridable, and recorded in every transcript, because the
# model is part of the result and not part of the environment.
DEFAULT_MODEL = "claude-opus-5"
MAX_TURNS = 24
MAX_TOKENS = 16_000

SYSTEM_PROMPT = (
    "You are an experimental scientist running a gene-knockdown screening "
    "campaign. You have a fixed budget and one submission. Work from the data "
    "you measure. Use the tools; do not describe what you would do instead of "
    "doing it. When you have an answer, call submit exactly once."
)

# What the model is told when it ends a turn without calling any tool. Kept to
# one sentence: a harness that argues with the model is a harness that is
# steering the result.
NUDGE = (
    "You did not call a tool. Use one of the available tools, or call submit "
    "with your final answer."
)


@dataclass
class HarnessResult:
    """One episode: the score, plus everything needed to reproduce it."""

    seed: int
    tier: str
    score: Dict[str, Any]
    turns: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    submitted: bool = False
    forced_submission: bool = False
    stop_reason: Optional[str] = None
    harness: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    # The finished environment. Carried so a caller can inspect what actually
    # happened -- plates, exclusions, the recorded submission -- rather than
    # inferring it from the score. score() treats a missing submission as an
    # empty one, so the score alone cannot tell "submitted nothing" from
    # "never submitted"; this can.
    env: Optional[AssayGym] = None

    @property
    def n_tool_errors(self) -> int:
        return sum(1 for c in self.tool_calls if c["is_error"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed, "tier": self.tier, "score": self.score,
            "turns": self.turns, "submitted": self.submitted,
            "forced_submission": self.forced_submission,
            "stop_reason": self.stop_reason, "harness": dict(self.harness),
            "n_tool_calls": len(self.tool_calls),
            "n_tool_errors": self.n_tool_errors,
        }


def default_client() -> Any:
    """Construct an ``anthropic.Anthropic()``. Imported lazily, on purpose.

    Keeping the import here means the module is importable, testable and
    scoreable with no SDK installed and no network — which is what lets the
    stub-client tests exist at all.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise ImportError(
            "the Anthropic SDK is not installed; `pip install anthropic`, or "
            "pass your own client to run_episode(client=...)"
        ) from exc
    return anthropic.Anthropic()


def format_briefing(briefing: Dict[str, Any]) -> str:
    """The opening user message. Pure serialisation — nothing is added or hidden.

    The briefing dict is handed over verbatim as JSON, including the literature
    prior and its caveat. A harness that summarised or reworded the briefing
    would be quietly changing the task.
    """
    return (
        "Here is your briefing for this screening campaign.\n\n"
        f"{json.dumps(briefing, indent=2)}\n\n"
        "Plan your plates, run them, and call submit once with your final "
        "answer."
    )


def _blocks(response: Any) -> List[Any]:
    return list(getattr(response, "content", None) or [])


def _block_type(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("type", ""))
    return str(getattr(block, "type", ""))


def _block_attr(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _serialise_content(response: Any) -> Any:
    """Assistant content, echoed back on the next request.

    Blocks are passed through unchanged when the SDK gives real block objects;
    the API accepts them directly. Dicts (what a stub returns) pass through too.
    """
    return _blocks(response)


def run_episode(
    seed: int,
    tier: str = "standard",
    client: Any = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = MAX_TURNS,
    max_tokens: int = MAX_TOKENS,
    system: str = SYSTEM_PROMPT,
    thinking: Optional[Dict[str, Any]] = None,
    extra_create_kwargs: Optional[Dict[str, Any]] = None,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> HarnessResult:
    """Run one episode end to end and score it.

    ``client`` may be anything exposing ``client.messages.create(**kwargs)`` and
    returning an object with ``.content`` and ``.stop_reason``. That is the whole
    contract, which is what makes the harness swappable and testable offline.
    """
    if client is None:
        client = default_client()
    if thinking is None:
        thinking = {"type": "adaptive"}

    env = AssayGym(seed, tier)
    briefing = env.reset()

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": format_briefing(briefing)}
    ]
    tool_calls: List[Dict[str, Any]] = []
    turns = 0
    stop_reason: Optional[str] = None

    def emit(kind: str, payload: Dict[str, Any]) -> None:
        if on_event is not None:
            on_event(kind, payload)

    while turns < max_turns and not env.done:
        turns += 1
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "tools": TOOL_SPEC,
            # A shallow copy per request. The harness mutates `messages` as the
            # episode proceeds, and a client that keeps the object it was
            # handed -- a recorder, a replay transport, a cache -- would
            # otherwise see every past request aliased to the final state.
            "messages": list(messages),
            "thinking": thinking,
        }
        create_kwargs.update(extra_create_kwargs or {})
        response = client.messages.create(**create_kwargs)
        stop_reason = getattr(response, "stop_reason", None)
        emit("response", {"turn": turns, "stop_reason": stop_reason})

        content = _serialise_content(response)
        messages.append({"role": "assistant", "content": content})

        # A server-side tool hit its iteration limit. Re-send to continue; the
        # paused assistant turn is already on the history.
        if stop_reason == "pause_turn":
            continue

        # A safety decline ends the episode. There is nothing to execute and
        # nothing to be gained by arguing, so fall through to the forced
        # submission below and let it score as the zero it is.
        if stop_reason == "refusal":
            emit("refusal", {"turn": turns,
                             "stop_details": getattr(response, "stop_details", None)})
            break

        uses = [b for b in content if _block_type(b) == "tool_use"]
        if not uses:
            # The model talked instead of acting. One short reminder, which
            # still consumes a turn, so this cannot loop forever.
            messages.append({"role": "user", "content": NUDGE})
            emit("nudge", {"turn": turns})
            continue

        # Every tool_result for a turn goes back in ONE user message. Splitting
        # them trains the model out of parallel tool calls.
        results: List[Dict[str, Any]] = []
        for use in uses:
            name = str(_block_attr(use, "name", ""))
            raw_input = _block_attr(use, "input", {}) or {}
            arguments = raw_input if isinstance(raw_input, dict) else {}
            result = env.call(name, arguments)
            is_error = "error" in result
            tool_calls.append({"turn": turns, "name": name,
                               "is_error": is_error,
                               "error": result.get("error")})
            emit("tool", {"turn": turns, "name": name, "is_error": is_error})
            results.append({
                "type": "tool_result",
                "tool_use_id": _block_attr(use, "id", ""),
                "content": json.dumps(result, default=str),
                # A refused call is reported as an error rather than dropped, so
                # the model can see the refusal and adapt to it.
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": results})

    submitted = env.done
    forced = False
    if not env.done:
        # Out of turns, refused, or the model simply never submitted. Force an
        # empty submission so the episode still scores instead of vanishing
        # from the denominator.
        env.submit([], {}, None)
        forced = True
        emit("forced_submission", {"turn": turns})

    return HarnessResult(
        env=env,
        seed=seed, tier=tier, score=score(env), turns=turns,
        tool_calls=tool_calls, submitted=submitted, forced_submission=forced,
        stop_reason=stop_reason,
        harness={
            "model": model, "max_turns": max_turns, "max_tokens": max_tokens,
            "thinking": dict(thinking), "system_prompt_chars": len(system),
            "n_tools": len(TOOL_SPEC), "module": __name__,
        },
        messages=messages,
    )
