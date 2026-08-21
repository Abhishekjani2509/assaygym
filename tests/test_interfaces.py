"""Phase 6 acceptance checks for assaygym/llm_harness.py and vf_adapter.py.

Run: ./.venv/bin/python -m pytest tests/ -q -s   (see CONTRIBUTING.md)

Every test here runs with **no network and no `anthropic` package installed**.
The harness takes an injected client whose entire contract is
``client.messages.create(**kwargs) -> object with .content and .stop_reason``,
so the four model behaviours that matter can be stubbed exactly:

  1. submits immediately
  2. never submits (burns every turn)
  3. calls a tool with malformed arguments
  4. emits no tool call at all

The verifiers adapter is tested on both paths: the real ImportError fallback,
and a fake `verifiers` module injected into `sys.modules`.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assaygym.env import TOOL_SPEC  # noqa: E402
from assaygym.llm_harness import (  # noqa: E402
    DEFAULT_MODEL,
    MAX_TURNS,
    PRICE_PER_MTOK,
    HarnessResult,
    episode_cost_usd,
    format_briefing,
    run_episode,
)
from assaygym.vf_adapter import (  # noqa: E402
    REWARD_SPEC,
    build_dataset,
    load_environment,
    make_env,
    score_episode,
)

INTERIOR = [f"{r}{c}" for r in "BCDEFG" for c in range(2, 12)]


# --------------------------------------------------------------------------
# Stub client plumbing
# --------------------------------------------------------------------------


class Block(dict):
    """A content block. dict-shaped, and also attribute-accessible.

    Real SDK blocks are objects with `.type` / `.id` / `.name` / `.input`; a
    stub returning plain dicts proves the harness reads both, which is what
    lets a different SDK or a replay transport be dropped in.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class Response:
    def __init__(self, content, stop_reason, usage=None):
        self.content = content
        self.stop_reason = stop_reason
        if usage is not None:
            self.usage = usage


class StubClient:
    """Replays a scripted list of responses and records every request."""

    def __init__(self, script, loop_last=False):
        self._script = list(script)
        self._loop_last = loop_last
        self.requests = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self._script:
            return self._script.pop(0)
        if self._loop_last:
            return self._last
        raise AssertionError("stub client ran out of scripted responses")

    def _create_wrapper(self, **kwargs):  # pragma: no cover - alias
        return self._create(**kwargs)


class RepeatClient:
    """Returns the same response forever. For the never-submits case."""

    def __init__(self, response):
        self.response = response
        self.requests = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def _tool_use(tool_id, name, arguments):
    return Block(type="tool_use", id=tool_id, name=name, input=arguments)


def _text(text):
    return Block(type="text", text=text)


def _plate_layout(n=20, condition="NTC"):
    return {w: condition for w in INTERIOR[:n]}


# --------------------------------------------------------------------------
# 1. A model that submits immediately
# --------------------------------------------------------------------------


def test_model_that_submits_immediately():
    client = StubClient([
        Response([_text("Submitting."),
                  _tool_use("t1", "submit",
                            {"hits": ["SYN01"], "signs": {"SYN01": -1},
                             "log_ec50": 2.0})],
                 "tool_use"),
    ])
    res = run_episode(7, "clean", client=client)

    print(f"\n[measured] submit-immediately: turns={res.turns}, "
          f"submitted={res.submitted}, forced={res.forced_submission}, "
          f"strict_pass={res.score['strict_pass']}, "
          f"tool_calls={len(res.tool_calls)}, errors={res.n_tool_errors}")
    assert res.turns == 1
    assert res.submitted is True and res.forced_submission is False
    assert res.n_tool_errors == 0
    assert res.score["diagnostics"]["n_plates"] == 0
    assert 0.0 <= res.score["strict_pass"] <= 1.0

    # The request the harness actually sent.
    req = client.requests[0]
    assert req["model"] == DEFAULT_MODEL
    assert req["tools"] is TOOL_SPEC
    assert req["thinking"] == {"type": "adaptive"}
    assert req["messages"][0]["role"] == "user"
    assert "previously_reported_hits" in req["messages"][0]["content"]


def test_a_real_campaign_scores_and_reports_the_transcript():
    """Two plates, a qc call, then submit. The ordinary path."""
    layout = _plate_layout(24)
    client = StubClient([
        Response([_tool_use("t1", "design_and_run",
                            {"layout": layout, "lot": "LOT-A"})], "tool_use"),
        Response([_tool_use("t2", "design_and_run",
                            {"layout": layout, "lot": "LOT-B"}),
                  _tool_use("t3", "qc", {"plate_id": "P1"})], "tool_use"),
        Response([_tool_use("t4", "submit",
                            {"hits": [], "signs": {}, "log_ec50": None})],
                 "tool_use"),
    ])
    res = run_episode(3, "standard", client=client)

    print(f"[measured] campaign: turns={res.turns}, "
          f"tool_calls={[c['name'] for c in res.tool_calls]}, "
          f"plates={res.score['diagnostics']['n_plates']}, "
          f"usd_spent=${res.score['diagnostics']['usd_spent']:,.0f}")
    assert res.turns == 3
    assert res.submitted and not res.forced_submission
    assert [c["name"] for c in res.tool_calls] == [
        "design_and_run", "design_and_run", "qc", "submit"]
    assert res.n_tool_errors == 0
    assert res.score["diagnostics"]["n_plates"] == 2

    # Two tool_use blocks in one turn produce ONE user message holding BOTH
    # tool_result blocks. Splitting them trains the model out of parallel calls.
    tool_result_msgs = [m for m in res.messages
                        if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(tool_result_msgs) == 3
    assert len(tool_result_msgs[1]["content"]) == 2
    for block in tool_result_msgs[1]["content"]:
        assert block["type"] == "tool_result"
        assert block["is_error"] is False
        json.loads(block["content"])            # results must be valid JSON

    # Every tool_result carries back the id of the tool_use it answers. A real
    # API rejects a mismatch; a stub will not, so it has to be asserted.
    sent_ids = [b["id"] for m in res.messages if m["role"] == "assistant"
                for b in m["content"] if b.get("type") == "tool_use"]
    echoed = [b["tool_use_id"] for m in tool_result_msgs for b in m["content"]]
    # submit's own result is appended before the loop notices env.done, so all
    # four ids round-trip.
    assert sent_ids == ["t1", "t2", "t3", "t4"]
    assert echoed == ["t1", "t2", "t3", "t4"]


# --------------------------------------------------------------------------
# 2. A model that never submits
# --------------------------------------------------------------------------


def test_model_that_never_submits_is_forced_to_an_empty_submission():
    """The episode must still score, with a real zero, not vanish."""
    client = RepeatClient(
        Response([_tool_use("t1", "qc", {"plate_id": "P1"})], "tool_use"))
    res = run_episode(5, "standard", client=client, max_turns=6)

    print(f"[measured] never-submits: turns={res.turns} (max 6), "
          f"submitted={res.submitted}, forced={res.forced_submission}, "
          f"strict_pass={res.score['strict_pass']}, "
          f"endpoint={res.score['endpoint']}")
    assert res.turns == 6
    assert res.submitted is False and res.forced_submission is True
    # The forced submission must actually reach the env, not just produce a
    # zero score. score() reads a missing submission as an empty one, so
    # asserting the score alone cannot tell the two apart -- found by
    # tools/mutate.py, which deleted the submit() call and stayed green.
    assert res.env.done is True
    assert res.env.submission == {"hits": [], "signs": {}, "log_ec50": None}
    assert res.score["strict_pass"] == 0.0
    assert res.score["endpoint"] == 0.0
    assert res.score["shaped"] == 0.0
    assert len(client.requests) == 6

    # Default cap is the spec's 24.
    assert MAX_TURNS == 24
    capped = run_episode(5, "standard", client=RepeatClient(client.response))
    assert capped.turns == 24 and capped.forced_submission is True


# --------------------------------------------------------------------------
# 3. A model that calls a tool with malformed arguments
# --------------------------------------------------------------------------


def test_malformed_tool_arguments_come_back_as_errors_not_crashes():
    """Every bad call is reported to the model and mutates nothing."""
    bad = [
        ("design_and_run", {"layout": "not-a-dict"}),
        ("design_and_run", {"layout": {"Z9": "NTC"}}),
        ("design_and_run", {"layout": {"B2": "NONSENSE"}}),
        ("design_and_run", {"layout": {"B2": "NTC"}, "lot": "LOT-Z"}),
        ("qc", {"plate_id": "P404"}),
        ("exclude_plate", {"plate_id": "P404", "reason": "typo"}),
        ("qc", {"wrong_kwarg": 1}),
        ("nonexistent_tool", {}),
        ("submit", {"hits": "SYN01", "signs": {}, "log_ec50": 2.0}),
    ]
    script = [
        Response([_tool_use(f"t{i}", name, args)], "tool_use")
        for i, (name, args) in enumerate(bad)
    ]
    script.append(Response(
        [_tool_use("done", "submit",
                   {"hits": ["SYN02"], "signs": {"SYN02": 1}, "log_ec50": 2.0})],
        "tool_use"))
    client = StubClient(script)
    res = run_episode(11, "hard", client=client)

    print(f"[measured] malformed args: {len(bad)} bad calls, "
          f"{res.n_tool_errors} reported as errors, "
          f"{res.score['diagnostics']['n_plates']} plates created, "
          f"usd_spent=${res.score['diagnostics']['usd_spent']:,.0f}")
    assert res.n_tool_errors == len(bad)
    assert res.submitted is True and res.forced_submission is False
    # Nothing was created and nothing was spent.
    assert res.score["diagnostics"]["n_plates"] == 0
    assert res.score["diagnostics"]["usd_spent"] == 0.0

    # Each failure went back as an is_error tool_result carrying the message,
    # so the model can see what it did wrong and adapt.
    error_blocks = [b for m in res.messages if m["role"] == "user"
                    and isinstance(m["content"], list) for b in m["content"]
                    if b.get("is_error")]
    assert len(error_blocks) == len(bad)
    for block in error_blocks:
        assert "error" in json.loads(block["content"])


def test_non_dict_tool_input_does_not_crash_the_harness():
    """A model that emits a string where an object belongs is survivable."""
    client = StubClient([
        Response([_tool_use("t1", "qc", "P1")], "tool_use"),
        Response([_tool_use("t2", "submit",
                            {"hits": [], "signs": {}, "log_ec50": None})],
                 "tool_use"),
    ])
    res = run_episode(2, "clean", client=client)
    assert res.tool_calls[0]["is_error"] is True
    assert res.submitted is True
    print("[measured] non-dict tool input coerced to {} and refused cleanly")


# --------------------------------------------------------------------------
# 4. A model that emits no tool call at all
# --------------------------------------------------------------------------


def test_model_that_emits_no_tool_call_is_nudged_then_forced():
    client = RepeatClient(Response([_text("Let me think about this.")], "end_turn"))
    res = run_episode(4, "clean", client=client, max_turns=5)

    assert res.env.done is True and res.env.submission is not None
    nudges = [m for m in res.messages
              if m["role"] == "user" and isinstance(m["content"], str)
              and "did not call a tool" in m["content"]]
    print(f"[measured] no-tool-call: turns={res.turns}, nudges={len(nudges)}, "
          f"forced={res.forced_submission}, strict_pass={res.score['strict_pass']}")
    assert res.turns == 5                      # the nudge cannot loop forever
    assert len(nudges) == 5
    assert res.forced_submission is True
    assert res.score["strict_pass"] == 0.0
    assert res.tool_calls == []


def test_model_that_talks_then_acts():
    """A nudge that works: text turn, then a real submit."""
    client = StubClient([
        Response([_text("Thinking out loud.")], "end_turn"),
        Response([_tool_use("t1", "submit",
                            {"hits": [], "signs": {}, "log_ec50": 1.0})],
                 "tool_use"),
    ])
    res = run_episode(4, "clean", client=client)
    assert res.turns == 2 and res.submitted and not res.forced_submission


# --------------------------------------------------------------------------
# Other stop_reasons
# --------------------------------------------------------------------------


def test_pause_turn_is_resumed_and_refusal_ends_the_episode():
    paused = StubClient([
        Response([_text("partial")], "pause_turn"),
        Response([_tool_use("t1", "submit",
                            {"hits": [], "signs": {}, "log_ec50": None})],
                 "tool_use"),
    ])
    res = run_episode(1, "clean", client=paused)
    assert res.turns == 2 and res.submitted is True
    # The paused assistant turn stays on the history so the next call resumes it.
    assert paused.requests[1]["messages"][-1]["role"] == "assistant"

    refused = RepeatClient(Response([_text("I can't help with that.")], "refusal"))
    res2 = run_episode(1, "clean", client=refused)
    print(f"[measured] pause_turn resumed in {res.turns} turns; refusal ended "
          f"after {res2.turns} turn, forced={res2.forced_submission}, "
          f"strict_pass={res2.score['strict_pass']}")
    assert res2.turns == 1
    assert res2.forced_submission is True and res2.score["strict_pass"] == 0.0


# --------------------------------------------------------------------------
# Thinness: the harness is a reported variable
# --------------------------------------------------------------------------


def test_harness_settings_are_recorded_and_swappable():
    """Two runs are comparable by diffing their `harness` blocks."""
    script = lambda: [Response(  # noqa: E731
        [_tool_use("t1", "submit", {"hits": [], "signs": {}, "log_ec50": None})],
        "tool_use")]
    a = run_episode(1, "clean", client=StubClient(script()))
    b = run_episode(1, "clean", client=StubClient(script()),
                    model="claude-sonnet-5", max_turns=8, max_tokens=4096,
                    system="Terse.", thinking={"type": "adaptive",
                                               "display": "summarized"})
    print(f"[measured] harness block A: {a.harness}")
    print(f"[measured] harness block B: {b.harness}")
    assert a.harness["model"] == DEFAULT_MODEL and b.harness["model"] == "claude-sonnet-5"
    assert a.harness["max_turns"] == 24 and b.harness["max_turns"] == 8
    assert a.harness["n_tools"] == len(TOOL_SPEC) == 4
    # Same seed, same env, identical scores: the harness settings changed and
    # the scoring did not.
    assert a.score == b.score

    events = []
    run_episode(1, "clean", client=StubClient(script()),
                on_event=lambda k, p: events.append(k))
    assert "response" in events and "tool" in events


def test_briefing_is_passed_through_verbatim():
    """A harness that summarised the briefing would be changing the task."""
    from assaygym.env import PRIOR_CAVEAT, AssayGym
    env = AssayGym(9, "hard")
    briefing = env.reset()
    text = format_briefing(briefing)
    assert PRIOR_CAVEAT in text
    for locus in briefing["loci"]:
        assert locus in text
    assert str(briefing["hit_threshold"]) in text
    # And nothing hidden leaks in.
    for hidden in ("true_delta", "compound_target", "bad_lots", "decoys"):
        assert hidden not in text
    print(f"[measured] briefing serialised to {len(text)} chars, caveat present, "
          f"all {len(briefing['loci'])} loci present, no hidden fields")


def test_result_to_dict_is_json_serialisable():
    client = StubClient([Response(
        [_tool_use("t1", "submit", {"hits": [], "signs": {}, "log_ec50": None})],
        "tool_use")])
    res = run_episode(1, "clean", client=client)
    assert isinstance(res, HarnessResult)
    json.dumps(res.to_dict())


# --------------------------------------------------------------------------
# Usage and cost reporting
# --------------------------------------------------------------------------


def test_usage_accumulates_across_turns_and_prices_out():
    """Cost is part of the harness being a reported variable, not a hidden one."""
    usage = {"input_tokens": 1000, "output_tokens": 500,
             "cache_read_input_tokens": 2000,
             "cache_creation_input_tokens": 400}
    client = StubClient([
        Response([_tool_use("t1", "qc", {"plate_id": "P1"})], "tool_use", usage),
        Response([_tool_use("t2", "submit",
                            {"hits": [], "signs": {}, "log_ec50": None})],
                 "tool_use", usage),
    ])
    res = run_episode(1, "clean", client=client, model="claude-sonnet-5")

    assert res.usage == {k: v * 2 for k, v in usage.items()}
    # 3.00/MTok in, 15.00/MTok out; cache read at 0.1x, cache write at 1.25x.
    billed_in = 2000 + 0.1 * 4000 + 1.25 * 800
    expected = billed_in / 1e6 * 3.00 + 1000 / 1e6 * 15.00
    print(f"\n[measured] usage over 2 turns: {res.usage}")
    print(f"[measured] cost = ${res.cost_usd:.6f} (analytic ${expected:.6f})")
    assert res.cost_usd == pytest.approx(expected)
    json.dumps(res.to_dict())

    # An unpriced model reports 0.0 rather than guessing a price.
    assert episode_cost_usd(usage, "some-unreleased-model") == 0.0
    assert set(PRICE_PER_MTOK) >= {"claude-opus-5", "claude-sonnet-5",
                                   "claude-haiku-4-5"}


def test_clients_that_report_no_usage_do_not_crash():
    """Every other stub in this file has no .usage attribute at all."""
    client = StubClient([Response(
        [_tool_use("t1", "submit", {"hits": [], "signs": {}, "log_ec50": None})],
        "tool_use")])
    res = run_episode(1, "clean", client=client)
    assert res.usage == {} and res.cost_usd == 0.0

    # A usage object rather than a dict, which is what the real SDK returns --
    # and whose fields are not all plain ints. `cache_creation` is a nested
    # object on some SDK versions and `None` on others; either must be skipped,
    # not summed. Found by tools/mutate.py: testing only the None case left the
    # int guard unexercised, because a `value is not None` check skips None too.
    class Usage:
        input_tokens, output_tokens = 10, 20
        cache_read_input_tokens = None              # null on some responses
        cache_creation_input_tokens = {"ephemeral_5m_input_tokens": 7}
    obj = StubClient([Response(
        [_tool_use("t1", "submit", {"hits": [], "signs": {}, "log_ec50": None})],
        "tool_use", Usage())])
    res2 = run_episode(1, "clean", client=obj, model="claude-haiku-4-5")
    assert res2.usage == {"input_tokens": 10, "output_tokens": 20}
    assert res2.cost_usd == pytest.approx(10 / 1e6 * 1.0 + 20 / 1e6 * 5.0)
    print(f"[measured] SDK-shaped usage object read correctly: {res2.usage}; "
          f"null and nested-object fields skipped, not summed; "
          f"cost ${res2.cost_usd:.8f}")


# --------------------------------------------------------------------------
# vf_adapter — the fallback path
# --------------------------------------------------------------------------


def test_build_dataset_one_row_per_seed_without_the_answer():
    rows = build_dataset("hard", 5, seed0=100)
    assert len(rows) == 5
    assert [r["info"]["seed"] for r in rows] == [100, 101, 102, 103, 104]
    for row in rows:
        assert set(row) == {"question", "answer", "task", "info"}
        assert row["answer"] == ""            # truth never travels with the row
        assert row["task"] == "assaygym-hard"
        assert row["info"]["budget_usd"] == 3300.0
        assert row["info"]["hit_threshold"] == 0.20
        assert row["info"]["n_loci"] == 12
        blob = json.dumps(row)
        for hidden in ("true_hits", "true_delta", "decoys", "omitted",
                       "compound_target", "true_log_ec50", "bad_lots"):
            assert hidden not in blob, hidden
    print(f"[measured] dataset: {len(rows)} rows, one per seed, answer field "
          f"empty, no ground-truth key present in any row")

    assert build_dataset("clean", 0) == []
    with pytest.raises(ValueError):
        build_dataset("nope", 1)


def test_ground_truth_is_recoverable_from_the_seed_alone():
    """Which is why the row does not need it."""
    row = build_dataset("hard", 1, seed0=41)[0]
    env = make_env(row["info"]["seed"], row["info"]["tier"])
    env.submit(list(env.world.true_hits), dict(env.world.true_signs),
               env.world.true_log_ec50)
    result = score_episode(env)
    print(f"[measured] seed {row['info']['seed']} regraded from the row alone: "
          f"strict_pass={result['strict_pass']}")
    assert result["strict_pass"] == 1.0


def test_load_environment_falls_back_without_verifiers():
    assert "verifiers" not in sys.modules or True   # not installed here
    env = load_environment("standard", n_episodes=4)
    print(f"[measured] fallback: verifiers_available="
          f"{env['verifiers_available']}, reason={env['fallback_reason']!r}")
    assert env["verifiers_available"] is False
    assert "not installed" in env["fallback_reason"]
    assert env["name"] == "assaygym-standard"
    assert len(env["dataset"]) == 4
    assert env["tools"] is TOOL_SPEC
    assert env["rubric"] is REWARD_SPEC
    assert env["rubric"]["primary"] == "strict_pass"

    # The fallback is usable, not just inspectable.
    e = env["make_env"](0, "standard")
    e.submit([], {}, None)
    assert env["score_episode"](e)["strict_pass"] == 0.0

    with pytest.raises(ValueError):
        load_environment("nope")


# --------------------------------------------------------------------------
# vf_adapter — the verifiers path, via an injected fake module
# --------------------------------------------------------------------------


def _install_fake_verifiers(monkeypatch, tool_env):
    module = types.ModuleType("verifiers")
    if tool_env is not None:
        module.ToolEnv = tool_env
    monkeypatch.setitem(sys.modules, "verifiers", module)
    return module


def test_load_environment_uses_verifiers_when_present(monkeypatch):
    captured = {}

    class FakeToolEnv:
        def __init__(self, dataset, tools, rubric, **kwargs):
            captured.update(dataset=dataset, tools=tools, rubric=rubric,
                            kwargs=kwargs)
            self.dataset = dataset

    _install_fake_verifiers(monkeypatch, FakeToolEnv)
    env = load_environment("hard", n_episodes=3, extra="passed-through")

    print(f"[measured] verifiers path: built {type(env).__name__} with "
          f"{len(captured['dataset'])} rows, {len(captured['tools'])} tools, "
          f"extra kwargs {captured['kwargs']}")
    assert isinstance(env, FakeToolEnv)
    assert len(captured["dataset"]) == 3
    assert captured["tools"] is TOOL_SPEC
    assert captured["rubric"] is REWARD_SPEC
    assert captured["kwargs"] == {"extra": "passed-through"}


def test_version_mismatch_degrades_visibly_rather_than_raising(monkeypatch):
    """A trainer stack that moved on must not take the module down with it."""
    class OldToolEnv:
        def __init__(self, examples):      # different signature entirely
            self.examples = examples

    _install_fake_verifiers(monkeypatch, OldToolEnv)
    env = load_environment("clean", n_episodes=2)
    print(f"[measured] signature mismatch -> {env['fallback_reason']}")
    assert env["verifiers_available"] is False
    assert "rejected the arguments" in env["fallback_reason"]
    assert len(env["dataset"]) == 2

    # A verifiers with no ToolEnv at all.
    _install_fake_verifiers(monkeypatch, None)
    env2 = load_environment("clean", n_episodes=1)
    assert env2["verifiers_available"] is False
    assert "no ToolEnv" in env2["fallback_reason"]


def test_reward_spec_matches_the_scorer():
    from assaygym.rewards import ENDPOINT_WEIGHTS, SHAPED_WEIGHTS
    assert REWARD_SPEC["endpoint_weights"] == ENDPOINT_WEIGHTS
    assert REWARD_SPEC["shaped_weights"] == SHAPED_WEIGHTS
    assert REWARD_SPEC["primary"] == "strict_pass"

    env = make_env(0, "clean")
    env.submit([], {}, None)
    result = score_episode(env)
    for metric in REWARD_SPEC["metrics"]:
        assert metric in result, metric
    for diag in REWARD_SPEC["diagnostics"]:
        assert diag in result["diagnostics"], diag
    print(f"[measured] REWARD_SPEC names {len(REWARD_SPEC['metrics'])} metrics "
          f"and {len(REWARD_SPEC['diagnostics'])} diagnostics; all present in "
          f"the scorer output")
