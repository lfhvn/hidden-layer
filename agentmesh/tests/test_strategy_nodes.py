"""
Smoke tests for agentmesh's research-strategy integration.

agentmesh is the product layer that wraps the multi-agent research strategies; its
``StrategyNode.execute`` calls ``harness.run_strategy``. ``run_strategy`` actually
lives in ``communication.multi_agent`` and is lazily re-exported by the harness, so
these tests guard against that wiring silently breaking again.

Everything runs offline via the deterministic ``sim`` provider (no API keys, no
network), so this doubles as a CI-safe end-to-end check.
"""

import asyncio

import pytest

from agentmesh.core.models import ExecutionContext
from agentmesh.core.nodes.strategy_nodes import NODE_REGISTRY, get_strategy_node


def _run(node, task, ctx):
    return asyncio.run(node.execute({"task": task}, ctx))


def test_harness_reexports_run_strategy():
    # The whole integration hinges on this being importable from the harness.
    from harness import run_strategy

    assert callable(run_strategy)


def test_single_node_executes_offline():
    ctx = ExecutionContext(provider="sim", model="sim")
    result = _run(get_strategy_node("single", {}), "What is 2+2?", ctx)
    assert isinstance(result.output, str) and result.output
    assert result.metadata["strategy"] == "single"
    assert result.latency_s >= 0.0


def test_debate_node_executes_offline():
    ctx = ExecutionContext(provider="sim", model="sim")
    result = _run(get_strategy_node("debate", {"n_debaters": 2, "n_rounds": 1}), "2+2?", ctx)
    assert isinstance(result.output, str) and result.output


def test_every_wired_strategy_runs():
    # Every node whose strategy_id is actually backed by the harness strategy
    # registry should execute without raising via the sim provider.
    from communication.multi_agent import STRATEGIES

    ctx = ExecutionContext(provider="sim", model="sim")
    wired = [sid for sid in NODE_REGISTRY if sid in STRATEGIES]
    assert wired, "expected at least one agentmesh node backed by a real strategy"
    for strategy_id in wired:
        result = _run(get_strategy_node(strategy_id, {}), "What is the capital of France?", ctx)
        assert result.output is not None, f"{strategy_id} produced no output"


def test_crit_node_is_not_yet_wired():
    # KNOWN GAP: agentmesh exposes a CRITNode (strategy_id="crit"), but CRIT is a
    # separate subsystem and is not in run_strategy's STRATEGIES registry, so the
    # node currently raises. This test documents that gap; when CRIT is wired into
    # run_strategy it will fail, prompting both this test and the product to update.
    from communication.multi_agent import STRATEGIES

    assert "crit" in NODE_REGISTRY
    assert "crit" not in STRATEGIES
    ctx = ExecutionContext(provider="sim", model="sim")
    with pytest.raises(ValueError):
        _run(get_strategy_node("crit", {}), "Critique this design.", ctx)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_strategy_node("does_not_exist", {})
