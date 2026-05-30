"""Tests for the harness 'sim' provider - the deterministic offline LLM stand-in."""

from harness import llm_call


def test_canned_response_returned_verbatim():
    r = llm_call("anything", provider="sim", sim_response="Answer: 42\nConfidence: 80%")
    assert r.text == "Answer: 42\nConfidence: 80%"
    assert r.provider == "sim"
    assert r.cost_usd == 0.0
    assert r.metadata["deterministic"] is True


def test_placeholder_is_deterministic_and_prompt_sensitive():
    a1 = llm_call("prompt A", provider="sim").text
    a2 = llm_call("prompt A", provider="sim").text
    b = llm_call("prompt B", provider="sim").text
    assert a1 == a2  # same prompt -> same output
    assert a1 != b  # different prompt -> different output


def test_sim_responses_list_is_stable_per_prompt():
    options = ["red", "green", "blue"]
    first = llm_call("pick one", provider="sim", sim_responses=options).text
    second = llm_call("pick one", provider="sim", sim_responses=options).text
    assert first == second
    assert first in options
    # Across many prompts we should see more than one option chosen.
    chosen = {llm_call(f"q{i}", provider="sim", sim_responses=options).text for i in range(30)}
    assert len(chosen) > 1


def test_sim_seed_changes_output_reproducibly():
    base = llm_call("same prompt", provider="sim", sim_seed=0).text
    seeded = llm_call("same prompt", provider="sim", sim_seed=1).text
    seeded_again = llm_call("same prompt", provider="sim", sim_seed=1).text
    assert base != seeded  # seed perturbs the stream
    assert seeded == seeded_again  # but is reproducible


def test_empty_sim_responses_list():
    r = llm_call("x", provider="sim", sim_responses=[])
    assert r.text == ""
