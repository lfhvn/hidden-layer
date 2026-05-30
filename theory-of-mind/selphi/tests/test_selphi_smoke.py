"""Offline smoke tests for SELPHI (theory-of-mind evaluation).

Runs via the harness ``sim`` provider — no API keys, no network.
"""

import theory_of_mind.selphi as selphi


def test_scenarios_are_defined():
    assert len(selphi.ALL_SCENARIOS) >= 5
    assert isinstance(selphi.SALLY_ANNE, selphi.ToMScenario)
    # Sally-Anne is the canonical false-belief task.
    assert selphi.SALLY_ANNE.tom_type == selphi.ToMType.FALSE_BELIEF


def test_tom_type_taxonomy():
    for member in ("FALSE_BELIEF", "PERSPECTIVE_TAKING", "SECOND_ORDER_BELIEF", "EPISTEMIC_STATE"):
        assert hasattr(selphi.ToMType, member)


def test_scenario_indexes_are_consistent():
    assert selphi.SALLY_ANNE.name in selphi.SCENARIOS_BY_NAME
    # Every scenario is reachable by its ToM type.
    by_type = selphi.SCENARIOS_BY_TYPE
    assert selphi.SALLY_ANNE in by_type[selphi.ToMType.FALSE_BELIEF]


def test_run_scenario_offline():
    result = selphi.run_scenario(selphi.SALLY_ANNE, provider="sim", model="sim", sim_response="basket")
    assert type(result).__name__ == "ToMTaskResult"
    assert result.scenario_name == selphi.SALLY_ANNE.name
    assert "basket" in result.model_response.lower()
