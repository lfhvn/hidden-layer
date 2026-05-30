"""Tests for dataset scoring, confidence parsing, and the simulated subject."""

import pytest

from theory_of_mind.metacognition import (
    Question,
    SimulatedSubject,
    build_prompt,
    load_dataset,
    make_arithmetic_dataset,
    parse_answer_and_confidence,
)


def test_bundled_dataset_loads():
    ds = load_dataset()
    assert len(ds) >= 20
    assert all(q.answers for q in ds)


def test_is_correct_token_boundaries():
    q = Question(id="t", question="symbol for gold?", answers=["au"])
    assert q.is_correct("The answer is Au.")
    # 'au' must not match inside another word like 'because'.
    assert not q.is_correct("because I said so")


def test_is_correct_multiword_phrase():
    q = Question(id="t", question="who painted it?", answers=["leonardo da vinci"])
    assert q.is_correct("It was Leonardo da Vinci, the master.")
    assert not q.is_correct("Leonardo DiCaprio")


@pytest.mark.parametrize(
    "text,expected_conf",
    [
        ("Answer: Paris\nConfidence: 90%", 0.90),
        ("Answer: Paris\nConfidence: 0.75", 0.75),
        ("Answer: Paris\nConfidence: 60", 0.60),
        ("I am about 85 percent sure it is Paris", 0.85),
        ("Answer: Paris", None),
    ],
)
def test_confidence_parsing(text, expected_conf):
    _, conf = parse_answer_and_confidence(text)
    if expected_conf is None:
        assert conf is None
    else:
        assert conf == pytest.approx(expected_conf)


def test_answer_parsing_extracts_answer_line():
    ans, _ = parse_answer_and_confidence("Answer: the Pacific Ocean\nConfidence: 70%")
    assert "pacific" in ans.lower()


def test_build_prompt_contains_question():
    q = Question(id="t", question="What is 2 plus 2?", answers=["4"])
    assert "2 plus 2" in build_prompt(q)


def test_simulated_subject_is_deterministic():
    ds = make_arithmetic_dataset(n=50, seed=1)
    s1 = SimulatedSubject(accuracy=0.6, profile="calibrated", seed=5)
    s2 = SimulatedSubject(accuracy=0.6, profile="calibrated", seed=5)
    r1 = [s1.answer(q) for q in ds]
    r2 = [s2.answer(q) for q in ds]
    assert [(r.correct, r.confidence) for r in r1] == [(r.correct, r.confidence) for r in r2]


def test_profiles_shift_confidence_as_expected():
    ds = make_arithmetic_dataset(n=400, seed=2)

    def mean_conf(profile):
        s = SimulatedSubject(accuracy=0.7, profile=profile, seed=3)
        recs = [s.answer(q) for q in ds]
        return sum(r.confidence for r in recs) / len(recs)

    over = mean_conf("overconfident")
    cal = mean_conf("calibrated")
    under = mean_conf("underconfident")
    assert over > cal > under


def test_invalid_profile_rejected():
    with pytest.raises(ValueError):
        SimulatedSubject(profile="nonsense")
