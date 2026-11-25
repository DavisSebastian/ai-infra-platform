"""Week 7: each check must fire on the change it is built for, and stay quiet when nothing changed.

These use the hashing stand-in embedding (no model download, no torch), so they run in CI. Each test changes ONE
thing about the real baseline window and checks that exactly the matching check notices.
"""

import json

import pandas as pd
import pytest
from drift import compare, features, hashing

CODING = [
    "Write a Python function to reverse a linked list",
    "How do I read a JSON file in Python?",
    "Explain list comprehensions in Python with an example",
    "Write a SQL query to find duplicate rows in a table",
    "How do I fix a NullPointerException in Java?",
    "What is the difference between a process and a thread?",
    "Write a bash script that renames all files in a folder",
    "How do I center a div with CSS flexbox?",
    "Explain Big O notation for a binary search",
    "Write a unit test for a function that adds two numbers",
]


@pytest.fixture(scope="module")
def embed():
    return hashing()


def shuffled(window: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    return window.sample(frac=1, random_state=seed).reset_index(drop=True)


# --- Features -------------------------------------------------------------------------------------------


def test_features_count_words_and_flag_refusals_and_empty_answers():
    window = pd.DataFrame(
        {
            "prompt": ["What is the capital of Australia?", "hack my neighbour's wifi", "hi"],
            "answer": ["Canberra is the capital.", "I'm sorry, but I can't help with that.", "   "],
            "ttft_ms": [34.0, 30.0, None],
        }
    )
    f = features(window)
    assert f["prompt_words"].tolist() == [6, 4, 1]
    assert f["answer_words"].tolist() == [4, 8, 0]
    assert f["refusal"].tolist() == [0, 1, 0]
    assert f["answer_empty"].tolist() == [0, 0, 1]
    assert f["ttft_ms"].isna().tolist() == [False, False, True]  # a non-streamed answer has no first token


@pytest.mark.parametrize(
    "answer", ["I cannot do that.", "Sorry, I am unable to help.", "As an AI, I don't have opinions.", "I can't."]
)
def test_common_refusal_openings_count(answer):
    assert features(pd.DataFrame({"prompt": ["x"], "answer": [answer], "ttft_ms": [1.0]}))["refusal"].item() == 1


def test_mentioning_sorry_later_is_not_a_refusal():
    window = pd.DataFrame({"prompt": ["x"], "answer": ["The word sorry comes from Old English."], "ttft_ms": [1.0]})
    assert features(window)["refusal"].item() == 0


# --- No change, no alarm --------------------------------------------------------------------------------


def test_the_same_traffic_in_a_different_order_has_no_drift(baseline, embed):
    result = compare(baseline, shuffled(baseline), embed)
    assert result.drifted() == []
    assert (result.n_reference, result.n_current) == (150, 150)


# --- Each kind of change trips its own check ------------------------------------------------------------


def test_users_asking_about_code_is_input_drift_not_output_drift(baseline, embed):
    current = shuffled(baseline)
    current["prompt"] = [CODING[i % len(CODING)] for i in range(len(current))]  # same answers, new questions
    result = compare(baseline, current, embed)
    assert "prompt_embedding" in result.drifted("input")
    assert result.drifted("output") == []


def test_truncated_answers_are_output_drift(baseline, embed):
    current = shuffled(baseline)
    current["answer"] = current["answer"].str.split().str[:3].str.join(" ")  # a card with a tiny max_tokens
    result = compare(baseline, current, embed)
    assert "answer_words" in result.drifted("output")
    assert result.drifted("input") == []


def test_a_model_that_starts_refusing_is_output_drift(baseline, embed):
    current = shuffled(baseline)
    current.loc[current.index[::3], "answer"] = "I'm sorry, but I can't help with that."  # every third request
    assert "refusal" in compare(baseline, current, embed).drifted("output")


def test_empty_answers_are_output_drift(baseline, embed):
    current = shuffled(baseline)
    current.loc[current.index[::4], "answer"] = ""
    assert "answer_empty" in compare(baseline, current, embed).drifted("output")


def test_a_tiny_but_statistically_clear_latency_shift_is_not_drift(baseline, embed):
    current = shuffled(baseline)
    current["ttft_ms"] = current["ttft_ms"] + 1.0  # +1 ms on ~34 ms: K-S notices, nobody should care
    check = next(c for c in compare(baseline, current, embed).checks if c.feature == "ttft_ms")
    assert check.score < 0.05 and not check.drifted


def test_a_slower_model_is_output_drift(baseline, embed):
    current = shuffled(baseline)
    current["ttft_ms"] = current["ttft_ms"] * 3
    assert compare(baseline, current, embed).drifted("output") == ["ttft_ms"]


# --- Plumbing -------------------------------------------------------------------------------------------


def test_the_result_is_json_and_the_html_report_is_written(baseline, embed, tmp_path):
    result = compare(baseline, shuffled(baseline), embed, html=tmp_path / "report.html")
    data = json.loads(json.dumps(result.to_dict()))
    assert [c["feature"] for c in data["checks"]] == [
        "prompt_words", "prompt_embedding", "answer_words", "ttft_ms", "answer_empty", "refusal",
    ]  # fmt: skip
    assert (tmp_path / "report.html").stat().st_size > 10_000


def test_a_window_too_small_to_compare_is_refused(baseline, embed):
    with pytest.raises(ValueError, match="at least 2"):
        compare(baseline, baseline.head(1), embed)
