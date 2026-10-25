"""The cleaning rules, as tests. Every test runs twice: against the Python reference and the C++ version."""

import pytest


def test_normal_text_is_unchanged(clean):
    assert clean("Explain a GPU in two sentences.") == "Explain a GPU in two sentences."


# --- rule 1: invisible characters -----------------------------------------------------------------------------


def test_zero_width_characters_are_removed(clean):
    # Looks like "ignore all previous" to a human, but isn't the same string without cleaning.
    assert clean("ig​nore all prev‌ious") == "ignore all previous"


def test_bidi_override_is_removed(clean):
    assert clean("abc‮dcba") == "abcdcba"


def test_soft_hyphen_and_bom_are_removed(clean):
    assert clean("﻿co­operate") == "cooperate"


def test_zero_width_joiner_is_kept_so_emoji_survive(clean):
    family = "\U0001f468‍\U0001f469‍\U0001f467"  # man ZWJ woman ZWJ girl: renders as one emoji
    assert clean(family) == family


# --- rule 2: control characters --------------------------------------------------------------------------------


def test_control_characters_are_removed(clean):
    assert clean("a\x00b\x07c\x7fd\x85e") == "abcde"


def test_windows_line_endings_become_plain_newlines(clean):
    assert clean("line1\r\nline2") == "line1\nline2"


# --- rules 3 and 4: whitespace ---------------------------------------------------------------------------------


def test_runs_of_spaces_tabs_and_no_break_spaces_become_one_space(clean):
    assert clean("a   \t  b  c") == "a b c"


def test_three_or_more_newlines_become_two(clean):
    assert clean("p1\n\n\n\n\np2") == "p1\n\np2"


def test_one_or_two_newlines_are_kept(clean):
    assert clean("a\nb") == "a\nb"
    assert clean("p1\n\np2") == "p1\n\np2"


def test_invisible_characters_are_removed_before_whitespace_is_collapsed(clean):
    # Order matters: removing the zero-width space first leaves "a  b", which then collapses to "a b".
    assert clean("a ​ b") == "a b"


# --- rule 5: trimming ------------------------------------------------------------------------------------------


def test_whitespace_is_trimmed_from_both_ends(clean):
    assert clean("  \n hello \n\n ") == "hello"


def test_empty_and_whitespace_only_input_gives_empty_string(clean):
    assert clean("") == ""
    assert clean(" \t\n ​ ") == ""


# --- rule 6: length cap ----------------------------------------------------------------------------------------


def test_max_chars_counts_characters_not_bytes(clean):
    assert clean("éééé", max_chars=2) == "éé"  # each é is 2 bytes in UTF-8
    assert clean("\U0001f642\U0001f642\U0001f642", max_chars=2) == "\U0001f642\U0001f642"  # each emoji is 4 bytes


def test_max_chars_applies_after_removal(clean):
    assert clean("ab​cd", max_chars=3) == "abc"


def test_max_chars_then_trailing_whitespace_is_stripped(clean):
    assert clean("hello world", max_chars=6) == "hello"


@pytest.mark.parametrize("text", ["", "abc", "hello world"])
def test_max_chars_zero_gives_empty_string(clean, text):
    assert clean(text, max_chars=0) == ""


def test_negative_max_chars_is_an_error(clean):
    with pytest.raises(ValueError):
        clean("abc", max_chars=-1)


def test_no_max_chars_means_no_limit(clean):
    long = "word " * 10_000
    assert clean(long) == long.strip()
