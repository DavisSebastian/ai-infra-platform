"""Property-based test: the C++ cleaner must match the Python reference on ANY text, not just the examples we
thought of. hypothesis generates thousands of strings and, when one fails, shrinks it to the smallest failing case.

The alphabet is weighted towards the characters the rules care about, plus multi-byte characters (2, 3 and 4
bytes in UTF-8), because a byte-level bug in C++ hides in exactly those.
"""

from hypothesis import given, settings
from hypothesis import strategies as st
from promptclean import clean, clean_py

TRICKY = (
    " \t\n\r "  # whitespace the rules collapse, and \r which they delete
    "\x00\x07\x7f\x85\x9f"  # control characters (1 and 2 bytes)
    "​‌‍⁠﻿­‪‮⁦⁩"  # invisible (ZWJ ‍ must survive)
    "aé中\U0001f642"  # 1, 2, 3 and 4-byte characters
)
tricky_text = st.text(alphabet=st.sampled_from(TRICKY), max_size=60)
any_text = st.text(max_size=200)  # all of Unicode (hypothesis leaves out lone surrogates, which UTF-8 can't encode)


@settings(max_examples=3000, deadline=None)
@given(text=st.one_of(tricky_text, any_text))
def test_cpp_matches_python_on_any_text(text):
    assert clean(text) == clean_py(text)


@settings(max_examples=3000, deadline=None)
@given(text=st.one_of(tricky_text, any_text), max_chars=st.integers(min_value=0, max_value=70))
def test_cpp_matches_python_with_a_length_cap(text, max_chars):
    assert clean(text, max_chars=max_chars) == clean_py(text, max_chars=max_chars)
