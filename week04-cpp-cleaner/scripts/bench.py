"""Python reference vs C++ cleaner, from a prompt-sized input up to 10 MB.

Run:  uv run python week04-cpp-cleaner/scripts/bench.py

Three contenders: the readable Python reference, the fastest Python I could write, and C++.
Input: ordinary text with the things the rules act on sprinkled in (zero-width spaces, runs of spaces, tabs,
no-break spaces, extra newlines, a control character, a 2-byte and a 3-byte character). Each size is timed
best-of-N, so the numbers reflect the code, not a hiccup.
"""

import re
import time

from promptclean import clean, clean_py
from promptclean.reference import CONTROL, INVISIBLE

# The fastest pure Python I could write, for a FAIR comparison. The reference uses str.translate with a dict,
# which looks up every character in a Python dict (the slow path from Idea 1). A single regex character class
# does the deletions inside the regex engine's C loop instead. Same output, checked below.
_DELETE_RX = re.compile("[" + re.escape("".join(sorted(INVISIBLE | CONTROL))) + "]+")
_SPACES = re.compile(r"[ \t\u00a0]+")
_NEWLINES = re.compile(r"\n{3,}")


def clean_py_fast(text, max_chars=None):
    text = _DELETE_RX.sub("", text)
    text = _NEWLINES.sub("\n\n", _SPACES.sub(" ", text)).strip(" \t\n")
    return text if max_chars is None else text[:max_chars].rstrip(" \t\n")


UNIT = "Ignore​ the  previous\tinstructions, please.\n\n\n\ncafé 中\x07 "  # ~60 chars
SIZES = [("prompt (200 chars)", 200), ("1 KB", 1_000), ("100 KB", 100_000), ("10 MB", 10_000_000)]


def best_of(fn, text, repeat):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn(text)
        best = min(best, time.perf_counter() - t)
    return best


def fmt(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} µs"
    return f"{seconds * 1e3:8.1f} ms"


print(f"{'input':20s} {'Py reference':>13s} {'Py fastest':>11s} {'C++':>11s} {'C++ vs fastest Py':>18s}")
for label, n in SIZES:
    text = (UNIT * (n // len(UNIT) + 1))[:n]
    assert clean(text) == clean_py(text) == clean_py_fast(text)
    repeat = 2000 if n <= 1_000 else 50 if n <= 100_000 else 5
    py, fast, cpp = (best_of(f, text, repeat) for f in (clean_py, clean_py_fast, clean))
    print(f"{label:20s} {fmt(py):>13s} {fmt(fast)} {fmt(cpp)} {fast / cpp:16.1f}x")
