"""Prompt cleaning for the gateway.

`clean_py` is the pure-Python reference: it DEFINES correct behaviour.
`clean` is the C++ version (pybind11): one pass over the UTF-8 bytes, and it must produce exactly the same output
for every input. tests/test_equivalence.py checks that on random Unicode.
"""

from promptclean._native import clean, utf8_bytes
from promptclean.reference import clean as clean_py

__all__ = ["clean", "clean_py", "utf8_bytes"]
