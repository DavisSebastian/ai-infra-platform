"""Step 2: the compiled module builds, installs and imports."""

from promptclean import utf8_bytes


def test_native_module_counts_utf8_bytes_not_characters():
    assert utf8_bytes("abc") == 3
    assert len("​") == 1 and utf8_bytes("​") == 3  # one character, three bytes: E2 80 8B
    assert len("\U0001f642") == 1 and utf8_bytes("\U0001f642") == 4
