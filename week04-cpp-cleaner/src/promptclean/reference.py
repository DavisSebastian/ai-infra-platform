"""The pure-Python reference cleaner. It defines what "correct" means; the C++ version must match it exactly.

The rules, applied in this order:
  1. REMOVE invisible format characters, often used to hide text from humans but not from the model:
     zero-width space U+200B, zero-width non-joiner U+200C, word joiner U+2060, byte-order mark U+FEFF,
     soft hyphen U+00AD, and the bidirectional controls U+202A-U+202E and U+2066-U+2069 ("Trojan Source").
     KEEP the zero-width joiner U+200D: emoji like the family emoji are built from it.
  2. REMOVE control characters U+0000-U+001F (except tab and newline), U+007F, and U+0080-U+009F.
     (So "\r\n" becomes "\n".)
  3. COLLAPSE every run of spaces, tabs and no-break spaces (U+00A0) into one plain space.
  4. COLLAPSE every run of 3 or more newlines into exactly 2 (keep paragraph breaks, drop the excess).
  5. STRIP whitespace (spaces, tabs, newlines) from both ends.
  6. If max_chars is given: keep the first max_chars CHARACTERS (not bytes), then strip trailing whitespace again.
     A negative max_chars is a ValueError.
"""

import re

INVISIBLE = frozenset("​‌⁠﻿­" + "".join(map(chr, range(0x202A, 0x202F))) + "".join(map(chr, range(0x2066, 0x206A))))


CONTROL = frozenset(chr(c) for c in [*range(0x00, 0x20), 0x7F, *range(0x80, 0xA0)] if chr(c) not in "\t\n")
# str.translate with a {codepoint: None} table deletes those characters (rules 1 and 2 in one call).
_DELETE = {ord(c): None for c in INVISIBLE | CONTROL}
_HORIZONTAL_SPACE_RUN = re.compile(r"[ \t\u00a0]+")  # rule 3
_EXCESS_NEWLINES = re.compile(r"\n{3,}")  # rule 4
_EDGE_WHITESPACE = " \t\n"  # rule 5 (no-break spaces are already plain spaces by then)


def clean(text: str, max_chars: int | None = None) -> str:
    """Clean a prompt by the rules above. Returns a new string.

    Readable, not fast: four passes over the text. The C++ version does the same work in one pass.
    """
    if max_chars is not None and max_chars < 0:
        raise ValueError("max_chars must be >= 0")  # Python slicing would silently drop characters from the end
    text = text.translate(_DELETE)  # rules 1 + 2
    text = _HORIZONTAL_SPACE_RUN.sub(" ", text)  # rule 3
    text = _EXCESS_NEWLINES.sub("\n\n", text)  # rule 4
    text = text.strip(_EDGE_WHITESPACE)  # rule 5
    if max_chars is not None:  # rule 6: Python slices count characters, so this never splits one
        text = text[:max_chars].rstrip(_EDGE_WHITESPACE)
    return text
