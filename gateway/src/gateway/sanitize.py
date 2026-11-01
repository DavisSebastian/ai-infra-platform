"""Prompt cleaning before forwarding (Week 4): the gateway's use of the C++ `promptclean` package.

For each chat message whose `content` is a string, apply promptclean's rules (strip invisible and control
characters, collapse whitespace, optional length cap). Everything else in the body is left alone.

Two paths, chosen by size:
- normal prompts are cleaned INLINE: the C++ takes ~2 µs for 200 characters, less than handing work to a thread.
- big inputs go through asyncio.to_thread: promptclean.clean releases the GIL, so the event loop keeps serving
  other requests while it works (Week 4 step 6: a thread alone is not enough; the GIL must be released too).
"""

import asyncio
import json

import promptclean

OFFLOAD_THRESHOLD = 64_000  # characters across all messages; above this, clean on a worker thread


def _clean_one(text: str, max_chars: int | None) -> str:
    try:
        return promptclean.clean(text, max_chars)
    except TypeError:
        # JSON allows "\ud800" escapes (lone surrogates) that can't be encoded as UTF-8, so pybind11 can't turn
        # the text into a std::string and reports "incompatible function arguments" (a TypeError, not a
        # UnicodeEncodeError). Only in that case fall back to the Python reference, which applies the same rules;
        # any other TypeError is a real bug and is re-raised.
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return promptclean.clean_py(text, max_chars)
        raise


async def clean_body(body: bytes, max_chars: int | None) -> tuple[bytes, int]:
    """Return (the body to forward, how many characters cleaning removed).

    A body that isn't a JSON object with a `messages` list is returned unchanged: the model server validates
    requests and answers 422, and the gateway shouldn't invent a second set of rules. An unchanged body is
    forwarded byte for byte.
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body, 0
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        return body, 0

    targets = [m for m in messages if isinstance(m, dict) and isinstance(m.get("content"), str)]
    total = sum(len(m["content"]) for m in targets)
    removed = 0
    for m in targets:
        before = m["content"]
        if total >= OFFLOAD_THRESHOLD:
            after = await asyncio.to_thread(_clean_one, before, max_chars)
        else:
            after = _clean_one(before, max_chars)
        removed += len(before) - len(after)
        m["content"] = after

    if removed == 0:
        return body, 0
    return json.dumps(payload).encode(), removed
