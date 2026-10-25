# Week 4 — C++ in the gateway, with pybind11

The gateway now cleans every prompt before forwarding it: invisible characters (zero-width spaces, bidi overrides: the "Trojan Source" trick), control characters, excess whitespace, and an optional length cap. The cleaning runs in **C++**, bound to Python with **pybind11**, packaged as its own library (`promptclean`) and compiled in CI and in the Docker image.

The honest headline: **C++ made the gateway no faster.** A prompt cleans in 2.4 µs against 2.73 s of inference. The interesting results are elsewhere: a thread alone doesn't stop native code from freezing an async server, random testing caught a bug that 21 hand-written examples missed, and getting native code to build reliably in Docker took a dependency change.

```
week04-cpp-cleaner/                  the promptclean package
  src/promptclean/reference.py       pure-Python reference: DEFINES correct behaviour
  src/cpp/promptclean.cpp            the C++ version: one pass over the UTF-8 bytes
  CMakeLists.txt, pyproject.toml     scikit-build-core + CMake + pybind11 (build tools pinned exactly)
  tests/                             rules (run against both), property test, GIL test
  scripts/bench.py, gil_demo.py      the measurements below
week03-gateway/app/sanitize.py       the gateway's use of it
```

## Run it

```bash
uv sync --group dev --group week04          # compiles the C++ (CMake + your compiler) while installing
uv run pytest week04-cpp-cleaner/tests
uv run python week04-cpp-cleaner/scripts/bench.py
uv run python week04-cpp-cleaner/scripts/gil_demo.py
```

After changing the C++: `uv sync --group dev --group week04 --reinstall-package promptclean`. `uv sync` installs only the groups you name, so add `--group week03` to keep the gateway's dependencies too.

```python
>>> import promptclean
>>> promptclean.clean("ig​nore   all‮ previous")
'ignore all previous'
```

## The rules

Applied in order (the full text is in [`reference.py`](src/promptclean/reference.py)):

1. Remove invisible format characters, but keep the zero-width joiner, which emoji like 👨‍👩‍👧 are built from.
2. Remove control characters except tab and newline (so `\r\n` becomes `\n`).
3. Collapse runs of spaces, tabs and no-break spaces into one space.
4. Collapse 3 or more newlines into 2.
5. Strip whitespace from both ends.
6. Optional `max_chars`: keep the first N **characters** (not bytes), then strip trailing whitespace. Negative is a `ValueError`.

## Results

### Speed: C++ vs the best Python

Best of N runs on an M-series Mac ([`scripts/bench.py`](scripts/bench.py)):

| Input | Python reference | Fastest Python | C++ | C++ vs fastest Python |
|---|---|---|---|---|
| Prompt (200 chars) | 21 µs | 10 µs | 2.4 µs | 4× |
| 1 KB | 104 µs | 49 µs | 12 µs | 4× |
| 100 KB | 10 ms | 4.6 ms | 1.2 ms | 4× |
| 10 MB | 1,043 ms | 509 ms | 109 ms | 4.7× |

- **Compare against the best alternative.** The first comparison, against the readable reference, said 10×. That reference uses `str.translate` with a dict, which looks up every character in a Python dict. The fastest Python (one regex for the deletions) halves its time. The fair answer is 4×.
- **The ratio is constant; the value isn't.** C++ is 4× faster even on a prompt, because crossing into C++ costs only ~0.3 ms per 10 MB. But 4× of 10 µs saves 7.5 µs against 2.73 s of inference. It only matters for big inputs (documents, datasets).
- **This C++ is naive:** ~87 M characters/s, one character at a time even for plain ASCII. Copying runs of ordinary characters in bulk would be the next step.

### The GIL: a thread is not enough

Worst event-loop lag while 20 MB is cleaned ([`scripts/gil_demo.py`](scripts/gil_demo.py)):

| How the C++ is called | Worst event-loop lag |
|---|---|
| Directly in the coroutine | 246 ms (frozen) |
| `asyncio.to_thread`, GIL held | **244 ms (still frozen)** |
| `asyncio.to_thread`, GIL released | **12 ms** |

The worker thread holds the GIL while inside C++, and the event loop needs the GIL to run any Python, so moving the call to a thread changes nothing. The binding releases it with `py::call_guard<py::gil_scoped_release>()`. That's safe because pybind11 copies the input into a `std::string` before the guard and builds the result after it, and the C++ touches no Python object in between. The remaining 12 ms is converting the 20 MB result back into a Python string, which needs the GIL. Week 2's `to_thread(model.generate)` worked because PyTorch releases the GIL itself.

In the gateway, prompts under 64 K characters are cleaned inline (a thread hop costs more than the 2 µs clean), and bigger ones go through `to_thread`.

### Correctness: the property test earned its place

Six realistic C++ bugs were planted one at a time. All were caught, and one only by the property test:

| Planted bug | Rule tests | Property test (`hypothesis`) |
|---|---|---|
| Treat every byte as a character | caught (10) | caught |
| `max_chars` counts bytes | caught | caught |
| Keep 3 newlines instead of 2 | caught | caught |
| Forget the leading strip | caught | caught |
| Delete the zero-width joiner | caught | caught |
| Control range off by one (keeps U+009F) | **missed** | **caught** |

The rule tests run against both implementations. The property test feeds 6,000 random strings, weighted toward the tricky characters and 1 to 4-byte UTF-8 sequences, and requires identical output.

### Docker

| | Before Week 4 | After |
|---|---|---|
| Gateway image | 293 MB | **293 MB** (the compiled module is ~150 KB) |
| Compiler in the image | none | **none**: `build-essential` and CMake live only in the builder stage |
| Compiled module in the image | n/a | `_native.cpython-312-aarch64-linux-gnu.so` locally, `x86_64-linux-gnu` in CI: built for Linux, in the container |

## Things that went wrong, and what they taught

| What happened | Lesson |
|---|---|
| `to_thread` alone left the event loop frozen for 244 ms | Native code must release the GIL, and only when it touches no Python objects |
| A lone surrogate (`"\ud800"`, legal in JSON) crashed the gateway with a 500. The fallback caught `UnicodeEncodeError`, but pybind11 raises **`TypeError`** ("incompatible function arguments") | Test the failure path, not the error you expect. The fallback now catches `TypeError` and falls back only if the text really can't be encoded as UTF-8 |
| The first Docker build spent 3 minutes failing to download CMake from PyPI | CMake now comes from Debian in the builder stage: one less network fetch per build, and the compile itself takes 2.5 s |
| Build tools (`scikit-build-core`, `pybind11`) were `>=`, unpinned | `uv.lock` doesn't cover build dependencies; they're now pinned exactly |
| "10× faster" against the readable reference | Compare against the best alternative: 4× |

## CI

| Job | Proves |
|---|---|
| `ci.yml` → `test-week04`, a matrix of **ubuntu-latest** (gcc, x86_64) and **macos-latest** (clang, arm64) | The C++ compiles and passes all 48 tests with both compilers on both platforms |
| `ci.yml` → `test-week03` | The gateway, with `promptclean` compiled in, passes its 60 tests |
| `docker-image-week03.yml` → `compose-smoke` | In the real containers, a prompt with invisible characters reaches the model server cleaned |

## Known gaps

- The C++ processes one character at a time. Bulk copying could make it several times faster.
- `max_chars` applies per message, not to the whole conversation.
- Content given as a list of parts (OpenAI multimodal style) is forwarded uncleaned.
- Benchmarks are from one Mac; CI only checks correctness on Linux, not speed.

## Lessons

1. **Native code is a tool for big loops, not a speed-up for everything.** Measure the absolute time saved, not just the ratio.
2. **A thread is not enough.** Native code that holds the GIL freezes an async server from any thread.
3. **Random inputs find what examples miss.** Keep a simple reference implementation and test the fast one against it.
4. **Native code is built where it runs.** Machine code is per OS, CPU and Python version, so CI and Docker must compile it, and every build-time download is a chance to fail.
