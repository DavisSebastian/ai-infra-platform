# ai-infra-platform

Hands-on path from backend engineering into AI infrastructure / LLM serving, building toward a capstone benchmark suite.

| Week | Topic | Folder |
|---|---|---|
| 1 | Async serving foundations: FastAPI + uvloop, SSE streaming, load testing | [week01-async-serving](week01-async-serving/) |
| 2 | The Thin Slice: a real LLM streamed from FastAPI, tests, CI, Docker | [week02-thin-slice](week02-thin-slice/) |

Setup: `uv sync --group week01` or `uv sync --group week02` (Python 3.12, managed by [uv](https://docs.astral.sh/uv/)).
