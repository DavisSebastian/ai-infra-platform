# ai-infra-platform

Hands-on path from backend engineering into AI infrastructure / LLM serving, building toward a capstone benchmark suite.

| Week | Topic | Folder |
|---|---|---|
| 1 | Async serving foundations: FastAPI + uvloop, SSE streaming, load testing | [week01-async-serving](week01-async-serving/) |
| 2 | The Thin Slice: a real LLM streamed from FastAPI, tests, CI, Docker | [week02-thin-slice](week02-thin-slice/) |
| 3 | The gateway: streaming proxy, Redis rate limiting, backpressure | [week03-gateway](week03-gateway/), [gateway](gateway/) |
| 4 | C++ prompt cleaner (pybind11) inside the gateway | [week04-cpp-cleaner](week04-cpp-cleaner/) |
| 5 | MLflow model registry: promotion gate, rollback, gateway reload | [week05-mlflow-registry](week05-mlflow-registry/) |
| 6 | Blue/Green model servers: traffic split, switch under load with 0 failed requests | [week06-blue-green](week06-blue-green/) |

Write-ups with diagrams: [davissebastian.github.io/ai-infra-platform](https://davissebastian.github.io/ai-infra-platform/).

Setup: one dependency group per week, e.g. `uv sync --group dev --group week05` (Python 3.12, managed by [uv](https://docs.astral.sh/uv/)).
