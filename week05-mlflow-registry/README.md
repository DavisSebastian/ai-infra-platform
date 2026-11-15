# Week 5: a model registry the gateway listens to

An MLflow model registry (backed by Postgres) that decides which settings the gateway serves. A version is scored before it can go live, promotion and rollback are one command each, and the gateway picks up a change without a restart.

Write-up with diagrams: [docs/week05.html](https://davissebastian.github.io/ai-infra-platform/week05.html).

## What a "model version" is here

Retraining is out of scope, and 16 GB of weights don't belong in an artifact store on a laptop. So a version is:

- a **settings card**: `model_id`, `temperature`, `max_tokens`, `system_prompt` ([cards/](cards/)), plus
- a **score sheet**: pass rate on a 10-prompt keyword eval ([eval_prompts.yaml](eval_prompts.yaml)) and time to first token.

Many LLM teams version prompts and configs this way rather than weights. Each card becomes a version of the registered model `gateway-llm`.

| Version | Card | Pass rate | TTFT p50 (M-series Mac, MPS) |
|---|---|---|---|
| v1 | Qwen2.5-0.5B-Instruct, temperature 0.7 | 90% (said 17 &times; 23 = 481) | 69 ms |
| v2 | Qwen2.5-1.5B-Instruct, temperature 0.3 | 100% | 116 ms |

## The pieces

```
register.py ──► MLflow: gateway-llm v1, v2, ... tagged eval=passed|failed
release.py  ──► moves the aliases: champion (served), fallback (last good)
gateway     ──► resolve_champion() at startup and on POST /admin/reload
```

| File | What it does |
|---|---|
| [docker-compose.yml](docker-compose.yml), [mlflow.Dockerfile](mlflow.Dockerfile) | MLflow 3.17.2 + Postgres 17. Records in Postgres, artifacts served by MLflow, both on named volumes. UI on **:5001** |
| [scripts/register.py](scripts/register.py) | Scores a card, logs it as a run, registers a version tagged `eval=passed` or `eval=failed` (threshold 0.8). Never moves `champion` |
| [scripts/release.py](scripts/release.py) | `status`, `promote N`, `rollback --reason ...` |
| [scripts/ci_registry_check.py](scripts/ci_registry_check.py) | The CI check against a real server: seed, restart, verify |
| [`gateway/src/gateway/champion.py`](../gateway/src/gateway/champion.py) | `resolve_champion()` and `apply_card()`, inside the gateway package |

### Rules for moving the aliases

| Command | Result |
|---|---|
| `promote N`, N has `eval=passed` | `champion` &rarr; N, the old champion becomes `fallback` |
| `promote N`, N has `eval=failed` or no tag | refused |
| `rollback` | `champion` &rarr; `fallback`, then `fallback` is **cleared**: the version just left is suspect, so it must not be one command from going live again |
| `rollback` with no `fallback` | refused |

MLflow stores only where an alias points now, not its history, so every move also tags the version: `promoted_at`, `promoted_by`, `rolled_back_at`, `rollback_reason`.

### What the gateway does with the card

| Setting | Rule |
|---|---|
| `model` | always the card's: the registry decides what is served |
| `max_tokens` | capped at the card's |
| `temperature` | the card's, unless the caller sent one |
| system prompt | the card's goes first, unless the caller sent their own |

- **Startup:** the champion is resolved before anything else. Registry on and no usable card means the gateway **refuses to start**.
- **`POST /admin/reload`** (header `X-Admin-Token`): re-reads the champion. On failure it keeps serving the current card, answers 503 with the reason, and `/health` turns `degraded`.
- **`/health`** shows the live version under `registry`.

| Gateway setting | Default | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | unset | unset = registry off: requests are forwarded unchanged, as in Weeks 3-4 |
| `REGISTRY_MODEL` | `gateway-llm` | which registered model to follow |
| `ADMIN_TOKEN` | unset | a secret; unset = `/admin/reload` disabled |
| `MLFLOW_HTTP_REQUEST_MAX_RETRIES` | `0` (set by the gateway) | the client's own default retries a dead server for ~4 minutes |

## Run it

From the repo root:

```bash
(cd week05-mlflow-registry && docker compose up -d --build --wait)    # MLflow UI: http://localhost:5001
uv sync --group dev --group week03 --group week05                     # week05 includes torch, for scoring

cd week05-mlflow-registry
uv run python scripts/register.py cards/v1-qwen-0.5b.yaml             # downloads the model on first run
uv run python scripts/register.py cards/v2-qwen-1.5b.yaml
uv run python scripts/release.py promote 1
uv run python scripts/release.py promote 2
uv run python scripts/release.py status
cd ..

MLFLOW_TRACKING_URI=http://localhost:5001 ADMIN_TOKEN=dev-token \
  uv run uvicorn gateway.main:app --port 8080
curl -s localhost:8080/health | jq .registry

uv run python week05-mlflow-registry/scripts/release.py rollback --reason "trying it out"
curl -s -X POST -H "X-Admin-Token: dev-token" localhost:8080/admin/reload
```

Tests:

```bash
uv run pytest week05-mlflow-registry/tests          # release.py rules, fake registry
uv run pytest gateway/tests                          # includes resolve_champion() and /admin/reload
```

`docker compose down` keeps every run and version (named volumes). `docker compose down -v` deletes them.

## Things that bit, and what they taught

- **The database URL needs the driver name.** `postgresql://` crashes with `No module named 'psycopg'`: SQLAlchemy 2.1 changed its default Postgres driver to psycopg v3. `postgresql+psycopg2://` names the one the image has.
- **MLflow 3 checks the `Host` header.** Calls from another container (Host: `mlflow`) got 403 while `/health` still said 200. Hence `--allowed-hosts`, and a CI check that makes a real registry call instead of trusting `/health`.
- **A dead MLflow costs 247 seconds** with the client's default retries. Capped to 0 in the gateway: startup fails in 2.3 s, a reload in 4 ms.
- **A fake can lie.** `MlflowException(..., error_code="RESOURCE_DOES_NOT_EXIST")` silently becomes `INTERNAL_ERROR`: it wants the enum constant. And a missing alias is `INVALID_PARAMETER_VALUE`, not "not found". CI now checks the real server's codes against what the fakes assume.
- **The client costs 119 MB.** `mlflow-skinny` took the gateway image from 293 MB to 412 MB, for two HTTP calls (get the alias, download one file). Accepted for now: it's the supported client and pinned to the server. A hand-written client for those two REST endpoints would remove it.
- **Two alias moves can't be atomic.** Each command orders its calls so a crash in between is harmless (rollback moves `champion` first, then clears `fallback`), and a test simulates the crash.

## CI/CD

The `test-week05` job in [ci.yml](../.github/workflows/ci.yml) starts this compose stack, registers three cards, checks a failed one is refused, **restarts the stack and checks `champion` survived**, then runs the gateway against it, rolls back and reloads. It installs `week05-registry` (no torch): CI registers cards but doesn't score them.

Deferred to Month 4: promotion as a GitHub Actions workflow with a required approval, once there's an MLflow server CI can reach.
