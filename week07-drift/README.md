# Week 7: catching a model that fails silently

A worse model throws no exception. Every request still returns HTTP 200, latency looks normal, and dashboards stay green; the answers are just different. This week the gateway records what it serves, and a drift check compares recent traffic with a known-good baseline, statistically, to answer three questions: has anything changed, did the **users** change or the **model**, and is it worth acting on?

Write-up with diagrams: [docs/week07.html](https://davissebastian.github.io/ai-infra-platform/week07.html).

## Result: a card that passed the eval, caught on live traffic

Card **v3** is v1 with one well-meant change to the system prompt: *"Reply in at most five words."* The Week 5 keyword eval scored it **100%**, better than v1's 90%, so the release gate let it through. On live traffic its answers shrank from a median of 18 words to 6.

Blue/Green, 50/50, both sides on the same Qwen2.5-0.5B server (Week 2's, on an M-series Mac), so the only thing that can differ is the card:

| Scenario | What changed | Verdict |
|---|---|---|
| **A′** canary, nothing changed | both sides on v1, everyday questions | **OK** |
| **B** the users changed | coding questions, sent to both sides | **SHIFT**: blue shows it too, so not green's model |
| **C** silent failure | green on card v3, everyday questions | **ALERT**: answer length differs from blue *and* from the baseline |

How many requests green must serve before the alert holds (200 random windows per size, from scenario C; false alarms from scenario A′):

| Green requests | Alert raised (card v3) | False alarm (no change) | Total requests at a 10% canary |
|---|---|---|---|
| 5 | 24% | 0% | 50 |
| 10 | 66% | 0% | 100 |
| 15 | **94%** | 0% | 150 |
| 20 | **100%** | 0% | 200 |

A change this large is clear after about 20 requests. `decide()` waits for **50** before judging anything, so that minimum, not the statistics, sets the detection time here. It is deliberately cautious: a subtler degradation needs more data. Raw results: [results/](results/).

## How it works

```
gateway (DRIFT_LOG=1) ──► Redis stream: one record per request (side, card, prompt, answer, timings)
drift_check.py          ──► features per record ──► Evidently: reference window vs current window
                        ──► decide(): insufficient_data / ok / shift / warn / alert   (Week 8 rolls back on alert)
```

**1. Record, don't compute.** The gateway puts each finished request on an in-memory queue; a background task writes it to a Redis stream. Requests never wait on it, and a Redis outage is counted, not felt (fail open). Measured: about +1 ms at p50. Code: [`gateway/src/gateway/request_log.py`](../gateway/src/gateway/request_log.py).

**2. Features: text becomes numbers.**

| Kind | Feature | Evidently's test | Drift when |
|---|---|---|---|
| input | `prompt_words` | Kolmogorov–Smirnov (K-S) | p < 0.05 |
| input | `prompt_embedding` (all-MiniLM-L6-v2, 384 numbers) | a classifier tries to tell the windows apart (ROC AUC) | AUC beats chance (bootstrap) |
| output | `answer_words` | K-S | p < 0.05 |
| output | `ttft_ms` | K-S | p < 0.05 **and** median moves ≥ 20% |
| output | `answer_empty`, `refusal` | Z-test of proportions | p < 0.05 |

**3. Blue is the control group.** During a canary the split is random per request, so blue and green serve the same users at the same time. Green differing from **blue** points at green's model; drift that blue shows too is the users (or the environment). Without that control, scenario B's longer answers would look like a model problem.

**4. Two agreeing signals before an alert.** Each check fires by chance about 1 time in 20. With four output checks, *any one* firing happens about 1 run in 5; *two or more*, about 1 in 71. So an alert needs two output checks, or one check confirmed by both comparisons (vs blue and vs the baseline). Input drift alone is only a warning, never a rollback. Code: [scripts/verdict.py](scripts/verdict.py).

**5. A shared shift is its own verdict.** If green differs from the baseline but blue shows the same difference, the verdict is **shift**: the users or the environment changed, and green is as good as blue. Week 8 found why this matters: under 2 requests/s the model server queues, TTFT rose on *both* sides versus a baseline captured one request at a time, and as a plain warning that blocked every canary promotion. Monitoring asks "is the world different from last week?"; a canary asks "is green as good as blue right now?".

## Things that bit, and what they taught

- **Time to first token measured 2 ms.** The Week 2 server sends a role-only event the moment it starts, before generating anything, so timing the first chunk measured time to first *byte*, flat however slow the model. The gateway now records chunk arrival times and the writer finds the first chunk that carries text: 34 ms.
- **Evidently's default embedding threshold false-alarms 11% of the time here.** Its rule is AUC > 0.55. On 200 random splits of the everyday prompts (no drift by construction, 150 vs 150) that fired 22 times. The bootstrap rule (beat 95% of AUCs from random labels at this sample size) fired 11 times, 6%. Calibrate a detector on your own no-drift data before trusting its defaults.
- **Repeated prompts faked drift.** Sending each prompt twice put the same prompt in both the classifier's training and test halves, so it memorised them: AUC 0.51 → 0.68 with nothing changed. Each prompt is now sent once.
- **Statistically significant, practically meaningless.** TTFT moved 0.6 ms (+1.8%) in three days and K-S gave p < 0.0001, because the values are tightly clustered. Latency depends on the machine too. TTFT now also needs a ≥ 20% median shift; a different model moved it 68% in Week 5.
- **A crude embedding false-alarms where a semantic one doesn't.** The CI stand-in (words hashed into buckets) said "drift" (AUC 0.59) on holdout vs baseline, where MiniLM said 0.506: the holdout asks the same kinds of things in different words.
- **Evidently 0.7.23 doesn't import with pandas 3.** It uses a pandas internal that pandas 3 removed. `pyproject.toml` pins `pandas<3` with the reason next to it.

## Run it

From the repo root, with the MLflow stack, Redis and the Week 2 model server running (setup in the [Week 5](../week05-mlflow-registry/README.md) and [Week 2](../week02-thin-slice/README.md) READMEs):

```bash
uv sync --group dev --group week02 --group week03-serve --group week05-registry --group week07

# both sides on one model server; only the card differs
REGISTRY_MODEL=gateway-llm uv run python week05-mlflow-registry/scripts/release.py deploy 1 --to blue
REGISTRY_MODEL=gateway-llm uv run python week05-mlflow-registry/scripts/release.py deploy 1 --to green
UPSTREAM_URL=http://localhost:8000 GREEN_URL=http://localhost:8000 TRAFFIC_GREEN=50 \
  MLFLOW_TRACKING_URI=http://localhost:5001 ADMIN_TOKEN=dev-token DRIFT_LOG=1 RATE_LIMIT_ENABLED=0 \
  uv run uvicorn gateway.main:app --port 8080

# capture a window, then judge green against the baseline and against blue
uv run python week07-drift/scripts/capture.py week07-drift/prompts/normal.txt --split holdout --out week07-drift/data/my-window.parquet
uv run python week07-drift/scripts/drift_check.py --current week07-drift/data/my-window.parquet --side green --against-side blue --html week07-drift/results/my-window.html

# or straight from the live stream
uv run python week07-drift/scripts/drift_check.py --stream-last 150 --side green --against-side blue

# the detection table, from the saved scenarios
uv run python week07-drift/scripts/detection_curve.py
```

Tests: `uv run pytest week07-drift/tests` (29: the checks and the verdict rules) and `uv run pytest gateway/tests` (includes the request log).

| Gateway setting | Default | Meaning |
|---|---|---|
| `DRIFT_LOG` | `0` | `1` records every forwarded request to Redis. Off by default: records hold prompt and answer text |
| `DRIFT_STREAM` | `gateway:requests` | the Redis stream |
| `DRIFT_STREAM_MAXLEN` | `10000` | newest records kept |

## Files

| Path | What |
|---|---|
| [prompts/normal.txt](prompts/normal.txt), [prompts/coding.txt](prompts/coding.txt) | 300 everyday questions (split 150/150 into baseline and holdout), 150 coding questions |
| [data/](data/) | the windows as the gateway logged them: baseline, holdout, scenarios A′, B, C |
| [scripts/capture.py](scripts/capture.py) | send prompts through the gateway, save what it logged |
| [scripts/drift.py](scripts/drift.py), [scripts/drift_check.py](scripts/drift_check.py) | the checks, and the command-line tool |
| [scripts/verdict.py](scripts/verdict.py) | `decide()` |
| [scripts/detection_curve.py](scripts/detection_curve.py) | the requests-until-detection table |

## CI/CD

The `test-week07` job runs the 29 tests on the committed baseline with the stand-in embedding: the `week07-drift` dependency group, no torch and no model download. Each test changes one thing about the real baseline window (questions, answer length, refusals, empty answers, latency) and checks that only the matching check notices, and that unchanged traffic raises nothing.

## Limits

- **Drift is not quality.** It says "different", not "worse". The refusal and empty-answer checks are crude keyword proxies; measuring quality needs labels or an LLM judge.
- **Small samples.** One request at a time on a laptop gives windows of about 75 per side. Production traffic gives far more, which detects smaller changes and needs the distance-based tests Evidently switches to above ~1000 rows.
- **The baseline ages.** It should be refreshed deliberately, for example when a new champion is promoted.
- **Privacy.** Records keep prompt and answer text. Fine for this project with the log off by default; a product would need consent and a retention policy.

Next, Week 8: act on the verdict. An alert sets green's traffic to 0 automatically.
