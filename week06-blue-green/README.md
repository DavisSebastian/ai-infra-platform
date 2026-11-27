# Week 6: Blue/Green model servers behind the gateway

Two model servers run side by side: **blue** (the version serving now) and **green** (the new one). The gateway moves traffic between them with one admin call, without a restart, and without cutting off a request that is already streaming.

The code lives in the gateway package: [`router.py`](../gateway/src/gateway/router.py) (the split) and [`main.py`](../gateway/src/gateway/main.py) (two sides, `POST /admin/traffic`). This folder holds the measurement.

## Result: switching under load

50 streaming requests per second for 10 s, switching from blue to green at 4 s. Fake model servers (20 chunks, 20 ms apart, so a stream takes 400 ms), everything on one M-series Mac. 3 runs per mode, all within 2 failures of each other.

| How the switch is done | Failed requests | Users affected for | p99 of successful streams |
|---|---|---|---|
| **Naive**: change `UPSTREAM_URL`, restart the gateway | **78–80 of 500 (16%)**, all "connection refused" | **1.55 s** | 424 ms |
| **Blue/Green**: `POST /admin/traffic {"green": 100}` | **0 of 500** | 0 s | 424 ms |
| **Canary**: green 10% → 50% → 100%, 2 s apart | **0 of 500** | 0 s | 425 ms |

Raw numbers: [results/mac-fakes.jsonl](results/mac-fakes.jsonl).

### What the numbers say

- **The naive failures are all new requests arriving while nothing is listening.** None of the in-flight streams were cut: uvicorn's SIGTERM handling stops accepting connections but lets running requests finish. The 1.55 s is the old process draining (0.5 s) plus the new one starting (1.1 s).
- **This is naive at its best.** Only the gateway restarted; both model servers stayed up. A real "stop v1, start v2" deploy restarts the **model server**, and loading weights takes much longer than starting a Python process, so the failure window grows with it. Blue/Green stays at zero because green is started and warmed up *before* it gets any traffic.
- **The switch adds no latency.** p99 is the same in all three modes (about 24 ms of gateway overhead on top of the 400 ms stream).

### Checking the tool before trusting it

Two answers can be worked out by hand, and the tool reproduces both:

- Blue/Green at 4 s, 50 req/s: exactly 200 requests to blue and 300 to green. It measured 200 / 300.
- Canary: green should get 2 s × 50 × (10% + 50% + 100%) = 160. It measured 158 (the split is random per request).

## Run it

From the repo root (needs the gateway package: `uv sync --group dev --group week03-serve`):

```bash
uv run python week06-blue-green/scripts/switch_bench.py --mode naive
uv run python week06-blue-green/scripts/switch_bench.py --mode bluegreen
uv run python week06-blue-green/scripts/switch_bench.py --mode canary --step 2
```

It starts its own two fake servers and gateway on free ports and stops them afterwards. `--rate`, `--duration` and `--switch-at` change the load; `--json` prints one summary line.

The load is **open-loop**: a request starts every 1/rate seconds whether or not earlier ones finished, like independent users. A closed loop (each worker waits for its last reply) would send fewer requests during an outage and hide the failures.

## How a release works now

```
release.py deploy 2 --to green       green's alias -> v2 (only eval=passed versions)
POST /admin/reload                   gateway loads green's card; green still has 0% of traffic
POST /admin/traffic {"green": 10}    canary; watch /health
POST /admin/traffic {"green": 100}   all new requests on v2; blue drains (traffic.blue.in_flight -> 0)
release.py promote 2                 record v2 as champion
```

Rollback at any point: `POST /admin/traffic {"green": 0}`. It's instant, with no restart and no registry change.

| Gateway setting | Default | Meaning |
|---|---|---|
| `UPSTREAM_URL` | `http://localhost:9000` | the blue model server |
| `GREEN_URL` | unset | the green model server; unset = one server, as in Weeks 3–5 |
| `TRAFFIC_GREEN` | `0` | starting percent of requests sent to green; a bad value stops startup |
| `ADMIN_TOKEN` | unset | needed for `/admin/traffic` and `/admin/reload` |
