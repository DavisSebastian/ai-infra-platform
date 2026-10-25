# Runbook: the gateway in front of Llama 3.1 8B on a RunPod GPU

Goal: measure the Week 2 model server **bare vs behind the gateway** on a real GPU, record the numbers, and **shut the pod down**. This turns the README's prediction ("Week 2's 56 s worst wait becomes ~8–11 s") into a result.

Budget: **about 1 hour, under $1.50** at ~$0.90/hour for an RTX 4090.

> RunPod screen names are from memory and may have changed. Where something matters there is a check to run. If the UI differs, trust the UI and fix this file afterwards.

## How this differs from Week 2's run

| | Week 2 | This run |
|---|---|---|
| Pod image | Our CI-built `week02-thin-slice` image | RunPod's **PyTorch template**, then `git clone` + `uv sync` |
| Processes | Model server | Model server **and** gateway, in one container |
| Load test from | The laptop, through RunPod's proxy (network time included) | **Inside the pod** (no proxy, no network time) |
| Ports exposed | 8000 | **None needed**: everything talks over `127.0.0.1` |

**Why not reuse the Week 2 image?** It can't run the gateway: it lacks the `redis` and `httpx` packages, its virtualenv is owned by root while it runs as a normal user (so nothing can be installed), and it has no `git`, `curl` or `uv`. A combined image would mean a new 9 GB build for one measurement. Cloning the repo and installing from `uv.lock` gives exactly the same pinned versions with no new image.

**Why no Redis?** Rate limiting is switched off for this measurement (`RATE_LIMIT_ENABLED=0`), as on the Mac, so only backpressure decides. The gateway reports `"degraded"` in `/health` because Redis is absent; that is expected and changes nothing here.

**Note on comparing with Week 2:** Week 2's 56 s was measured through the proxy. This run measures the bare server again, inside the pod, so the bare-vs-gateway comparison is fair within this run. Quote this run's bare number next to its gateway number, not Week 2's.

---

## 0. Before you start (no GPU billing yet)

- [ ] **Week 3 is on `main`** (PR #5 merged). The pod clones `main`.
- [ ] **Llama access** still approved, and a **read** token ready (the rotated one). Keep it out of chat, git and screenshots. Without access: use `Qwen/Qwen2.5-7B-Instruct`, which needs no token.
- [ ] **RunPod balance** checked (was $9.46). Prepaid credit is the safety net.
- [ ] **Predict first** and write it down (section 4 has the table):
  - Bare, 16 users: worst wait for the first word? (Week 2, through the proxy: 56 s.)
  - Gateway, 16 users: worst wait? (One answer of 100 tokens took ~3.7 s. The last accepted user waits for two answers, then its own first word.)
  - Gateway, 16 users: how many served, how many 503?

---

## 1. Pick the machine

| Need | Why |
|---|---|
| **24 GB VRAM or more** (RTX 4090, L4, A5000, or a 48 GB card) | 8B parameters × 2 bytes ≈ 16 GB of weights, plus working memory |
| **CUDA 13 support** (driver ~580 or newer) | `uv.lock` pins PyTorch 2.14 built for CUDA 13.0. Use the CUDA-version filter on the deploy page. **Two Week 2 hosts failed on this.** |
| **Container disk ~60 GB** | Python env with torch (~6 GB) + model (~16 GB) + room |

Write down: GPU model, price per hour, Secure or Community cloud.

---

## 2. Configure and start the pod

Deploy with RunPod's official **PyTorch** template (any recent 2.x).

| Field | Value |
|---|---|
| Template | RunPod PyTorch 2.x |
| Container disk | 60 GB |
| Volume disk | 0, or leave the default (we don't use `/workspace`) |
| Exposed ports | none needed (the template's own SSH/Jupyter ports are fine to leave) |
| Environment variable | `HF_TOKEN` = your token, ideally from RunPod **Secrets** (plain values are visible in the pod config) |

Deploy, and **note the time**. Open the pod's **web terminal** (or SSH).

> The template's own PyTorch is ignored: `uv` builds a separate environment from `uv.lock`, with its own Python 3.12 and CUDA 13 libraries. Only the machine's **driver** has to support CUDA 13.

---

## 3. Set up inside the pod

Check the GPU and driver first:

```bash
nvidia-smi
```

The top line shows `CUDA Version: 13.x`? If it says 12.x, **stop here, terminate, pick another host** (that was Week 2's failure).

Install `uv` (pinned to the version the Dockerfiles use), clone, install:

```bash
curl -LsSf https://astral.sh/uv/0.12.17/install.sh | sh
source $HOME/.local/bin/env
cd ~ && git clone --depth 1 https://github.com/DavisSebastian/ai-infra-platform.git
cd ~/ai-infra-platform
uv sync --locked --group week02 --group week03
```

`--locked` fails if `uv.lock` doesn't match, so the pod gets exactly the versions CI tested. Expect a few minutes (torch and the CUDA libraries are several GB).

Start the **model server** in the background (`nohup` keeps it alive if the web terminal disconnects):

```bash
cd ~/ai-infra-platform/week02-thin-slice
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct DEVICE=cuda HF_HOME=$HOME/hf \
  nohup uv run uvicorn app.main:app --port 8001 > ~/model.log 2>&1 &
tail -f ~/model.log
```

Wait for `loaded meta-llama/Llama-3.1-8B-Instruct on cuda in ...s` (this includes the 16 GB download), then **Ctrl-C** (stops `tail`, not the server). Note the load time.

Start the **gateway** in front of it:

```bash
cd ~/ai-infra-platform/week03-gateway
UPSTREAM_URL=http://127.0.0.1:8001 RATE_LIMIT_ENABLED=0 \
  nohup uv run uvicorn app.main:app --port 8080 > ~/gateway.log 2>&1 &
sleep 3
curl -s localhost:8001/health; echo
curl -s localhost:8080/health; echo
```

Check: the model server says `"device":"cuda"`, the gateway says `"in_flight":0,"queued":0` (and `"redis":"down"`, which is expected).

Smoke test through the gateway (words should stream):

```bash
curl -N localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain a GPU in two sentences."}],"max_tokens":60,"stream":true}'
```

Note VRAM with the model loaded and idle: `nvidia-smi` (expect ~16 GB).

---

## 4. Measure

One script runs the whole matrix: 1, 4, 16 and 64 users, bare then gateway, plus two repeats at 16 users. It writes one labelled JSON line per run. It was rehearsed on the Mac before this runbook was written.

Run it in the background so a terminal disconnect can't kill it, then watch:

```bash
cd ~/ai-infra-platform/week03-gateway
OUT=results/llama8b-4090.jsonl LABEL="Llama 3.1 8B, <GPU model>, RunPod, measured in-pod" \
  nohup ./scripts/bench_matrix.sh > ~/bench.log 2>&1 &
tail -f ~/bench.log
```

Expect **15–20 minutes**, nearly all of it the bare server at 64 users (the last user waits behind 63 answers). The gateway runs take seconds each.

**During a bare c=16 run**, run `nvidia-smi` in a second terminal and note peak VRAM.

Fill in as it goes:

| | Prediction | Result |
|---|---|---|
| Bare, c=16: worst wait (3 runs) | | ______ / ______ / ______ |
| Gateway, c=16: worst wait (3 runs) | | ______ / ______ / ______ |
| Gateway, c=16: served / 503 | | ______ |
| Bare, c=64: worst wait | | ______ |
| Gateway, c=64: worst wait / time to reject | | ______ |
| Any `queue_timeout` 503s? (`curl localhost:8080/health`) | | ______ |
| VRAM idle / peak | | ______ |
| Model download + load time | | ______ |

**One thing to watch:** if answers take longer than ~5 s here, the second queued request waits more than the 10 s `QUEUE_TIMEOUT` and gets a 503 `queue_timeout` instead of an answer. That is the gateway working as configured, not a failure, and it is worth recording.

---

## 5. Bring the results home

```bash
cat ~/ai-infra-platform/week03-gateway/results/llama8b-4090.jsonl
grep "loaded" ~/model.log
```

Copy both outputs into chat (they contain no secrets). They go into `week03-gateway/results/` and the README. Don't `git push` from the pod: it would need a GitHub token there.

---

## 6. Shut it down (do not skip)

1. **Terminate** the pod (not just Stop).
2. Reload **Pods**: the list must be empty. Check **Billing**: nothing running.
3. Note the final balance and the session cost.
4. Phone reminder for 30 minutes later: "is the RunPod pod really gone?"

---

## If something goes wrong

| Symptom | Likely cause and fix |
|---|---|
| `nvidia-smi` shows `CUDA Version: 12.x` | Driver too old for the CUDA 13 wheels. Terminate, filter for CUDA 13, redeploy |
| Model server crashes at start with a CUDA error | Same as above: `DEVICE=cuda` turns a bad driver into a loud crash on purpose |
| `uv sync --locked` fails | `uv.lock` doesn't match `pyproject.toml` on `main`. Check that CI on `main` is green |
| `401` / `403` downloading the model | Token missing, wrong, or not approved. `echo ${HF_TOKEN:+set}` should print `set` (it never prints the token). Or switch to `Qwen/Qwen2.5-7B-Instruct` |
| `No space left on device` | Container disk too small. Terminate and redeploy with 60 GB+ |
| `CUDA out of memory` | Card under ~24 GB, or a second model server is running: `pgrep -fa uvicorn` |
| `address already in use` | A server from an earlier attempt is still running: `pkill -f uvicorn`, then start again |
| The script stopped when the terminal closed | It was started without `nohup`. Restart it with the `nohup` line; the results file keeps earlier lines |
| Load test times out | The bare c=64 run is the slowest. The script already allows 900 s per run. Lower `C_LIST` to `"1 4 16"` if time or money is short |
| Worried about the bill | Terminate first, ask questions after |
