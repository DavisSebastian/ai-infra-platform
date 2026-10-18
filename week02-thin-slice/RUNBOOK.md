# Runbook: run the Week 2 server on a RunPod GPU

Goal: serve **Llama 3.1 8B** from the Docker image CI built, re-run the load test (1, 4 and 16 users), record the
numbers, and **shut the pod down**. Plan for 1 to 2 hours.

> Every RunPod screen name below is from memory and may have changed. Where something matters, there is a
> check you can run to confirm it. If the UI differs from this page, trust the UI and fix this file afterwards.

---

## 0. Before you start (about 10 minutes, no GPU billing yet)

- [ ] **PR #3 is merged**, and the "Docker image" workflow on `main` finished green. Until then no image exists.
- [ ] **The GHCR package is public** (GitHub profile -> Packages -> `ai-infra-platform/week02-thin-slice` ->
      Package settings -> Change visibility). Public is fine: the image has no secrets and no weights.
      If you keep it private, RunPod needs a registry login with a token that has `read:packages`.
- [ ] **You know the image tag.** Use the immutable one, not `latest`:
      `ghcr.io/davissebastian/ai-infra-platform/week02-thin-slice:sha-<full 40-character commit sha of main>`
      (on GitHub: Actions -> the run on `main` -> the commit sha, or `git rev-parse origin/main`).
- [ ] **Llama access is approved** (huggingface.co/meta-llama/Llama-3.1-8B-Instruct shows no "request access"
      banner) and you have a **read** token. Keep the token out of chat, git and screenshots.
      No access yet? Use `Qwen/Qwen2.5-7B-Instruct` instead. It needs no token.
- [ ] **RunPod credit is prepaid.** That is your safety net: the worst case is bounded by your balance.
- [ ] A text file open to **jot down numbers** as you go (section 5 has the table).

---

## 1. Pick the machine

| Need | Why |
|---|---|
| **24 GB of VRAM or more** (RTX 4090, L4, RTX A5000, or a 48 GB card for headroom) | 8B parameters x 2 bytes = about 16 GB of weights, plus working memory |
| **CUDA 13 support** | Our lockfile pins PyTorch libraries built for CUDA 13, which needs a recent NVIDIA driver (roughly 580 or newer). RunPod's deploy page lets you filter by CUDA version. Choose 13.x if offered |
| **Container disk of about 50 GB** | Image (about 10 GB) plus the model download (about 16 GB), with room to spare |

Write down: the GPU model, the **price per hour**, and which cloud tier (Secure or Community).

---

## 2. Configure the pod

Deploy a pod with a **custom container image** (a template). Fill in:

| Field | Value |
|---|---|
| Container image | `ghcr.io/davissebastian/ai-infra-platform/week02-thin-slice:sha-...` |
| Container disk | 50 GB |
| Expose HTTP port | `8000` |
| Start command | **leave empty**: the image's own `CMD` starts uvicorn on `0.0.0.0:8000` |

Environment variables:

| Name | Value | Why |
|---|---|---|
| `MODEL_ID` | `meta-llama/Llama-3.1-8B-Instruct` | Which model to serve |
| `HF_TOKEN` | your token | Lets the container download a gated model. Use RunPod's **Secrets** feature if you can (Settings -> Secrets, then reference it from the variable). Plain values are visible in the pod's config |
| `DEVICE` | `cuda` | **Important.** Without it the server picks "the best available" and silently falls back to the CPU if CUDA isn't usable. An 8B model on a CPU would sit there for ages. With `DEVICE=cuda`, a broken GPU setup crashes at startup instead, which is what you want |
| `MAX_NEW_TOKENS` | leave the default (512) | |

**Why no volume for this run?** The image runs as a non-root user (`uid 1000`) and owns `/models`, so downloads
into the container disk just work. A RunPod volume would be mounted owned by root, and that user probably couldn't
write to it. The cost: if the pod is *restarted*, the 16 GB model downloads again. For one session that is fine.
(Optional later: check `ls -ld /workspace` in a pod, then decide whether to add a volume or change the image.)

Deploy it, and **note the time**.

---

## 3. Watch it start (this is your cold-start measurement)

In the pod's logs, you should see, in order:

1. Image pull (can take minutes: about 10 GB)
2. `loaded meta-llama/Llama-3.1-8B-Instruct on cuda in ...s`  (this includes the 16 GB download)
3. `Uvicorn running on http://0.0.0.0:8000`

Note three times: **deploy clicked -> container started -> server ready**.

Open the pod's web terminal (or SSH). The image is minimal and has **no `curl`**, so inside the pod use Python
(your laptop's `curl` against the pod's public URL works too). Check:

```bash
python -c "import urllib.request as u; print(u.urlopen('http://127.0.0.1:8000/health').read().decode())"
```

**It must say `"device":"cuda"`.** Then:

```bash
nvidia-smi
```

Write down **memory used with the model loaded and idle**. Expect about 16 to 17 GB. (If `nvidia-smi` is missing, use
`python -c "import torch; f,t=torch.cuda.mem_get_info(); print((t-f)/1e9,'GB used')"`.)

Smoke test (prints the answer word by word):

```bash
python - <<'EOF'
import json, urllib.request as u

body = {"messages": [{"role": "user", "content": "Explain what a GPU is in two sentences."}], "max_tokens": 80, "stream": True}
req = u.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"content-type": "application/json"})
with u.urlopen(req) as resp:
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("data:") and not line.endswith("[DONE]"):
            text = json.loads(line[5:])["choices"][0]["delta"].get("content")
            if text:
                print(text, end="", flush=True)
print()
EOF
```

You should see words streaming. Are they noticeably better than the 0.5B model's answers? (They should be.)

---

## 4. Measure

Run the load test **from inside the pod**, not from your laptop. RunPod's public URL goes through a proxy that cuts
connections after about 100 seconds, which a queue of 16 users would hit, and it adds network time to every number.
Inside the pod you measure the server itself.

The image doesn't include the script, so fetch it (after the merge, it is on `main`). The image has no `curl`, so use
Python:

```bash
python -c "import urllib.request as u; u.urlretrieve('https://raw.githubusercontent.com/DavisSebastian/ai-infra-platform/main/week02-thin-slice/scripts/loadtest.py', 'loadtest.py')"
```

Then, one at a time (the first run includes a warm-up request automatically):

```bash
P="Write a long story about a robot."
python loadtest.py /v1/chat/completions --prompt "$P" --max-tokens 100 -n 3  -c 1  --base-url http://127.0.0.1:8000
python loadtest.py /v1/chat/completions --prompt "$P" --max-tokens 100 -n 4  -c 4  --base-url http://127.0.0.1:8000
python loadtest.py /v1/chat/completions --prompt "$P" --max-tokens 100 -n 16 -c 16 --base-url http://127.0.0.1:8000
```

**During the c=16 run**, open a second terminal and run `nvidia-smi` a few times. Write down the **peak memory used**.

**Predict first** (from the Mac numbers): will TTFT at c=16 be longer or shorter than on the Mac, and will ITL be longer or
shorter? Think about it: an 8B model has 16x the weights, but the GPU has far more memory bandwidth than your Mac.

Optional extras if time allows: `--max-tokens 300`, or `-c 8`.

---

## 5. Record the results

Fill this in as you go. It becomes the Week 2 README table.

| | Mac, Qwen 0.5B | RunPod GPU, Llama 3.1 8B |
|---|---|---|
| GPU / price per hour | n/a | ______ / $______ |
| Image pull + model download + ready (cold start) | 111 s (model download only) | ______ |
| VRAM, idle with model loaded | n/a | ______ GB |
| VRAM, peak at c=16 | n/a | ______ GB |
| c=1: TTFT p50 / ITL mean | 32 ms / 12 ms | ______ / ______ |
| c=4: TTFT p50 / max | 2.1 s / 2.2 s | ______ / ______ |
| c=16: TTFT p50 / max | 6.6 s / 11.1 s | ______ / ______ |
| Total output at c=16 (chunks/s) | 80 | ______ |
| Session length and total cost | n/a | ______ min / $______ |

---

## 6. Shut it down (do not skip)

1. In the RunPod dashboard, **Terminate** the pod. ("Stop" pauses the GPU charge but may keep storage billing. For
   a one-off run, terminate.)
2. Reload **Pods** and confirm the list is empty. Check **Billing** that nothing is still running.
3. Delete any volume or disk you created.
4. Note the final cost in the table above.
5. Set a phone reminder for 30 minutes later: "is the RunPod pod really gone?"

---

## If something goes wrong

| Symptom | Likely cause and fix |
|---|---|
| Pod can't pull the image | The GHCR package is still private. Make it public, or add registry credentials in RunPod |
| Container exits at startup with a CUDA error | `DEVICE=cuda` is doing its job: the driver is too old for CUDA 13. Redeploy on a machine that supports CUDA 13 |
| `/health` says `"device":"cpu"` | `DEVICE` wasn't set. Set it to `cuda` and redeploy |
| `401` or `403` downloading the model | Llama access not approved yet, or the token is wrong or not a read token. Switch to `Qwen/Qwen2.5-7B-Instruct` to keep going |
| `CUDA out of memory` | The card has less than about 24 GB, or another process is using it. Check `nvidia-smi` |
| `PermissionError` writing to `/models` | You mounted a volume over it. Remove the volume, or mount it somewhere else and set `HF_HOME` to match |
| Load test times out at c=16 | Raise `--timeout`, or lower `--max-tokens`. The last user waits behind 15 others |
| Worried about the bill | Terminate the pod first, ask questions after |
