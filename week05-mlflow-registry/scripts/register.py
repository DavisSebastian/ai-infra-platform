"""Week 5, step 2: turn a settings card into a registered, scored model version.

    cd week05-mlflow-registry
    uv run --group week05 python scripts/register.py cards/v1-qwen-0.5b.yaml

One call does four things:
  1. read the card            which model, which settings (the "settings card")
  2. score it                 run the eval prompts with those settings: pass rate + time to first token
  3. log an MLflow run        params = the card, metrics = the scores, artifact = serving_config.yaml
  4. register a version       `gateway-llm` vN pointing at that artifact, tagged eval=passed|failed

Nothing here moves the `champion` alias. Registering and promoting are separate steps on purpose,
like `mvn deploy` (publish the artifact) vs actually rolling it out.

torch and transformers are imported inside the scoring functions only: CI imports log_and_register() (steps 3-4)
with no model libraries installed (ci_registry_check.py).
"""

import argparse
import os
import re
import statistics
import threading
import time
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = os.getenv("REGISTRY_MODEL", "gateway-llm")  # same variable the gateway reads
EXPERIMENT = "gateway-llm-cards"
EVAL_FILE = Path(__file__).resolve().parent.parent / "eval_prompts.yaml"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ask(model, tokenizer, card: dict, prompt: str) -> tuple[str, float]:
    """One prompt with the card's settings. Returns (answer, ms until the first piece of text)."""
    from transformers import TextIteratorStreamer

    messages = [{"role": "system", "content": card["system_prompt"]}, {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # temperature 0 means "always pick the most likely token" (greedy), which transformers spells do_sample=False.
    if card["temperature"] > 0:
        sampling = {"do_sample": True, "temperature": card["temperature"]}
    else:
        sampling = {"do_sample": False}

    # generate() blocks until the whole answer is done, so it runs on a thread while this one
    # reads the streamer. That's the only way to see *when* the first token arrived.
    t0 = time.perf_counter()
    worker = threading.Thread(
        target=model.generate,
        kwargs={**inputs, "max_new_tokens": card["max_tokens"], "streamer": streamer, **sampling},
    )
    worker.start()
    ttft_ms, parts = None, []
    for piece in streamer:
        if piece and ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000
        parts.append(piece)
    worker.join()
    return "".join(parts).strip(), ttft_ms or 0.0


def passes(answer: str, keywords: list[str]) -> bool:
    # Whole words only, so the keyword "au" matches "Au" but not "because".
    return any(re.search(rf"\b{re.escape(k)}\b", answer, re.IGNORECASE) for k in keywords)


def score(card: dict) -> tuple[dict, list[dict]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    cases = yaml.safe_load(EVAL_FILE.read_text())
    device = pick_device()
    print(f"loading {card['model_id']} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(card["model_id"])
    model = AutoModelForCausalLM.from_pretrained(card["model_id"], dtype=torch.bfloat16).to(device)

    set_seed(0)  # temperature > 0 means random sampling; a fixed seed makes reruns comparable
    ask(model, tokenizer, card, "Hello")  # warm-up: the first call pays one-off GPU setup costs, don't time it

    results = []
    for case in cases:
        answer, ttft_ms = ask(model, tokenizer, card, case["prompt"])
        ok = passes(answer, case["keywords"])
        results.append({**case, "answer": answer, "passed": ok, "ttft_ms": round(ttft_ms, 1)})
        print(f"  {'PASS' if ok else 'FAIL'}  {ttft_ms:6.0f} ms  {case['prompt'][:45]:45}  -> {answer[:60]!r}")

    metrics = {
        "pass_rate": sum(r["passed"] for r in results) / len(results),
        "ttft_p50_ms": statistics.median(r["ttft_ms"] for r in results),
    }
    return metrics, results


def ensure_registered_model(client: MlflowClient) -> None:
    # The registered model is the "folder"; versions go inside it. Create it once, then reuse.
    try:
        client.create_registered_model(MODEL_NAME, description="Settings cards the gateway can serve.")
    except MlflowException as e:
        if e.error_code != "RESOURCE_ALREADY_EXISTS":
            raise


def log_and_register(
    card: dict, card_file: str, metrics: dict, results: list[dict], verdict: str, extra_params: dict | None = None
):
    """Steps 3-4: log one run for the card, register it as a new version. Returns (version, run)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    with mlflow.start_run(run_name=Path(card_file).stem) as run:
        mlflow.log_params({**card, **(extra_params or {})})
        mlflow.log_metrics(metrics)
        # The card itself, under a fixed name, so whoever loads "the champion" always finds the same file.
        mlflow.log_text(yaml.safe_dump(card, sort_keys=False), "card/serving_config.yaml")
        # Every question, answer and timing: when a version fails, this is where you look to see why.
        mlflow.log_dict(results, "eval/results.json")

    ensure_registered_model(client)
    version = client.create_model_version(
        name=MODEL_NAME,
        source=f"{run.info.artifact_uri}/card",  # the version points at the card folder in that run
        run_id=run.info.run_id,
        tags={"eval": verdict, "card_file": card_file},
        description=f"{card['model_id']} @ temperature {card['temperature']}",
    )
    return version, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("card", type=Path, help="path to a settings card (YAML)")
    parser.add_argument("--threshold", type=float, default=0.8, help="minimum pass_rate to tag eval=passed")
    args = parser.parse_args()

    card = yaml.safe_load(args.card.read_text())
    metrics, results = score(card)
    verdict = "passed" if metrics["pass_rate"] >= args.threshold else "failed"
    version, run = log_and_register(
        card, args.card.name, metrics, results, verdict, extra_params={"eval_threshold": args.threshold}
    )

    print(
        f"\n{MODEL_NAME} v{version.version}: pass_rate={metrics['pass_rate']:.0%} "
        f"ttft_p50={metrics['ttft_p50_ms']:.0f}ms  eval={verdict} (threshold {args.threshold:.0%})"
    )
    print(f"run: {TRACKING_URI}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")


if __name__ == "__main__":
    main()
