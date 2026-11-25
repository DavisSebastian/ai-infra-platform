"""Week 7: compare two windows of gateway traffic and say, feature by feature, whether they differ.

    result = compare(reference_window, current_window, embed=minilm())
    result.drifted("output")   # e.g. ["answer_words", "refusal"]

Each record (one request) becomes numbers ("features"), and each feature's two samples are compared with a
statistical test. Evidently picks the test and runs it:

  kind    feature           from              Evidently's test (small samples)   drift when
  input   prompt_words      the prompt        Kolmogorov-Smirnov (K-S)            p < 0.05
  input   prompt_embedding  the prompt        a classifier tries to tell them     AUC beats chance (bootstrap)
                                              apart (ROC AUC)
  output  answer_words      the answer        K-S                                 p < 0.05
  output  answer_empty      the answer        Z-test of proportions               p < 0.05
  output  refusal           the answer        Z-test of proportions               p < 0.05
  output  ttft_ms           the timing        K-S                                 p < 0.05 AND median moves >= 20%

Input drift means the USERS changed (what they ask). Output drift means the ANSWERS changed. Only output drift that
isn't explained by input drift points at the model; deciding that is decide()'s job, not this module's.

Why bootstrap for the embeddings: Evidently's default calls drift when AUC > 0.55. Measured on 200 random splits of
the normal prompts (no drift by construction), that rule raised a false alarm 11% of the time at 150 vs 150 records;
the bootstrap rule (AUC must beat 95% of AUCs from random labels at this sample size) raised one 6% of the time.
Calibrate a detector on your own no-drift data before trusting its defaults.

Why TTFT also needs a minimum effect size: a p-value says a difference is real, not that it matters. TTFT is tightly
clustered, so the same model on the same Mac three days later (median 33.6 -> 34.2 ms, +1.8%) gave K-S p < 0.0001.
Latency depends on the machine as well as the model. A genuinely different model moved it far more (Week 5: 69 vs
116 ms, +68%), so TTFT counts as drifted only when K-S fires AND the median moves by at least 20%.

The embedding function is passed in (dependency injection again): minilm() for real runs, hashing() for CI, which
needs no model download and no torch.
"""

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.legacy.metrics.data_drift.embedding_drift_methods import ModelDriftMethod
from evidently.metrics import EmbeddingsDrift, ValueDrift

Embed = Callable[[list[str]], np.ndarray]

NUMERIC = {"input": ["prompt_words"], "output": ["answer_words", "ttft_ms"]}
YES_NO = {"output": ["answer_empty", "refusal"]}
EMBEDDING = "prompt_embedding"

# How refusals usually start. Crude on purpose: a keyword check catches "the model started refusing", not subtle
# quality loss. Measuring quality properly needs labels or an LLM judge, which is out of scope here.
# Practical significance: a statistically clear but tiny change is not drift. Fraction of the reference median.
MIN_MEDIAN_SHIFT = {"ttft_ms": 0.20}

REFUSAL = re.compile(r"^\s*(i'?m sorry|i am sorry|sorry,|i can'?t|i cannot|i'?m unable|i am unable|as an ai)", re.I)


def features(window: pd.DataFrame) -> pd.DataFrame:
    """One row of numbers per request."""
    answer = window["answer"].fillna("")
    return pd.DataFrame(
        {
            "prompt_words": window["prompt"].fillna("").str.split().str.len().astype(float),
            "answer_words": answer.str.split().str.len().astype(float),
            "ttft_ms": pd.to_numeric(window["ttft_ms"], errors="coerce"),
            "answer_empty": (answer.str.strip() == "").astype(int),
            "refusal": answer.str.match(REFUSAL).astype(int),
        }
    )


def minilm() -> Embed:
    """Real embeddings: all-MiniLM-L6-v2, 384 numbers per sentence. Needs the week07 group (torch)."""
    from sentence_transformers import SentenceTransformer  # imported here: CI runs without torch

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return lambda texts: model.encode(texts, normalize_embeddings=True)


def hashing(dims: int = 64) -> Embed:
    """A stand-in embedding for CI: each word is hashed into one of `dims` buckets (a bag of words).

    It knows nothing about meaning ("capital" and "city" are unrelated to it), but it does move when the vocabulary
    moves, e.g. from geography questions to Python questions. That is enough to test the plumbing without a model.
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(n_features=dims, alternate_sign=False, norm="l2")
    return lambda texts: vectorizer.transform(texts).toarray()


@dataclass(frozen=True)
class Check:
    feature: str
    kind: str  # "input" (what users asked) or "output" (what the model answered)
    method: str  # e.g. "K-S p_value", "Z-test p_value", "model ROC AUC"
    score: float
    drifted: bool


@dataclass(frozen=True)
class DriftResult:
    n_reference: int
    n_current: int
    checks: tuple[Check, ...]

    def drifted(self, kind: str | None = None) -> list[str]:
        return [c.feature for c in self.checks if c.drifted and kind in (None, c.kind)]

    def to_dict(self) -> dict:
        return {"n_reference": self.n_reference, "n_current": self.n_current,
                "checks": [asdict(c) for c in self.checks]}  # fmt: skip


def compare(reference: pd.DataFrame, current: pd.DataFrame, embed: Embed, html: Path | None = None) -> DriftResult:
    """Every check, reference window vs current window. Optionally saves Evidently's HTML report."""
    if len(reference) < 2 or len(current) < 2:
        raise ValueError(f"need at least 2 records per window, got {len(reference)} and {len(current)}")
    ref, cur = features(reference), features(current)
    ref_emb = embed(reference["prompt"].fillna("").tolist())
    cur_emb = embed(current["prompt"].fillna("").tolist())
    emb_cols = [f"e{i}" for i in range(ref_emb.shape[1])]
    ref = pd.concat([ref.reset_index(drop=True), pd.DataFrame(ref_emb, columns=emb_cols)], axis=1)
    cur = pd.concat([cur.reset_index(drop=True), pd.DataFrame(cur_emb, columns=emb_cols)], axis=1)

    numeric = [c for cols in NUMERIC.values() for c in cols]
    yes_no = [c for cols in YES_NO.values() for c in cols]
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=yes_no, embeddings={EMBEDDING: emb_cols})
    method = ModelDriftMethod(bootstrap=True)
    report = Report(
        [ValueDrift(column=c) for c in numeric + yes_no]
        + [EmbeddingsDrift(embeddings_name=EMBEDDING, drift_method=method)]
    )
    snapshot = report.run(Dataset.from_pandas(cur, data_definition=definition),
                          Dataset.from_pandas(ref, data_definition=definition))  # fmt: skip
    if html:
        html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html))

    kind_of = {c: k for group in (NUMERIC, YES_NO) for k, cols in group.items() for c in cols}
    checks = []
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if config["type"].endswith("ValueDrift"):
            name, value, test = config["column"], float(metric["value"]), config["method"]
            # A p-value is "how likely is a difference this big by chance": small means drift. A distance (used for
            # large samples) is "how different": big means drift.
            drifted = value < config["threshold"] if "p_value" in test else value >= config["threshold"]
            if name in MIN_MEDIAN_SHIFT:
                before, after = ref[name].median(), cur[name].median()
                shift = abs(after - before) / before if before else float("inf")
                drifted = drifted and shift >= MIN_MEDIAN_SHIFT[name]
                test += f", median {shift:+.0%}" if before else ""
            checks.append(Check(name, kind_of[name], test, round(value, 4), bool(drifted)))
    # The report gives the embeddings' AUC but not the bootstrap's yes/no, so ask the method directly. It is
    # deterministic (fixed random seeds), so it computes the same AUC the report shows.
    auc, drifted, _ = method(pd.DataFrame(cur_emb), pd.DataFrame(ref_emb))
    checks.insert(1, Check(EMBEDDING, "input", "model ROC AUC", round(float(auc), 4), bool(drifted)))
    return DriftResult(len(reference), len(current), tuple(checks))
