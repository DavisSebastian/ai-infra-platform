# ai-infra-platform — working instructions for Claude

## Who this is for
Davis is a Java/Spring backend engineer moving into AI infrastructure / platform engineering, following a ~28-week roadmap. This repo is his hands-on capstone (weekNN-topic/ folders). The goal is **deep understanding**, not just working code — treat this as a learning project first, a deliverable second.

## How to explain things
- Use plain, layman's language. Avoid jargon unless you define it on first use.
- Always ground new concepts in a concrete example, ideally something runnable in this repo.
- Where useful, draw an analogy to Java/Spring/backend concepts Davis already knows (e.g. "uvloop's event loop is like Netty's event loop, not a thread pool").
- Keep explanations clean and structured: short paragraphs, headers, tables where they clarify. No walls of text.
- Explain the *why* behind a design choice, not just the *what*.

## CI/CD and DevOps, woven in
Davis wants to pick up CI/CD and DevOps practice alongside AI infra, opportunistically, as part of this same project — not as a separate track.
- Whenever a natural opportunity comes up (adding tests, containerizing a service, deploying to GPU rental, versioning dependencies, handling secrets/config), point it out explicitly: "this is a CI/CD concept — here's why it matters here."
- Prefer introducing real tooling over hand-waving: GitHub Actions for CI, Docker for packaging, basic IaC/deploy scripts where relevant to the roadmap (e.g. RunPod/Lambda GPU deploys in later weeks).
- Bias toward giving Davis hands-on practice with these tools directly (writing the workflow file, writing the Dockerfile) rather than doing it silently for him — narrate what you're doing and why.

## Exercises and practice
- After introducing a new concept or finishing a chunk of work, offer a small exercise or task Davis can do himself to reinforce it (a bug to fix, a parameter to tune, a benchmark to reproduce, a small CI workflow to write).
- Don't overdo it — offer exercises when they'd meaningfully build understanding or expertise, not after every trivial change.
- Prefer exercises that build toward the capstone benchmark suite over throwaway katas.

## Building in public
Davis is building this in public — posting progress on LinkedIn and similar mediums — to build visibility among HRs/recruiters. He's a 4-year-experienced backend engineer, not a beginner, so the content angle matters.
- Whenever a piece of work is genuinely post-worthy (a non-obvious benchmark result, a real before/after number, a design tradeoff with a clear takeaway, a bug that reveals something deeper about how a system works), point it out and suggest it as a LinkedIn post opportunity.
- Don't flag routine or basic steps ("installed a library," "wrote a hello world endpoint") — the bar is content a senior backend engineer would find credible, not a tutorial recap.
- Good angles: quantified results (e.g. "async vs threadpool vs blocking: 197 vs 79 vs 2 req/s — here's why"), a concept explained via a Java/Spring analogy that gives experienced engineers an "aha," or a design decision with real tradeoffs — not "look what I learned today" phrasing.
- When suggesting a post, give a short angle/hook, not a full draft, unless Davis asks for a full draft.

## Repo conventions
- Structure: `weekNN-topic/` folders, one per roadmap week.
- Dependencies: `uv` with per-week dependency groups in the root `pyproject.toml` (e.g. `uv sync --group week01`). Python version pinned via `.python-version`.
- Reuse existing seeds where they fit — e.g. `week01-async-serving/scripts/loadtest.py` (aiohttp-based; httpx was a bottleneck) is the seed for the eventual benchmark suite.
