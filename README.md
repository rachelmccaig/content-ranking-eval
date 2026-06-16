# Content Ranking Eval Pipeline

A model-as-judge eval pipeline for validating content ranking signals against real human engagement data.

## What this is

Fetches live HackerNews stories, scores them using Claude Opus as a model-as-judge across a rubric of content quality dimensions, then validates those scores against actual upvote scores using Spearman rank correlation. The goal: test whether a given rubric captures signal that predicts real engagement, and surface where it doesn't.

This is a POC with 50 stories. A production version would run 500+.

## Iteration history

**v1** — 4 dimensions: title clarity, specificity, engagement_potential, information density. Spearman correlation: **-0.021**.

Problem: `engagement_potential` is a circular predictor. You can't use "how likely is this to get engagement" to predict engagement — that dimension already encodes the answer, making the other dimensions irrelevant.

**v2** — Replaced `engagement_potential` with `topic_novelty` (a genuine input signal, not a restatement of the outcome). Added `author_karma` and `article_age_hours` as data signals from the HN API. Switched to Opus for richer reasoning.

## What v3 would tackle

- **Domain authority** — `tonsky.me`, `paulgraham.com` carry community trust that account karma doesn't capture
- **Golden set validation** — human-scored sample to validate judge quality independent of the engagement signal
- **Audience fit** — production ranking asks "is this right for *this person* on *this surface*?", not just "is this good in isolation?"
- **Topic trend signal** — whether a topic is currently active on HN would likely improve correlation significantly

## Usage

```bash
# Demo mode (no API key, heuristic scorer)
python content_ranking_eval.py --demo

# Full run with Claude Opus as judge
ANTHROPIC_API_KEY=your_key_here python content_ranking_eval.py

# Larger sample
python content_ranking_eval.py --n 100
```

**Requirements:** `pip install anthropic requests numpy`

## Background

Modeled on content prioritization work I did at Meta: using query rate, AI/agent citation rate, and documentation coverage gaps to decide where content investment would move the needle. The underlying operation is the same — validate that your quality signal predicts the outcome you care about, then improve the rubric where it diverges.
