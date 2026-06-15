# Content Ranking Eval Pipeline

A model-as-judge eval pipeline for validating content ranking signals against real human engagement data.

## What this is

This pipeline fetches live content from HackerNews, scores it using Claude Haiku as a model-as-judge across a grading framework of quality dimensions, then validates those scores against actual HN upvote scores using Spearman rank correlation. The goal is to test whether a given rubric is capturing signal that predicts real human engagement — and to surface where it isn't.

Built as a hands-on prototype of the eval-to-signal feedback loop used by content ranking teams: define a rubric, run the judge, validate against ground truth, identify where the rubric diverges, iterate.

## What I found

**v1** — 4-dimension grading framework (title clarity, specificity, engagement potential, information density). Spearman correlation: **-0.021**. No signal.

The rubric measured writing quality, not community engagement. The clearest case: "Every Frame Perfect" (813 pts, HN rank #1) scored judge rank #45. By quality rubric standards the title is vague — but it's by a trusted community author, a signal the rubric had no visibility into. Qualitative observation → quantitative hypothesis: author identity is a missing dimension.

**v2** — Added `author_reputation` as a 5th dimension using HN karma as a proxy for author standing. Spearman correlation: **-0.099**. Moved the wrong direction.

Karma is a noisy proxy. "Every Frame Perfect" is posted from `tonsky.me` — domain authority carries the reputation signal, not account karma. The hypothesis was right (author identity matters) but the signal was wrong (karma ≠ trust).

## What v3 would tackle

- **Domain authority** — `tonsky.me`, `paulgraham.com` carry community trust that karma doesn't capture
- **Golden set validation** — human-scored sample to validate judge quality independent of engagement signal
- **Audience fit over content quality** — production ranking asks "is this right for *this person* on *this surface*?", not just "is this good?" The rubric currently evaluates content in isolation
- **Topic trend signal** — timely content consistently outperforms quality predictions

## Why this matters

The transition from human review to model-as-judge at scale is the core operational challenge for content ranking teams right now. This pipeline is a small prototype of that workflow: replacing human annotation with an LLM judge, then using correlation against real-world outcomes to validate whether the judge is trustworthy.

The more interesting finding wasn't the correlation number — it was learning to read the disagreements. Where the judge and humans diverge is where the rubric is missing signal. That diagnosis loop is what makes eval work useful.

## Usage

```bash
# Demo mode — no API key required (heuristic scorer)
python content_ranking_eval.py --demo

# Full run with Claude as model-as-judge
ANTHROPIC_API_KEY=your_key_here python content_ranking_eval.py

# Adjust sample size
python content_ranking_eval.py --n 30
```

**Requirements:** `pip install anthropic requests numpy`

## Background

This prototype is modeled on content prioritization work I did at Meta, where I built systems using query rate, AI/agent citation rate, and documentation coverage gaps to decide where content investment would move the needle. The underlying operation is the same: attach structured meaning to a content asset, validate that meaning against a real-world outcome signal, improve the rubric where it diverges.
