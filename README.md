# Content Ranking Eval Pipeline

A model-as-judge eval pipeline for validating content ranking signals against real human engagement data.

## What this is

Fetches live HackerNews stories, scores them using Claude Opus as a model-as-judge across a rubric of content quality dimensions, then validates those scores against actual upvote scores using Spearman rank correlation. The goal: test whether a given rubric captures signal that predicts real engagement, and surface where it doesn't.

Default sample size is 500 stories. Run with `--n 50` for a quick test.

## Results (8 runs)

| Run | Model | N | Correlation | p-value | Notes |
|---|---|---|---|---|---|
| v1 heuristic | — | 50 | -0.245 | — | Rule-based scorer, no API |
| v1 LLM | Opus | 50 | -0.021 | — | 4 dimensions incl. engagement_potential (circular) |
| v2 LLM | Opus | 50 | -0.099 | — | Replaced with topic_novelty, added author karma |
| v2 LLM | Opus | 50 | -0.172 | — | Same day, different stories |
| v3 LLM | Opus | 50 | -0.234 | — | Ground truth updated to composite engagement signal |
| v3 LLM | Opus | 50 | -0.159 | 0.273 | Added title_word_count signal |
| v3 LLM | Haiku | 499 | **-0.113** | **0.012** | First statistically significant result |
| v3 LLM | Haiku | 499 | **-0.104** | **0.022** | Second significant result — finding confirmed |

The final run is the headline: p=0.012 at n=499 confirms the negative correlation is real, not noise. The judge is measurably anti-correlated with human engagement.

Switching from Opus to Haiku produced no meaningful change in the correlation direction or magnitude. This suggests the construct gap is structural — a function of what the rubric measures, not how well the model reasons. More capable models don't close the gap; better signals would.

**Consistent patterns across runs:**

Stories the judge overrates (HN ignores): niche Show HN projects, obscure technical simulators, new low-karma authors with technically precise titles. "Nipkow Disk Mechanical TV Simulator" appeared as an overrated outlier in three consecutive runs — the judge loves niche-but-precise technical titles that HN doesn't engage with.

Stories HN loves (judge underrates): vague-but-resonant titles ("Running local models is good now"), community hero worship ("I admire Fabrice Bellard"), 1.0 releases of known projects ("Iroh 1.0"), breaking news cycles ("Statement on US government directive to suspend access to Fable 5" — #1 by engagement, #363 by judge), and pg essays the judge flags as clickbait ("How to earn a billion dollars" — judge can't see the paulgraham.com domain authority).

Three distinct failure modes identified:
1. **Community resonance** — short titles for projects with cult followings; the judge sees an uninformative title, HN sees a community event
2. **Breaking news cycles** — the judge has no concept of topic momentum; a story that's the 3rd piece of coverage on an active cycle gets penalized for a sparse title
3. **Domain authority** — paulgraham.com attached to a vague title is a quality guarantee to HN; the judge just sees the words

These consistent over- and under-raters form a natural **golden set** — the cases that stress-test the construct mismatch most clearly.

## Iteration history

**v1** — 4 dimensions: title clarity, specificity, engagement_potential, information density. Spearman correlation: **-0.021**.

Problem: `engagement_potential` is a circular predictor. You can't use "how likely is this to get engagement" to predict engagement — that dimension already encodes the answer, making the other dimensions irrelevant.

**v2** — Replaced `engagement_potential` with `topic_novelty` (a genuine input signal, not a restatement of the outcome). Added `author_karma`, `article_age_hours`, and `title_word_count` as data signals from the HN API. Switched to Opus for richer reasoning.

**v3** — Updated ground truth from raw upvote score to composite engagement signal: `upvotes + (comments x 2)`. Comments require active intent; upvotes are passive. Chosen interaction signals are more valuable for ranking than impression-level signals.

## What v4 would tackle

The 6-run evidence points to one clear next lever: the LLM judge has hit its ceiling on text-based signals. More prompt tuning won't close the gap. The missing signal is behavioral.

- **Historical engagement feature** — count how many times an author or domain has landed in the HN top 100 in the last 30 days. A simple API-backed lookup that would capture the "Iroh 1.0" class of misses without requiring the judge to know community context it doesn't have.
- **Domain authority** — `tonsky.me`, `paulgraham.com` carry community trust that account karma doesn't capture. Domain-level historical engagement is a cleaner signal than per-author karma.
- **Golden set validation** — the consistent over- and under-raters identified across 6 runs (Nipkow Disk, Iroh 1.0, Fabrice Bellard) are a natural starting point for a human-scored golden set to validate judge quality independently of the engagement signal.
- **Audience fit** — production ranking asks "is this right for *this person* on *this surface*?", not just "is this good in isolation?" That requires personalization infrastructure that text evals can't replicate.

## Cross-model validation

After the core iteration was complete, the v3 script was submitted to Codex for an independent engineering review. A second model then validated that review before any changes were accepted.

The review confirmed the analytical findings — the biggest methodological concern flagged was that engagement is affected by age, timing, author reputation, and HN's own ranking algorithm, not just content quality. That's the construct validity problem the README already describes.

Engineering improvements identified and implemented in v4 (available on request): scipy Spearman with tie-aware fallback, LLM output validation, retry/backoff, checkpointing for long runs, age-adjusted engagement normalization, author karma removed from judge prompt by default.

The most interesting methodological suggestion came from the review: instead of sampling `topstories` (stories that already won), score `newstories` at T0 and collect engagement 6–24 hours later. That removes the feedback loop — HN's ranking algorithm affects visibility, which affects engagement, which contaminates the ground truth. A T0 design would test whether the judge can predict future engagement from content alone, before the platform's own signal intervenes. That's the v5 direction.

## A dimension that was considered and rejected

A `community_identity` dimension was proposed — scoring whether content would resonate based on community-specific signals like author reputation or in-group references. It was ruled out deliberately.

Community identity is a property of the relationship between content and audience, not a property of the content itself. To score it, the judge would need to know who Fabrice Bellard is, why HN cares about him, and what the current community temperature is around him. None of that is in the title. Asking the judge to estimate it from text alone would produce hallucinated scores that look plausible but have no grounding — making the correlation worse in ways that are hard to diagnose.

The honest version of that dimension requires behavioral data: author historical engagement rates, topic trend graphs, community interaction history. That's not available from public HN text. It's the kind of signal that lives in a platform's social graph — and it's why production ranking systems need behavioral infrastructure that text-based evals can't replicate.

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

Modeled on documentation prioritization work I did at Meta: using query rate, AI/agent citation rate, and documentation coverage gaps to decide which tables to document and where content investment would move the needle. The underlying pattern is the same — use evals to validate that your output quality signal predicts the outcome you care about. There, the outcome was documentation strings that were more reliable and more helpful to agents and engineers. Here, it's surfacing where a text-based quality rubric diverges from what human engagement actually rewards — and why.
