# Content Ranking Eval Pipeline

An AI evaluation prototype for testing whether a model-as-judge quality rubric predicts real engagement on a ranked content surface.

This project uses Hacker News because it provides public content, engagement data, and an existing ranking surface. The goal is not to recreate the HN ranker or build a production recommendation system. The goal is narrower: define a content-quality rubric, score stories with an LLM judge, compare those scores against observed engagement, and diagnose where the rubric diverges from what people actually respond to.

<p align="center">
  <img src="images/methodology.svg" alt="Methodology diagram showing Hacker News stories flowing through an LLM judge, quality rubric, composite score, rank correlation, and failure analysis." width="720">
</p>

## Key Finding

Across 8 runs, the conclusion stayed stable: a text-quality rubric did not predict Hacker News engagement. In the two larger 499-story runs, the correlation was small, negative, and statistically significant.

| Run | Model | N | Correlation | p-value | Notes |
|---|---|---:|---:|---:|---|
| Heuristic baseline | none | 50 | -0.245 | - | Rule-based scorer, no API |
| Initial LLM judge | Opus | 50 | -0.021 | - | Included `engagement_potential`, later identified as circular |
| Revised rubric | Opus | 50 | -0.099 | - | Replaced circular predictor with `topic_novelty` |
| Composite engagement | Opus | 50 | -0.234 | - | Used upvotes + weighted comments as target |
| Title length added | Opus | 50 | -0.159 | 0.273 | Added `title_word_count` |
| Larger run | Haiku | 499 | **-0.113** | **0.012** | First statistically significant result |
| Repeat larger run | Haiku | 499 | **-0.104** | **0.022** | Finding held across another sample |

The result is a construct validity finding: the judge reliably measures one construct, content quality from text, while HN engagement rewards another, community resonance. More capable models did not close the gap. Better signals would.

## Methodology

The pipeline:

1. Fetches live Hacker News stories from the public API.
2. Scores each story with a Claude model-as-judge using a four-dimension rubric:
   - `title_clarity`
   - `specificity`
   - `topic_novelty`
   - `information_density`
3. Combines rubric scores into a weighted quality score.
4. Builds a composite engagement target from upvotes and comments.
5. Uses Spearman rank correlation to compare judge ranking against engagement ranking.
6. Surfaces the largest disagreements for qualitative analysis.

Intermediate debugging runs are intentionally omitted from the main table unless they changed the methodology. The goal of the README is to summarize the evaluation design and stable finding, not preserve every experiment log.

## Evolution of the Evaluation

| Version | Change | Why it mattered |
|---|---|---|
| v1 | Initial rubric included `engagement_potential` | This was circular: predicting engagement with a field that already encoded engagement likelihood |
| v2 | Replaced `engagement_potential` with `topic_novelty` | Kept the rubric focused on content-quality inputs rather than the target outcome |
| v3 | Changed ground truth from raw upvotes to `upvotes + comments x 2` | Comments require more active intent than upvotes, making the target a stronger engagement signal |
| v3.1 | Added story age and title length as context | Tested whether simple available metadata explained the gap |
| v3.2 | Ran larger 499-story samples with Haiku | Confirmed the negative correlation was statistically significant |

## Evaluation Philosophy

The goal was not to force the metric upward. The goal was to learn whether the rubric was measuring the same thing as the outcome signal.

When the correlation stayed negative across methodology changes, the right conclusion was not "prompt harder." It was that text-quality dimensions were measuring a different construct than HN engagement. That distinction matters for AI evaluation work: a reliable judge can still be misaligned with the outcome you care about if the rubric is pointed at the wrong construct.

This is also why a `community_identity` dimension was considered and rejected. Community identity is not a property of the title text. It lives in behavioral data: author history, domain reputation, topic momentum, and audience context. Asking the judge to infer it from title text alone would create plausible but ungrounded scores.

For detailed failure modes, repeated examples, and future methodology notes, see [ANALYSIS.md](ANALYSIS.md).

## What I Would Do Next

- **Historical engagement features:** count how often an author or domain has appeared in the HN top 100 over a recent window.
- **Domain authority:** capture trust signals for domains like `paulgraham.com` or `tonsky.me` separately from title quality.
- **Golden set validation:** use repeated over- and under-raters as a human-scored stress test set.
- **T0 prediction design:** sample `newstories`, score at publish time, then collect engagement 6-24 hours later to reduce the feedback loop from HN's own ranking algorithm.
- **Audience fit:** evaluate whether content is right for a specific user or surface, not just whether it is good in isolation.

## Usage

```bash
pip install -r requirements.txt

# Demo mode, no API key
python content_ranking_eval.py --demo

# Full run
ANTHROPIC_API_KEY=your_key_here python content_ranking_eval.py

# Smaller test run
python content_ranking_eval.py --n 50
```

Default sample size is 500 stories. Use `--n 50` for a faster sanity check.

## Background

This prototype is modeled on evaluation work I did at Meta around table documentation quality: using query rate, AI/agent citation rate, documentation coverage gaps, and eval results to decide which tables needed better docstrings and whether the generated documentation was more reliable and useful for agents and engineers.

The content type is different, but the evaluation pattern is similar: define a quality signal, validate it against the outcome you care about, inspect where it diverges, and improve the system based on what the eval reveals.
