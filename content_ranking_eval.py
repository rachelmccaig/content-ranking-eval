#!/usr/bin/env python3
"""
Content Ranking Eval Pipeline
==============================
Built by Rachel McCaig (Barden) as a hands-on prototype of the model-as-judge
workflow used by Product Content Engineering teams at consumer content platforms.

WHAT THIS IS
------------
A model-as-judge eval pipeline that fetches real content from HackerNews,
scores it against a grading framework of quality dimensions, then validates
whether those scores predict actual human engagement (HN upvote score).

This is the core PCE loop: define a rubric, run a model-as-judge, validate
the judge against a ground truth signal, identify where the rubric is missing
signal, iterate.

The pipeline is modeled on content prioritization work I did at Meta: using
query rate, AI/agent citation rate, and documentation coverage gaps as signals
to decide where content investment would actually move the needle — behavioral
signals revealing content need, not just quality in isolation. The underlying
operation is the same: attach structured meaning to a content asset, validate
that meaning against a real-world outcome signal, improve the rubric where it
diverges.

ITERATION HISTORY — rubric validation and pipeline optimization
---------------------------------------------------------------
v1 — Grading framework: 4 dimensions (title clarity, specificity, engagement
     potential, information density). Model-as-judge: Claude Haiku.
     Ground truth: HackerNews upvote score (50 stories).
     Spearman correlation: -0.021 (p=0.883). No signal.

     Pipeline optimization finding: The rubric measured writing quality, not
     community engagement. Qualitative observation: "Every Frame Perfect"
     (813 pts, HN rank #1) scored judge rank #45. The title is objectively
     vague — but it's by Tonsky, a trusted community author. Author identity
     was a missing signal the rubric couldn't capture.

     Translation: qualitative divergence → quantitative hypothesis →
     new rubric dimension.

v2 — Added author_reputation as a 5th rubric dimension, using HN karma as
     a proxy for author standing. Reweighted all dimensions.
     Spearman correlation: -0.099. Moved in the wrong direction.

     Pipeline optimization finding: Karma is a noisy proxy. "Every Frame
     Perfect" is posted from tonsky.me — domain authority carries the
     reputation signal, not account karma. High-karma authors also post
     content that doesn't land. The hypothesis was right (author identity
     matters) but the signal was wrong (karma ≠ trust).

KNOWN LIMITATIONS — what v3 would address
------------------------------------------
1. No golden set for rubric validation against human labels. Currently
   validating the model-as-judge against engagement signal only. A proper
   golden set — human-scored sample compared against judge scores — would
   measure judge quality independent of engagement and surface rubric bias
   more precisely.

2. Quality vs. relevance: this pipeline evaluates content in isolation.
   Production ranking on Instagram or Facebook asks "is this right for THIS
   person, on THIS surface, right now?" — not just "is this good?"
   The next version would incorporate audience segment signals to test
   whether the rubric predicts engagement for specific user cohorts, not
   just aggregate scores.

3. Domain authority: tonsky.me and paulgraham.com carry community trust
   that karma doesn't capture. A domain reputation signal would be a
   stronger proxy than account karma for v3.

4. Topic trend signal: timely content (release announcements, ongoing news
   cycles) consistently outperforms quality predictions. A "topic recency"
   dimension would likely improve correlation significantly.

Usage:
    # Demo mode (no API key needed — uses heuristic scorer):
    python content_ranking_eval.py --demo

    # Full run with real model-as-judge:
    ANTHROPIC_API_KEY=your_key_here python content_ranking_eval.py

    # Adjust number of stories:
    python content_ranking_eval.py --n 30
"""

import argparse
import json
import os
import sys
import time
import csv
from datetime import datetime

import requests
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────

HN_API = "https://hacker-news.firebaseio.com/v0"
DEFAULT_N = 50
MODEL = "claude-haiku-4-5-20251001"   # fast + cheap; swap to sonnet for richer reasoning

# Rubric dimension weights for composite judge score.
# v1 grading framework had 4 dimensions. author_reputation was added in v2 after
# rubric validation showed near-zero correlation and pipeline optimization surfaced
# author identity as a missing signal. Weights redistributed for v2.
# v2 finding: karma is too noisy a proxy for author trust — domain authority
# would be a stronger signal for v3.
WEIGHTS = {
    "title_clarity":        0.15,
    "specificity":          0.20,
    "engagement_potential": 0.30,
    "information_density":  0.15,
    "author_reputation":    0.20,
}

# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_author_karma(username: str) -> int:
    """
    Fetch a HackerNews user's karma score — used as an author reputation proxy
    in the v2 grading framework. Added after v1 rubric validation surfaced author
    identity as a missing signal. Karma turned out to be noisy (see KNOWN LIMITATIONS);
    domain authority would be a stronger rubric dimension for v3.
    """
    if not username:
        return 0
    try:
        user = requests.get(f"{HN_API}/user/{username}.json", timeout=5).json()
        return user.get("karma", 0) if user else 0
    except Exception:
        return 0


def fetch_stories(n: int) -> list[dict]:
    """Pull top stories from HackerNews public API, including author karma."""
    print(f"Fetching {n} stories from HackerNews...")
    resp = requests.get(f"{HN_API}/topstories.json", timeout=10)
    resp.raise_for_status()
    ids = resp.json()[:n * 2]   # overfetch to account for non-story items

    stories = []
    for story_id in ids:
        if len(stories) >= n:
            break
        item = requests.get(f"{HN_API}/item/{story_id}.json", timeout=10).json()
        if not item or item.get("type") != "story" or not item.get("title"):
            continue
        domain = ""
        url = item.get("url", "")
        if url and "//" in url:
            domain = url.split("/")[2].replace("www.", "")
        author = item.get("by", "")
        karma  = fetch_author_karma(author)
        stories.append({
            "id":           story_id,
            "title":        item.get("title", ""),
            "domain":       domain,
            "hn_score":     item.get("score", 0),
            "comments":     item.get("descendants", 0),
            "text":         (item.get("text") or "")[:300],
            "author":       author,
            "author_karma": karma,
        })
        time.sleep(0.05)   # be a good API citizen

    print(f"  → Got {len(stories)} stories\n")
    return stories


# ── Scoring: LLM judge ───────────────────────────────────────────────────────

def score_with_llm(story: dict, client) -> dict:
    """
    Score a story on 4 content quality dimensions using Claude as judge.
    Returns a dict with dimension scores (1-10) and a one-sentence reasoning.
    """
    # Bucket karma into a readable label so the judge can reason about it naturally.
    # Raw numbers mean nothing to an LLM — labels give it interpretable context.
    karma = story.get("author_karma", 0)
    if karma > 50000:
        karma_label = f"very high ({karma:,} karma) — likely a well-known, trusted contributor"
    elif karma > 10000:
        karma_label = f"high ({karma:,} karma) — established community member"
    elif karma > 1000:
        karma_label = f"moderate ({karma:,} karma) — active contributor"
    else:
        karma_label = f"low ({karma:,} karma) — newer or less active account"

    context_lines = [f"Title: {story['title']}"]
    if story["domain"]:
        context_lines.append(f"Source domain: {story['domain']}")
    if story.get("author"):
        context_lines.append(f"Author reputation: {karma_label}")
    if story["text"]:
        context_lines.append(f"Text preview: {story['text'][:200]}")

    prompt = f"""You are a content quality evaluator for a large social content platform.
Score the following content on 5 dimensions. Respond ONLY with valid JSON — no preamble, no markdown.

{chr(10).join(context_lines)}

Dimensions to score (each 1–10):
- title_clarity: How clear and immediately understandable is the title?
- specificity: How specific and concrete is this vs. vague/generic?
- engagement_potential: How likely is a tech-savvy professional audience to engage with and discuss this?
- information_density: How much useful signal is in the title? (penalize clickbait and fluff)
- author_reputation: How much does the author's community standing suggest this content is worth reading?

Required format:
{{"title_clarity": <int>, "specificity": <int>, "engagement_potential": <int>, "information_density": <int>, "author_reputation": <int>, "reasoning": "<one sentence>"}}"""

    for attempt in range(3):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == 2:
                raise ValueError(f"Bad JSON after 3 attempts. Raw: {repr(raw)}")
            time.sleep(1)


# ── Scoring: heuristic demo (no API key required) ────────────────────────────

def score_heuristic(story: dict) -> dict:
    """
    Simple heuristic scorer for demo mode — no API key required.
    Not a real judge; exists only to show the pipeline structure.
    """
    title = story["title"]
    words = title.split()
    word_count = len(words)
    has_number = any(w.replace(",", "").replace(".", "").isdigit() for w in words)
    has_colon   = ":" in title
    is_question = title.endswith("?")
    length_score = min(10, max(1, word_count))

    title_clarity        = min(10, 5 + (2 if has_colon else 0) + (1 if word_count < 12 else -1))
    specificity          = min(10, 4 + (3 if has_number else 0) + (2 if has_colon else 0))
    engagement_potential = min(10, 5 + (2 if is_question else 0) + (1 if has_number else 0))
    information_density  = min(10, max(1, length_score - (2 if is_question else 0)))

    # Heuristic author reputation: bucket karma into 1-10
    karma = story.get("author_karma", 0)
    author_reputation = min(10, max(1, int(karma / 10000) + 1))

    return {
        "title_clarity":        title_clarity,
        "specificity":          specificity,
        "engagement_potential": engagement_potential,
        "information_density":  information_density,
        "author_reputation":    author_reputation,
        "reasoning":            "[demo mode — heuristic scorer, not a real LLM judge]",
    }


# ── Spearman correlation (numpy, no scipy required) ─────────────────────────

def _spearmanr(x: np.ndarray, y: np.ndarray):
    """Compute Spearman rank correlation and two-tailed p-value using numpy."""
    n = len(x)
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    d  = rx - ry
    rs = 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))
    # t-distribution approximation for p-value
    t  = rs * np.sqrt((n - 2) / max(1 - rs**2, 1e-10))
    from math import gamma, sqrt, pi
    def beta(a, b):
        return gamma(a) * gamma(b) / gamma(a + b)
    # two-tailed p via regularized incomplete beta (approx for large n)
    df = n - 2
    x2 = df / (df + t**2)
    # simple approximation: use normal dist for n > 20
    import math
    if n > 20:
        z = 0.5 * math.log((1 + rs) / max(1 - rs, 1e-10)) * math.sqrt(n - 3)
        p = 2 * (1 - _norm_cdf(abs(z)))
    else:
        p = float("nan")
    return float(rs), float(p)

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function."""
    import math
    return (1 + math.erf(x / math.sqrt(2))) / 2


# ── Composite score ──────────────────────────────────────────────────────────

def composite(scores: dict) -> float:
    return sum(scores[dim] * weight for dim, weight in WEIGHTS.items())


# ── Report ───────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_report(stories: list[dict], corr: float, p_value: float, demo: bool):
    mode_label = " [DEMO MODE — heuristic scorer]" if demo else ""
    print(f"\n{'='*60}")
    print(f"  CONTENT RANKING EVAL RESULTS — v2 (+ author reputation){mode_label}")
    print(f"{'='*60}")

    # Correlation
    sig = "✓ statistically significant" if p_value < 0.05 else "✗ not significant (need more data or better dimensions)"
    print(f"\nSpearman rank correlation: {corr:+.3f}  (p = {p_value:.3f})")
    print(f"Signal quality:  {sig}")
    print("""
What this means:
  A positive correlation means your LLM judge is picking up real signal —
  content it scores highly also tends to get more human engagement.
  A low or negative correlation means the dimensions need recalibration:
  either they're measuring the wrong things, or weighting is off.""")

    # Sort helpers
    by_llm = sorted(stories, key=lambda s: s["llm_rank"])
    by_hn  = sorted(stories, key=lambda s: s["hn_rank"])

    print_section("TOP 10 BY LLM JUDGE SCORE")
    for s in by_llm[:10]:
        print(f"  #{int(s['llm_rank']):>2}  [{s['llm_composite']:.1f}/10]  "
              f"HN rank #{int(s['hn_rank']):>3}  {s['title'][:65]}")

    print_section("TOP 10 BY ACTUAL HN SCORE")
    for s in by_hn[:10]:
        print(f"  #{int(s['hn_rank']):>2}  [{s['hn_score']:>4} pts]  "
              f"LLM rank #{int(s['llm_rank']):>3}  {s['title'][:65]}")

    print_section("BIGGEST DISAGREEMENTS — where LLM and humans diverged")
    gaps = sorted(stories, key=lambda s: abs(s["llm_rank"] - s["hn_rank"]), reverse=True)
    for s in gaps[:5]:
        gap = int(abs(s["llm_rank"] - s["hn_rank"]))
        direction = "LLM overrated" if s["llm_rank"] < s["hn_rank"] else "LLM underrated"
        print(f"\n  {direction} by {gap} positions")
        print(f"  Title:    {s['title'][:70]}")
        print(f"  LLM rank: #{int(s['llm_rank'])}  |  HN rank: #{int(s['hn_rank'])}  |  HN score: {s['hn_score']}")
        print(f"  Author: {s.get('author', '?')}  (karma: {s.get('author_karma', 0):,})")
        print(f"  Reasoning: {s['reasoning']}")

    print("""
What to learn from disagreements:
  Cases where humans rate something highly but your judge doesn't often reveal
  signals your dimensions don't capture — timeliness, community-specific
  interest, author reputation, or novelty. This is where you'd iterate on
  your eval rubric or add new dimensions.""")


# ── Save results ─────────────────────────────────────────────────────────────

def save_csv(stories: list[dict], corr: float, p_value: float):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ranking_eval_results_{timestamp}.csv"
    fields = [
        "title", "domain", "author", "author_karma", "hn_score", "comments",
        "title_clarity", "specificity", "engagement_potential", "information_density", "author_reputation",
        "llm_composite", "llm_rank", "hn_rank", "rank_gap", "reasoning",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stories)
    print(f"\n✓ Full results saved to: {filename}")
    print(f"  Spearman correlation: {corr:+.3f}  (p = {p_value:.3f})")
    return filename


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Content ranking eval pipeline")
    parser.add_argument("--n",    type=int,  default=DEFAULT_N, help="Number of stories to evaluate")
    parser.add_argument("--demo", action="store_true",          help="Run without API key using heuristic scorer")
    args = parser.parse_args()

    # ── Setup scorer ──
    if args.demo:
        print("Running in DEMO MODE (heuristic scorer — no API key required)")
        print("To use a real LLM judge: ANTHROPIC_API_KEY=sk-... python content_ranking_eval.py\n")
        scorer = score_heuristic
        client = None
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set.")
            print("Run in demo mode:  python content_ranking_eval.py --demo")
            print("Or set your key:   export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("ERROR: anthropic package not installed.")
            print("Install it: pip install anthropic")
            sys.exit(1)
        scorer = None

    # ── Fetch data ──
    stories = fetch_stories(args.n)
    if not stories:
        print("No stories fetched. Check your internet connection.")
        sys.exit(1)

    # ── Score ──
    print(f"Scoring {len(stories)} stories with {'heuristic' if args.demo else 'LLM'} judge...")
    for i, story in enumerate(stories):
        prefix = f"  [{i+1:>2}/{len(stories)}]"
        print(f"{prefix} {story['title'][:60]}...")
        scores = scorer(story) if args.demo else score_with_llm(story, client)
        story.update(scores)
        story["llm_composite"] = composite(scores)
        if not args.demo:
            time.sleep(0.3)  # stay under rate limits

    # ── Rank ──
    sorted_by_llm = sorted(stories, key=lambda s: s["llm_composite"], reverse=True)
    sorted_by_hn  = sorted(stories, key=lambda s: s["hn_score"],      reverse=True)
    for rank, s in enumerate(sorted_by_llm, 1):
        s["llm_rank"] = rank
    for rank, s in enumerate(sorted_by_hn, 1):
        s["hn_rank"] = rank
    for s in stories:
        s["rank_gap"] = abs(s["llm_rank"] - s["hn_rank"])

    # ── Validate ──
    # Spearman rank correlation: validates whether the model-as-judge grading framework
    # predicts actual human engagement in the same order. Used instead of Pearson because
    # judge scores (1-10) and engagement signal (raw HN points) are different scales —
    # we care about order agreement, not magnitude.
    # +1.0 = perfect agreement, -1.0 = perfectly opposite, 0 = no relationship.
    # Low correlation = rubric needs optimization; high correlation = judge is capturing
    # real signal.
    llm_ranks = np.array([s["llm_rank"] for s in stories], dtype=float)
    hn_ranks  = np.array([s["hn_rank"]  for s in stories], dtype=float)
    corr, p_value = _spearmanr(llm_ranks, hn_ranks)

    # ── Report ──
    run_report(stories, corr, p_value, args.demo)
    save_csv(stories, corr, p_value)


if __name__ == "__main__":
    main()
