#!/usr/bin/env python3
"""
Content Ranking Eval Pipeline
Rachel McCaig (Barden)

A model-as-judge pipeline for validating content ranking signals against real
human engagement data. Fetches HackerNews stories, scores them on a rubric of
non-circular predictor dimensions, and validates using Spearman rank correlation.

Iteration history:
  v1: 4 dimensions (clarity, specificity, engagement_potential, information_density)
      Correlation: -0.021. engagement_potential was circular — predicting engagement
      with a dimension that already encodes engagement likelihood.
  v2: Replaced engagement_potential with topic_novelty. Added author_karma and
      article_age_hours as data signals. Switched to Opus for richer reasoning.
  v3: Updated ground truth from raw upvote score to a composite engagement signal
      weighting comments more heavily than upvotes. Comments require active intent;
      upvotes are passive. Chosen interaction signals are more valuable for ranking
      than impression-level signals.

Usage:
    python content_ranking_eval.py --demo          # no API key, heuristic scorer
    ANTHROPIC_API_KEY=sk-... python content_ranking_eval.py
    python content_ranking_eval.py --n 100
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests

HN_API = "https://hacker-news.firebaseio.com/v0"
DEFAULT_N = 500  # POC uses 50; 500 recommended for statistical significance. Note: Opus at 500 stories takes ~20-30 min and costs ~$10-15.
MODEL = "claude-haiku-4-5-20251001"

# Ground truth weighting. Comments require active intent; upvotes are passive.
# Weighting comments more heavily reflects that chosen interaction is a stronger
# engagement signal than impressions. Hypothesis: 2x is a reasonable starting point.
COMMENT_WEIGHT = 2

# Rubric weights. engagement_potential removed (circular predictor).
# topic_novelty added as a genuine input signal.
WEIGHTS = {
    "title_clarity":     0.20,
    "specificity":       0.25,
    "topic_novelty":     0.30,
    "information_density": 0.25,
}


# -- Data fetching ------------------------------------------------------------

def fetch_author_karma(username: str) -> int:
    if not username:
        return 0
    try:
        user = requests.get(f"{HN_API}/user/{username}.json", timeout=5).json()
        return user.get("karma", 0) if user else 0
    except Exception:
        return 0


def fetch_stories(n: int) -> list[dict]:
    print(f"Fetching {n} stories from HackerNews...")
    resp = requests.get(f"{HN_API}/topstories.json", timeout=10)
    resp.raise_for_status()
    ids = resp.json()[:n * 2]

    now = datetime.now(timezone.utc).timestamp()
    stories = []
    for story_id in ids:
        if len(stories) >= n:
            break
        item = requests.get(f"{HN_API}/item/{story_id}.json", timeout=10).json()
        if not item or item.get("type") != "story" or not item.get("title"):
            continue
        url = item.get("url", "")
        domain = url.split("/")[2].replace("www.", "") if url and "//" in url else ""
        author = item.get("by", "")
        posted_at = item.get("time", now)
        hn_score = item.get("score", 0)
        comments = item.get("descendants", 0)
        title = item.get("title", "")
        stories.append({
            "id":               story_id,
            "title":            title,
            "title_word_count": len(title.split()),
            "domain":           domain,
            "hn_score":         hn_score,
            "comments":         comments,
            "engagement":       hn_score + (comments * COMMENT_WEIGHT),
            "text":             (item.get("text") or "")[:300],
            "author":           author,
            "author_karma":     fetch_author_karma(author),
            "article_age_hours": round((now - posted_at) / 3600, 1),
        })
        time.sleep(0.05)

    print(f"  -> Got {len(stories)} stories\n")
    return stories


# -- LLM judge ----------------------------------------------------------------

def score_with_llm(story: dict, client) -> dict:
    karma = story.get("author_karma", 0)
    if karma > 50000:
        karma_label = f"very high ({karma:,}) — well-known contributor"
    elif karma > 10000:
        karma_label = f"high ({karma:,}) — established member"
    elif karma > 1000:
        karma_label = f"moderate ({karma:,}) — active contributor"
    else:
        karma_label = f"low ({karma:,}) — newer account"

    lines = [f"Title: {story['title']} ({story.get('title_word_count', len(story['title'].split()))} words)"]
    if story["domain"]:
        lines.append(f"Source: {story['domain']}")
    if story.get("author"):
        lines.append(f"Author karma: {karma_label}")
    lines.append(f"Article age: {story['article_age_hours']} hours old")
    if story["text"]:
        lines.append(f"Text: {story['text'][:200]}")

    prompt = f"""You are a content quality evaluator. Score the following on 4 dimensions.
Respond ONLY with valid JSON, no other text.

{chr(10).join(lines)}

Dimensions (1-10 each):
- title_clarity: How clear and understandable is the title?
- specificity: How specific and concrete vs. vague/generic?
- topic_novelty: How fresh or non-obvious is this topic? Penalize rehashed takes.
- information_density: How much real signal is packed in? Penalize clickbait.

Format: {{"title_clarity": <int>, "specificity": <int>, "topic_novelty": <int>, "information_density": <int>, "reasoning": "<one sentence>"}}"""

    for attempt in range(3):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == 2:
                raise ValueError(f"Bad JSON after 3 attempts: {repr(raw)}")
            time.sleep(1)


# -- Heuristic scorer (demo mode) ---------------------------------------------

def score_heuristic(story: dict) -> dict:
    title = story["title"]
    words = title.split()
    word_count = len(words)
    has_number = any(w.replace(",", "").replace(".", "").isdigit() for w in words)
    has_colon = ":" in title
    is_question = title.endswith("?")

    title_clarity    = min(10, 5 + (2 if has_colon else 0) + (1 if word_count < 12 else -1))
    specificity      = min(10, 4 + (3 if has_number else 0) + (2 if has_colon else 0))
    topic_novelty    = min(10, 5 + (2 if not is_question else 0) + (1 if has_number else 0))
    info_density     = min(10, max(1, min(word_count, 10) - (2 if is_question else 0)))

    return {
        "title_clarity":      title_clarity,
        "specificity":        specificity,
        "topic_novelty":      topic_novelty,
        "information_density": info_density,
        "reasoning":          "[demo mode — heuristic scorer]",
    }


# -- Spearman correlation (no scipy) ------------------------------------------

def _norm_cdf(x: float) -> float:
    return (1 + math.erf(x / math.sqrt(2))) / 2


def _spearmanr(x: np.ndarray, y: np.ndarray):
    n = len(x)
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    d = rx - ry
    rs = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))
    if n > 20:
        z = 0.5 * math.log((1 + rs) / max(1 - rs, 1e-10)) * math.sqrt(n - 3)
        p = 2 * (1 - _norm_cdf(abs(z)))
    else:
        p = float("nan")
    return float(rs), float(p)


# -- Composite score ----------------------------------------------------------

def composite(scores: dict) -> float:
    return sum(scores[dim] * weight for dim, weight in WEIGHTS.items())


# -- Report -------------------------------------------------------------------

def run_report(stories: list[dict], corr: float, p_value: float, demo: bool):
    label = " [DEMO]" if demo else ""
    print(f"\n{'='*60}")
    print(f"  CONTENT RANKING EVAL — v2{label}")
    print(f"{'='*60}")

    sig = "significant" if p_value < 0.05 else "not significant"
    print(f"\nSpearman correlation: {corr:+.3f}  (p={p_value:.3f}, {sig})")
    print(f"Ground truth: composite engagement score (upvotes + comments x{COMMENT_WEIGHT})")

    by_llm = sorted(stories, key=lambda s: s["llm_rank"])
    by_eng = sorted(stories, key=lambda s: s["hn_rank"])

    print("\nTop 10 by judge score:")
    for s in by_llm[:10]:
        print(f"  #{int(s['llm_rank']):>2}  [{s['llm_composite']:.1f}]  "
              f"Engagement #{int(s['hn_rank']):>3}  {s['title'][:65]}")

    print("\nTop 10 by composite engagement:")
    for s in by_eng[:10]:
        print(f"  #{int(s['hn_rank']):>2}  [{s['hn_score']} pts + {s['comments']} comments]  "
              f"Judge #{int(s['llm_rank']):>3}  {s['title'][:65]}")

    print("\nBiggest disagreements:")
    gaps = sorted(stories, key=lambda s: abs(s["llm_rank"] - s["hn_rank"]), reverse=True)
    for s in gaps[:5]:
        gap = int(abs(s["llm_rank"] - s["hn_rank"]))
        direction = "overrated" if s["llm_rank"] < s["hn_rank"] else "underrated"
        print(f"\n  Judge {direction} by {gap} positions")
        print(f"  Title:  {s['title'][:70]}")
        print(f"  Judge #{int(s['llm_rank'])}  |  Engagement #{int(s['hn_rank'])}  |  "
              f"{s['hn_score']} pts + {s['comments']} comments = {s['engagement']} engagement  |  "
              f"age: {s.get('article_age_hours', '?')}h  |  karma: {s.get('author_karma', 0):,}")
        print(f"  Note:   {s['reasoning']}")


# -- Save ---------------------------------------------------------------------

def save_csv(stories: list[dict], corr: float, p_value: float):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ranking_eval_results_{timestamp}.csv"
    fields = [
        "title", "title_word_count", "domain", "author", "author_karma", "article_age_hours",
        "hn_score", "comments", "engagement",
        "title_clarity", "specificity", "topic_novelty", "information_density",
        "llm_composite", "llm_rank", "hn_rank", "rank_gap", "reasoning",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stories)
    print(f"\nResults saved to {filename}")
    print(f"Spearman correlation: {corr:+.3f}  (p={p_value:.3f})")
    return filename


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Content ranking eval pipeline")
    parser.add_argument("--n",    type=int,  default=DEFAULT_N)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        print("Demo mode (heuristic scorer, no API key needed)\n")
        scorer = score_heuristic
        client = None
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Set ANTHROPIC_API_KEY or run with --demo")
            sys.exit(1)
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("pip install anthropic")
            sys.exit(1)
        scorer = None

    stories = fetch_stories(args.n)
    if not stories:
        print("No stories fetched.")
        sys.exit(1)

    print(f"Scoring {len(stories)} stories...")
    for i, story in enumerate(stories):
        print(f"  [{i+1:>2}/{len(stories)}] {story['title'][:60]}...")
        scores = scorer(story) if args.demo else score_with_llm(story, client)
        story.update(scores)
        story["llm_composite"] = composite(scores)
        if not args.demo:
            time.sleep(0.3)

    sorted_by_llm = sorted(stories, key=lambda s: s["llm_composite"], reverse=True)
    sorted_by_eng = sorted(stories, key=lambda s: s["engagement"],    reverse=True)
    for rank, s in enumerate(sorted_by_llm, 1):
        s["llm_rank"] = rank
    for rank, s in enumerate(sorted_by_eng, 1):
        s["hn_rank"] = rank
    for s in stories:
        s["rank_gap"] = abs(s["llm_rank"] - s["hn_rank"])

    llm_ranks = np.array([s["llm_rank"] for s in stories], dtype=float)
    hn_ranks  = np.array([s["hn_rank"]  for s in stories], dtype=float)
    corr, p_value = _spearmanr(llm_ranks, hn_ranks)

    run_report(stories, corr, p_value, args.demo)
    save_csv(stories, corr, p_value)


if __name__ == "__main__":
    main()
