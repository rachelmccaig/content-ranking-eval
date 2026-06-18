# Analysis Notes

This document captures the detailed observations behind the README: repeated failure modes, examples from individual runs, and methodology questions for future iterations.

## What the Eval Revealed

The model-as-judge consistently rewarded stories with clear, specific, information-dense titles. Hacker News engagement consistently rewarded a different set of signals:

- community resonance
- breaking news cycles
- known projects and authors
- domain authority
- recurring community rituals
- provocative framing

That gap is the central finding. The judge was not random. It was reliably measuring text quality. The problem was that text quality and HN engagement were different constructs.

## Repeated Underrated Stories

These are stories HN engaged with heavily while the judge ranked them low.

### Product or project launches with existing community followings

Examples:

- `Iroh 1.0`
- `Claude Fable 5`

The titles are short and low-information in isolation. The judge sees sparse text. HN sees a known project or product moment. This is not visible from title quality alone.

### Community rituals

Example:

- `Ask HN: What are you working on?`

The judge correctly identifies this as low-information content. HN still engages because the post is a recurring community ritual. The value is in the social pattern, not the title.

### Community hero worship

Examples:

- `I admire Fabrice Bellard`
- Paul Graham essays on `paulgraham.com`

The judge penalizes vague or clickbait-like titles. HN responds to the person or domain behind the content. The missing signal is reputation and cultural context.

### Breaking news and topic momentum

Example:

- `Statement on US government directive to suspend access to Fable 5`

This ranked highly by engagement while the judge ranked it low. The story belonged to an active news cycle, but the judge had no topic-momentum signal. It could only evaluate the title text.

### Provocative framing

Example:

- `Stop Using JWTs`

The judge penalizes rehashed or provocative takes. HN often engages with them because they create debate. This is an engagement dynamic, not necessarily a content-quality signal.

## Repeated Overrated Stories

These are stories the judge liked more than HN did.

### Niche but precise technical titles

Example:

- `Nipkow Disk Mechanical TV Simulator`

This appeared as an overrated outlier in multiple runs. The judge liked that it was concrete, specific, and technically interesting. HN did not engage at the same level.

### Technically polished but socially low-momentum posts

The judge often rewarded niche Show HN projects and obscure technical simulators because they matched the rubric. That does not mean they had broad community resonance.

## Golden Set Candidates

The repeated over- and under-raters form a natural starting point for a golden set: not a random sample, but a stress test for the known mismatch between content quality and community engagement.

Potential golden set categories:

- sparse titles with high engagement because of project reputation
- vague titles with high engagement because of author or domain authority
- technically precise titles with low engagement
- breaking-news titles with sparse text but high momentum
- recurring community ritual posts

A human-scored golden set could separate two questions that are currently blended:

1. Is the judge accurately applying the quality rubric?
2. Does that rubric predict engagement?

Those are different evaluation problems.

## Why `community_identity` Was Rejected

A `community_identity` dimension was considered but rejected.

The reason: community identity is not in the title. It is a property of the relationship between content and audience. To score it properly, the system would need access to behavioral data such as:

- author historical engagement
- domain historical engagement
- topic trend graphs
- community interaction history
- audience segment fit

Without those signals, a model would be guessing from text alone. The scores might sound plausible, but they would not be grounded.

## Methodology Notes

### Why Hacker News?

Hacker News was useful because it provides:

- public story data
- public engagement signals
- a ranked content surface
- enough volume for repeatable experiments

It is not a perfect proxy for Instagram or Facebook. Its audience is narrow and its engagement dynamics are community-specific. That limitation is useful: it makes the construct mismatch visible.

### Why Spearman correlation?

The judge produces rubric scores on a 1-10 scale. HN engagement is measured in points and comments. The raw numbers are not comparable, but their order is. Spearman correlation asks whether the two systems rank stories similarly.

### Why not optimize until correlation improves?

Because improving correlation by adding ungrounded proxy dimensions would make the eval less honest. The point was to identify whether text-based quality features predict engagement. The answer was no, not reliably.

The next meaningful improvement is not more prompt tuning. It is adding real behavioral signals.

## Engineering Review Notes

After the core iteration, the script was reviewed with Codex for engineering hardening. The review identified useful robustness improvements:

- scipy Spearman when available, with tie-aware fallback
- LLM output validation
- retry and backoff for API calls
- checkpointing for long runs
- optional age-adjusted engagement
- support for `topstories`, `newstories`, and `beststories`

Those changes were kept separate from the canonical script because some alter the methodology used for the reported results. The main repo preserves the script that generated the README findings.

## Future Direction

The strongest next experiment would be a T0 prediction design:

1. Sample stories from `newstories`.
2. Score them before HN's ranking algorithm has amplified visibility.
3. Wait 6-24 hours.
4. Collect engagement.
5. Test whether the judge predicts future engagement.

This would reduce the feedback loop where HN's own ranking system affects visibility, which affects engagement, which then becomes the ground truth.
