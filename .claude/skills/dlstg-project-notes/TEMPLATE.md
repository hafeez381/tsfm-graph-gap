# Templates

## Notes file skeleton (create only if `DLSTG Project — Project Notes.md` is missing)

```markdown
---
title: "DLSTG Project — Project Notes"
tags: [dlstg, project, research-notes]
status: active
created: YYYY-MM-DD
---

# DLSTG Project — Project Notes

> [!abstract] What this note is
> The research companion to [[DLSTG Project — Implementation Guide]]: the question, the hypotheses, what each measurement means, the findings and the open questions. Implementation lives in the guide and the `tsfm-graph-gap` repository.

> [!tip] Read first
> - Everything: <links to lecture-note headings every part assumes>
> - RQ1 …: <links, one line of why each>

## 0 The big picture
### 0.1 The question
### 0.2 Two competing hypotheses
### 0.3 The model as mathematics
### 0.4 What each measurement means

## 1 Theory per measurement
### 1.1 <measurement>

## 2 Research questions and pre-registered hypotheses

## 3 Comparison set and protocol in one table

## 4 Findings
### 4.1 RQ1 …
### 4.2 RQ2 …

## 5 Open questions and proposed experiments

## 6 Connections to other work

## 7 Self-check
> [!question]- Self-check
> 1. Question? → answer.
```

## A findings subsection

```markdown
#### 4.1.2 Chronos-Bolt zero-shot vs the naive floors (METR-LA test, one seed)

Masked MAE in mph over observed targets of $P_{\text{all}}$; windows with no observed lookback excluded.

| Model | 15 min | 30 min | 60 min | Mean |
|---|---|---|---|---|
| Last value | … | … | … | … |
| Chronos-Bolt zero-shot | … | … | … | … |

**What it shows.** One or two sentences, each tied to a number in the table, and the hypothesis it bears on: supports / contradicts / cannot yet decide [[DLSTG Project — Implementation Guide#<H heading>|H1]].

**What it does not yet show.** The missing seed, baseline, population or confound, and the run that would settle it (with its §0.9 row id if one exists).
```

## A proposal subsection (section 5)

```markdown
### 5.3 Does the adapter only learn the daily profile? (§0.9 row P3)

**Question.** One sentence.
**Why it matters.** Which RQ or hypothesis it changes, and how.
**Proposed experiment.** What changes and what is held fixed, stated as mathematics or protocol, not code.
**What each outcome would mean.** If A, then …; if B, then ….
**Cost.** GPU-hours on a Kaggle T4, people-days, and who would run it.
```

## Guide §0.9 (create under §0 of the guide if missing, after §0.8)

```markdown
### 0.9 Proposed experiments and conditional runs

Every proposal from [[DLSTG Project — Project Notes#5 Open questions and proposed experiments|the notes' §5]] that could become a run. Check this table at every decision point; log each adoption or drop in [[#0.5 Change log|§0.5]].

| # | Proposal | Run it if | Cost | Answers | Status | Reasoning |
|---|---|---|---|---|---|---|
| P1 | … | <trigger: a pilot result, spare GPU time, a failed check> | <T4 GPU-h, people-days> | <RQ / H link> | proposed | [[DLSTG Project — Project Notes#5.1 Heading\|notes §5.1]] |
```
