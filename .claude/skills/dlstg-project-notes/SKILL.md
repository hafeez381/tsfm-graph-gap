---
name: dlstg-project-notes
description: Keep the research notes of the AI651 DLSTG group project ("Do time-series foundation models need a graph?") up to date. Use after finishing any C-task, training run, evaluation or pilot of the project, when a result arrives, when an idea or proposed experiment comes up, or when asked to update the project notes, findings, open questions or the guide's proposed-experiments table (§0.9). Also use before starting a run, to check §0.9 for a conditional run whose trigger has fired.
---

# DLSTG project notes

The team (Abdul Hafeez, Huzaifa Abid, Israr Ali) keeps two documents in the `dlstg-course` repository, folder `DLSTG Project/`:

| File | Holds | Who edits |
|---|---|---|
| `DLSTG Project — Implementation Guide.md` | how to build everything: tasks C0–C23, decisions, config, interfaces, change log (§0.5), proposed-experiments table (§0.9) | whoever changes the plan |
| `DLSTG Project — Project Notes.md` | the research side: the question, hypotheses, theory per measurement, findings, open questions, proposals, self-check | whoever produces a result or an idea |

The code lives in a separate repository, `tsfm-graph-gap`. The notes never describe it.

## When to run

- **After a C-task, a training run, an evaluation or a pilot finishes:** add its findings (step 3), update any §0.9 row it affects (step 4).
- **When an idea, doubt or proposed experiment comes up** (in conversation, in a result, in a paper): add it to Open questions and proposals, and give it a §0.9 row if it could become a run.
- **Before starting any run, and at every decision point** (a gate, a sync, a failed check): read §0.9 and say whether any row's "Run it if" trigger has fired.

Ideas and decisions go into the notes or the guide, never only into the chat reply: the team rarely rereads chat. The reply only points to where you wrote them.

## Steps

1. **Sync first.** `git pull` in the `dlstg-course` repository before reading or editing. If the notes file does not exist yet, create it from [TEMPLATE.md](TEMPLATE.md).
2. **Read what you need.** The notes file in full; the guide's §0.5 change log, §0.6 research questions, §0.9 table, §4.x pre-registered hypotheses (H1–H5) and the task you just finished; the result files the run produced (`metrics.json`, `meta.json` in `tsfm-graph-gap/results/<run-id>/`). Copy every number from those files, never from memory or from a chat summary.
3. **Write the findings.** In the `## 4 Findings` section, under the RQ it informs, add a subsection in the format of TEMPLATE.md: a table of the numbers, then **What it shows**, then **What it does not yet show** (the confound, the missing seed, the missing baseline, the population it was not measured on). State each number with its unit, population, horizon, split, dataset and seed count. Compare against the pre-registered hypothesis it tests and say whether the result supports, contradicts or cannot yet decide it.
4. **Update the guide's §0.9 table** (`### 0.9 Proposed experiments and conditional runs`, columns `# | Proposal | Run it if | Cost | Answers | Status | Reasoning`):
   - Every proposal in the notes that could become a run has exactly one row, and the row's Reasoning cell links back to the notes subsection (`[[DLSTG Project — Project Notes#5.3 Heading|notes §5.3]]`), and the notes subsection names its row (`§0.9 row P4`).
   - Status is one of: `proposed`; `waiting on <X>` (name the result or task); `adopted` (link the task that now carries it, e.g. `adopted → [[#C16 …|C16]]`); `dropped` (with the reason).
   - When a row becomes adopted or dropped, add a line to the §0.5 change log (date, change, why, by).
   - Row ids `P1, P2, …` are stable: never renumber; a new proposal gets the next id.
5. **Update the theory** only if the result needs a concept the notes do not explain yet (step "Theory rules" below).
6. **Extend the self-check** with one or two conceptual questions the new result raises.
7. **Verify the rendering** (checklist below), then commit and push:
   - `git add` the two file paths explicitly, never `git add .` or `-A`.
   - One Conventional-Commits line, nothing else: `docs(project): add <run> findings to pa notes`, `docs(project): add proposal P5 to guide 0.9`. Notes and guide changes from one result can share a commit.
   - `git pull --rebase` then `git push` the same day. On a conflict in either file, keep both changes and tell the teammate who made the other one.

## What the notes contain, and what they never contain

**Contain:** the question and the competing hypotheses; what each measurement means and why it answers its RQ; theory; findings as tables with their interpretation and limits; open questions and proposed experiments with what each would show and what it costs (GPU-hours on a Kaggle T4, people-days); connections to the members' other work where a lesson carries over; a collapsed self-check.

**Never contain:** code blocks, file or function names, file paths, scripts, commands, test names, raw run output or logs, hardware or pipeline mechanics, implementation pitfalls, tooling or setup, process history (who reviewed what). Those belong in `tsfm-graph-gap` and the guide. The model may appear as mathematics (equations, shapes, what is frozen and what is trained), never as code.

## Theory rules

- **Lecture notes first.** The course lecture notes are in `My Notes/` of the same repository (`DLSTG_Lecture_<n> - <title>.md`, plus paper notes such as `DLSTG_Paper - Graph WaveNet.md`). Before linking, list a file's headings with `grep -n "^#" "<file>"`: headings get edited, so never link from memory. Link by exact heading, dropping characters Obsidian strips (a colon): `[[DLSTG_Lecture_6 - Spatiotemporal Modeling#3.2 Heading|L6 §3.2]]`. Quote the lecture's own sentence or formula verbatim in a `>` block, then explain how it applies here.
- **Do not repeat the project notes.** Where `DLSTG Project — Background Primer.md` or `DLSTG Project — Research Directions.md` already explains something, link its section instead of re-explaining it.
- **Where the lectures do not cover it**, explain it from the papers in `DLSTG Project/papers/` and say the lectures do not cover it yet.

## Writing rules

- **Every cross-reference is a link, never plain text.** Inside the notes: `[[#4.2 Heading|§4.2]]`, equations `[[#^eq-3|eq. (3)]]`. To the guide: `[[DLSTG Project — Implementation Guide#C12 Adapter training script|C12]]`, decisions `[[DLSTG Project — Implementation Guide#^decision-7|Decision 7]]`, hypotheses and RQs by their headings. Before handing over, scan for any `§n`, `C<n>`, `Decision n`, `H<n>` or `eq. (n)` outside a `[[…]]` and fix it.
- **Precise.** Exact definitions, quantities with units, claims with their conditions. Say "masked MAE in mph on METR-LA test, 60-minute horizon, one seed (651)", not "the error dropped". When a source is loose, state the precise version and say in one line what was tightened.
- **Notation.** Vectors carry an over-arrow, `$\vec{h}_i \in \mathbb{R}^{d}$`, never bold. State the size of every vector and matrix where it first appears, give functions their domain and codomain, use `\arg\min`, `\operatorname{}` for named functions, and full index sets on sums. Keep the guide's symbols ($\hat y^{(0)}_i$, $s_i$, $G$, $\Gamma_i$, $P_{\text{onset}}$, …); no symbol carries two meanings.
- **Voice.** Plain, short sentences. Each paragraph and list item opens with a sentence that states its point. Bold lead phrases, tables for comparisons, one idea per bullet. No filler, no hype ("significantly", "crucially") unless a test backs it.
- **One seed is a pilot.** With seed 651 only, say "one seed" next to every comparison and do not call a difference significant unless the paired day bootstrap (guide Decision 15) was run on it.

## Rendering checklist (Obsidian)

- YAML frontmatter is the first line and parses.
- Every table has a blank line above it, its `|---|` separator on its own line, equal column counts, and `\|` for any `|` inside a cell (wikilinks with an alias included).
- Every `[[note#heading]]` target exists in the current file (check with grep).
- Math: no unmatched `$`; no bare `_` or `*` outside math.
- Callouts: every line of a callout body starts with `> `; the self-check is collapsed, `> [!question]- Self-check`.
- Block ids (`^eq-3`) sit on their own line after the closing `$$`.
