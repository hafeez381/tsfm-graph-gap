# CLAUDE.md — tsfm-graph-gap (AI651 DLSTG group project)

Shared by the team (Abdul Hafeez, Huzaifa Abid, Israr Ali). Committed; change it through a pull request.

## What this repo is
Code for "Do time-series foundation models need a graph?": a frozen TSFM (Chronos-Bolt Small, TimesFM 2.5)
plus a small graph attention adapter on its cached outputs, compared with zero-shot, Chronos-2, full
fine-tuning, PatchTST and Graph WaveNet on METR-LA and PeMS-Bay. Mid report Mon Nov 30, final Tue Dec 22.

## The course repository (`dlstg-course`): plan, notes and theory

The plan, the research notes and the course material live in a second GitHub repository,
`https://github.com/hafeez381/dlstg-course` (private; all three of us have write access). Clone it once,
next to this repository, for example `git clone https://github.com/hafeez381/dlstg-course.git ~/dlstg-course`.
If its local path is not known, ask the user once and remember it for the session. Everything below is
relative to that clone. `git pull` it before reading or editing anything in it.

**It is an Obsidian vault.** The files are Obsidian-flavoured Markdown, written to be read in Obsidian
(open the clone with "Open folder as vault"). That means wikilinks `[[Note#Heading|text]]`, block links
`[[#^decision-3]]`, embeds `![[image.png]]`, callouts `> [!info]`, YAML frontmatter, and MathJax math
`$...$`. Keep that syntax when editing; never convert it to plain Markdown links. File names contain
an em dash (`DLSTG Project — Implementation Guide.md`): quote paths in the shell.

| Path in `dlstg-course` | What it is | How to use it |
|---|---|---|
| `DLSTG Project/DLSTG Project — Implementation Guide.md` | **The plan.** Tasks C0–C23, each with owner, wave, **Needs**, decisions used, steps (checkboxes), verification checks and pitfalls. §0.1 owners, §0.2 dates, §0.5 change log, §0.6 RQs, §0.7 GPU jobs, §0.8 waves and hand-offs, §0.9 proposed runs, §2 notation and equations, §4 design decisions, §5 config, §6.2 file interfaces | Read the whole task before coding and follow it. Its checks are the definition of done. Where the code must differ from the guide, change the guide too (step below) |
| `DLSTG Project/DLSTG Project — Project Notes.md` | **The research notes**: question, hypotheses, theory per measurement, findings, open questions, proposals, self-check. No code | Patch it with the `dlstg-project-notes` skill after every task or result |
| `DLSTG Project/DLSTG Project — Background Primer.md`, `DLSTG Project — Research Directions.md` | Why each method works; literature, hypotheses, measurement toolkit | Theory for the project; link to their sections instead of repeating them |
| `My Notes/DLSTG_Lecture_<n> - <title>.md`, `DLSTG_Paper - Graph WaveNet.md` | The course's lecture notes (the user's) | **Theory comes from here first**: cite the exact heading and quote the lecture verbatim |
| `DLSTG Project/papers/*.pdf`, `Instructor Notes/` | Project papers; instructor slides and readings | Read for anything the lecture notes do not cover |
| `DLSTG Project/Proposal Concise/`, `proposal/` | The submitted proposal | The source of `[FIXED — P]` values in the guide |
| `attachments/` | Images used by the notes | See "Images" below |

**Following the guide.**
- A task's ID (C3, C6a) never changes; cite it in branch names (`feat/c13-gwn`), PR titles and commits.
- Values tagged `[FIXED — B]` (course brief) or `[FIXED — P]` (submitted proposal) are not ours to change.
  `[YOURS]` values and design decisions can change, but only with a change-log row (date, change, why, who).
- File formats between people are the interfaces of §6.2. Do not change one without telling the team.
- When a task is done: tick its checkboxes, and if a date, decision, check or interface changed, edit that
  place in the guide and add a change-log row. Commit with `docs(project): <summary>` and push the same day.

**Theory.** For any concept a task or a note needs, look in `My Notes/` first. List a note's headings with
`grep -n "^#" "My Notes/<file>.md"` and link the exact heading (Obsidian drops colons from heading links):
`[[DLSTG_Lecture_7 - Graph Neural Networks#6.4 GCN versus GAT — a dynamic adjacency|L7 §6.4]]`. Quote the
lecture's own sentence or formula. Then the Background Primer and Research Directions, then the papers.

**Links that will not open in your clone.** Some notes link to the user's private notes outside this
repository (`Assignments/…`, `CS6304 ATML/…`) and embed images by the user's vault path
(`![[03 - Attachments/<name>]]`; copies of those images are in `attachments/`). Leave those links as they
are: they resolve in the user's vault. Never create the missing files or rewrite those links.

**Images.** To add a figure to a note, save it in `attachments/` and embed it by bare file name,
`![[DLSTG_<topic>_<name>.png]]`, which resolves both in a clone and in the user's vault.

**Checking an edit.** Run `python3 -I .claude/skills/dlstg-project-notes/scripts/linkcheck.py "<note>.md"`
(from this repository, on the file in your `dlstg-course` clone). It reports broken links, headings, block
ids, uneven tables and unbalanced math. In a clone it will also list the private links above; only new
problems need fixing.

## Before starting a task
1. `git pull` in both repos.
2. Read the task in the guide: its **Needs** line (do not start until those exist), its decisions, its checks.
3. Read the guide's §0.9 and say whether any "Run it if" trigger has fired.

## Working rules
- `main` is protected. One branch per task (`feat/c13-gwn`), a pull request, merged with a **merge commit**
  (`gh pr merge --merge`), never a rebase or squash: results record the commit sha in `meta.json`.
  GitHub deletes the branch on merge; then run `git switch main && git pull && git branch -d <branch>` locally.
- Commit messages: one Conventional-Commits line, `type(scope): summary`. `git add` explicit paths only.
- Commit the code before the results it produced, then regenerate results at that commit.
- Every number comes from a file: models write `results/<run>/{split}_preds.npz` and only
  `tgg.eval.harness.score()` turns them into metrics. Never compute report numbers inside a training script.
- Every script runs `--mode smoke` locally before any GPU job. GPU jobs run on the owner's Kaggle account via
  `kaggle/build_code.sh` and `kaggle/push_kernel.sh <job>`; rules in `docs/compute.md`.
- Every checkpoint and log keeps a sha256-verified copy off the VM (`tgg.ckpt.push`). Ask the user before
  deleting or overwriting anything: files, Kaggle kernels or datasets, Hugging Face repos.
- Adapter runs and analyses use code at or after the `v-prereg` tag; `configs/prereg.yaml` changes after the
  tag go in the guide's change log with a reason.
- Data: `python -m scripts.build_data` then `python -m scripts.build_populations`; the populations file's
  sha256 must equal the one in `configs/prereg.yaml`.

## After every task, training run or result — required
Invoke the **`dlstg-project-notes`** skill (in `.claude/skills/`). It writes the findings into the project
notes, updates guide §0.9 and the change log, and commits and pushes `dlstg-course`. Also tick the task's
checkboxes in the guide and log any change to a decision, date or interface. Do this in the same session,
before reporting the task as done; ideas and decisions go into the notes or guide, not only into chat.
