# CLAUDE.md — tsfm-graph-gap (AI651 DLSTG group project)

Shared by the team (Abdul Hafeez, Huzaifa Abid, Israr Ali). Committed; change it through a pull request.

## What this repo is
Code for "Do time-series foundation models need a graph?": a frozen TSFM (Chronos-Bolt Small, TimesFM 2.5)
plus a small graph attention adapter on its cached outputs, compared with zero-shot, Chronos-2, full
fine-tuning, PatchTST and Graph WaveNet on METR-LA and PeMS-Bay. Mid report Mon Nov 30, final Tue Dec 22.

The plan lives in a second repository, `dlstg-course` (folder `DLSTG Project/`):
- `DLSTG Project — Implementation Guide.md`: tasks C0–C23 with owner, wave, Needs line, steps and checks;
  design decisions; interfaces (§6.2); change log (§0.5); proposed runs (§0.9). Source of truth for what to build.
- `DLSTG Project — Project Notes.md`: the research notes (findings, theory, open questions). No code.

If the local path of `dlstg-course` is not known, ask the user once and remember it for the session.

## Before starting a task
1. `git pull` in both repos.
2. Read the task in the guide: its **Needs** line (do not start until those exist), its decisions, its checks.
3. Read the guide's §0.9 and say whether any "Run it if" trigger has fired.

## Working rules
- `main` is protected. One branch per task (`feat/c13-gwn`), a pull request, merged with a **merge commit**
  (`gh pr merge --merge`), never a rebase or squash: results record the commit sha in `meta.json`.
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
