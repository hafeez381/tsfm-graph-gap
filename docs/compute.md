# Compute rules

Every training or caching job runs detached on a Kaggle T4 (or Colab for long jobs), never inside one long interactive call.

1. **Smoke first.** Every script has `--mode smoke` (synthetic data of the real shapes). It passes on a laptop before any job touches GPU quota.
2. **Code traces to a commit.** `kaggle/build_code.sh` refuses a dirty tree and packages `HEAD`; every kernel prints the commit it ran. Compare it with `git rev-parse HEAD` before trusting a result.
3. **Pick the GPU by memory and cost.** A T4 has 16 GB of GPU memory but only 12 GB of host RAM with no swap: load one model per process.
4. **Two copies of everything, one off the VM.** Every checkpoint, resume snapshot and training log is pushed to a private Hugging Face repository with `tgg.ckpt.push`, which raises unless the Hub's sha256 matches the local file. A file whose only copy is on the VM counts as lost. Never delete an older copy until its replacement is verified off the VM.
5. **Snapshots every ~25% of a run or every 30 minutes**, written to a temporary path and renamed, then pushed. Keep the full log off the VM, updated at every snapshot.
6. **Poll with short calls** (`kaggle kernels status`, a log tail). No single call stays open longer than about 4 minutes.
7. **When infrastructure fails mid-run** (quota, preemption, out of memory, auth), stop, write down what is safe and what is at risk, and agree the recovery with the team before spending more GPU hours or deleting anything.
8. **Never free storage by deleting backups.** If a Hub or Kaggle quota blocks a run, stop and ask the team where else to store.
