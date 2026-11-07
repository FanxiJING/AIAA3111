# AIGC log – Part 2 (basic model training)

Working notes for the report's Disclosures section. The team must verify all content before using it.

**Tool:** Claude Code (Anthropic), model Claude Opus 5.5.
**Sessions:** 2026-11-06 to 2026-11-08.

## Decisions made by the student
- Scope of each step, and when to commit.
- Selecting the best model by CV macro-F1 instead of test score.
- Turning off HGB early stopping and using a fixed `max_iter` of 300.
- Running the `dt_mean`/`dt_max` ablation, then excluding those features by default.
- Adding user-level bootstrap CIs, CV fold spread and per-user accuracy.
- Adding temporal smoothing as the final step, choosing k on blind-cut training windows, and later the
  extended k grid with a one-SE rule.
- Choosing the paired one-SE rule after the AI tool showed that the unpaired rule picked k = 1.
- Keeping the SVM as the headline model after smoothing was added.
- Asking for the segment output, segment metrics, handover note, figures and report draft.

## Changes made by the AI tool (in `src/model_training`, at the student's request)

**Session 1 (2026-11-06)**
- `models.py`: added `select_best`, `user_bootstrap`, `per_user_accuracy` and the default feature exclusion; added CV fold spread; changed the HGB settings.
- `train_basic.py`: selection by CV; new output files; `--keep-dt` and `--n-boot`; separate default output folders.
- `build_windows.py`: fixed a crash under pandas 3 by loading plain strings instead of Arrow-backed ones and storing the event-rate flags as floats; corrected a docstring path.
- `README.md`, `CLAUDE.md`: updated paths, method notes and project status.

**Sessions 2–3 (2026-11-07 to 2026-11-08)**
- New `smoothing.py`: sliding majority vote within blocks, mode-change and isolated-flip counts.
- New `segments.py`: segment output and segment metrics (IoU match, transition precision/recall).
- New `tune_smoothing.py`: out-of-fold predictions on blind training windows; scores for each k; the paired one-SE rule.
- New `evaluate_smoothing.py`: applies the fixed k to test; CIs, segments and segment metrics.
- New `make_figures.py`: five report/poster figures and their captions.
- `requirements.txt` pinned to the versions used; `README.md` (Part 2 and top level) and `CLAUDE.md` updated.
- New `docs/part2_handover.md` and `docs/part2_report_draft.md` (a **draft** for the team to rewrite).

The tool also proposed:
- the selection fix
- the separate `--no-tune` output folder
- the paired bootstrap design
- the paired version of the one-SE rule
- the rule for choosing the example trip

The earlier Part 2 code was written before these sessions; this log does not cover it.

## Commands run by the AI tool on the real data
- `build_windows.py`: window table for W = 50. Point totals matched Part 1.
- `build_windows.py --no-pure-train --out windows_w50_blind.csv`: blind windows for all users.
- `train_basic.py --no-tune`: smoke test.
- `train_basic.py` before the default change: with-dt run, saved to `results/basic_w50`.
- `train_basic.py --drop-features dt_mean dt_max --out results\basic_w50_no_dt`: ablation.
- `train_basic.py` after the default change: headline run, saved to `results/basic_w50_no_dt`. The first attempt was stopped by the system for low memory; the rerun with 8 workers reproduced the ablation's scores exactly.
- `tune_smoothing.py` (refit once; later `--reuse-oof` for the extended grid and the paired rule).
- `evaluate_smoothing.py`, then `make_figures.py`.

Test predictions were first smoothed with k = 9 under the earlier argmax rule on k ∈ {1, …, 9}. The
student then asked for a larger grid and the one-SE rule. k was finally fixed from training
out-of-fold predictions only, before any test segment metric was computed. The final SVM value is
again k = 9.

## Analysis by the AI tool
- Checked point totals against Part 1 and looked for data problems such as the two train users with too few points.
- Summarised model comparison, confusion matrix, CV-vs-test gap, ablation, bootstrap, smoothing and segment results in chat.
- Drafted the Methodology/Experiments text and failure analysis in `docs/part2_report_draft.md`.
- Listed what is still missing against the course requirements.

## Not done by the AI tool
- No final report or poster text; the draft is marked for rewriting.
- No references were invented. The draft cites only sources named in the repository docs and marks the rest **[citation needed]**.
- Nothing has been pushed to GitHub.

## To complete before submission
- Rewrite the draft in the team's own words and verify every number against the result files.
- Add any later AI use, including for report or poster writing.
- Name any external collaborators, or state that there were none.
- Add teammates' AI use for their own parts.
