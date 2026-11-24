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

## 2026-11-08/09 – Part 2 v2 (hardening)
- Claude (chat) reviewed Part 2 against the course guide and listed methodological risks (selection rule vs final pipeline, mixed k, unequal SVM subsample, validation-window cutting, bootstrap unit).
- Claude Code wrote `pipeline_v2.py`, `notes_v2.py`, `make_figures_v2.py` and soft voting in `smoothing.py`, from a prompt with pre-declared adoption and selection rules. The student ran all stages on the real data.
- Claude (chat) read the result files, found and fixed a bug in the smoothing-gain CI (commit `517bda2`), and wrote `docs/part2_v2_findings.md` and `docs/part2_notes_for_teammates.md` as notes. No report prose, no references.

## 2026-11-19 – Local GPS downsampling experiment preparation
- Codex checked GitHub branch heads and audited the committed v2 artifacts; reproduced selected-model F1 and accuracy from the saved confusion matrix. This was an artifact audit, not a new training run.
- The user clarified their scope: GPS downsampling first, then fixed-point versus fixed-duration windows.
- Codex added configurable pipeline paths, timestamp-based sampling with recomputed motion features, point/time window preparation, behavior tests and a local experiment protocol.
- The local checkout lacks real point data and window bundles. No real downsampling condition has been trained or evaluated in this session; synthetic inputs are used only by tests.

## 2026-11-20 – Real-data sampling experiments
- The teammate supplied quality_features (69 Parquet files, 64 nonempty users, 4,347,764 points) and the reference window bundle. Codex rebuilt the original-sampling windows and verified that all window keys and 27 model features match the supplied reference exactly.
- Codex created a project-local experiment environment with the reference dependency pins, fixing a PyArrow 19 reader incompatibility with the supplied files. All formal conditions share the same environment and frozen training-user folds.
- Before viewing any new real test results, Codex stopped an expensive full C/weight/voting pilot and recorded a protocol amendment: freeze the existing C=1/balanced/soft reference model and select only smoothing k on training-user CV. Pilot logs are preserved and excluded from formal results.
- Codex implemented checkpointed CV, test-point membership maps, common-support paired bootstrap, window-scheme comparison and plotting scripts. Formal CV covers four fixed-point conditions and three feasible fixed-duration conditions. The 30-second sampling / 120-second window combination has no training windows with at least 10 points and is recorded as infeasible.
- Synthetic smoke checks verify interfaces and checkpoint recovery only; their scores are never included in real experiment tables.
- All seven formal conditions completed training and test evaluation. Codex independently reconstructed confusion matrices and scores from saved predictions and window counts (78/78 checks passed), generated paired common-support/window-scheme comparisons and PNG/SVG figures, and wrote the local findings and shareable Chinese brief. No generated formal results have been pushed to GitHub.

## 2026-11-20/24 – Full-coverage window follow-up
- At the user's request, Codex implemented a separate original-GPS comparison of 50-point and 120-second windows, retaining all positive-length training windows to match all 51 training users and 3,486,792 points. Model settings and a common centered 20-minute point-score smoother were fixed before this follow-up evaluation.
- The experiment paused at the user's request after five folds for 50 points and four folds for 120 seconds. On November 24, Codex resumed the remaining fold, froze both CV records, then completed both final fits and tests without changing the prepared training source or inputs.
- Codex generated paired group-bootstrap comparisons, per-user/per-class tables and PNG/SVG figures. All 18 unit tests passed with PYTHONPATH=src; 30 independent artifact checks passed, including a pandas rolling reconstruction of every test smoothing prediction and frozen environment/source checks.
- Codex drafted Chinese findings, a shareable brief and suggested English poster wording. No clear window advantage was detected; the findings explicitly retain the small-group, reused-test, offline-smoothing and non-equivalence limitations. The student should verify the figures and rewrite submitted prose in the team's own words. No outputs were pushed to GitHub.

## 2026-11-24 – Default-branch integration
- At the user's explicit request, Codex fetched the latest remote heads, recovered the shallow history, and confirmed main is an ancestor of part2-fixes. The integration preserves all original Part 1/2 commits and their authors.
- Codex added the local experiment code/tests/documentation and a lightweight results package with hashes, tables and poster figures. Raw GPS, virtual environments, large models and intermediate point tables remain local. No existing teammate files were deleted; only the documented pipeline path/fold options and project documentation were extended.
- All 18 tests passed in the integration checkout. The default branch is updated using a normal fast-forward push, without force or history rewriting.
