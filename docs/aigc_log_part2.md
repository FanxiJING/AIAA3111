# AIGC log – Part 2 (basic model training)

Working notes for the report's Disclosures section. The team must verify all content before using it.

Entries below are chronological: the November 6-8 workflow is historical v1, followed by v2 and
local documentation work. Historical selection decisions are not instructions for the current run.

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

## 2026-11-23 - Documentation Alignment To The v2 Handover

**Tool:** Codex (OpenAI), at the student's request to follow `docs/part2_handover.md`.

- Read the handover, current implementation and delivered v2 tables; updated the root README,
  training README, agent context (`CLAUDE.md`) and report draft to reference v2 rather than v1.
- Replaced stale report tables/figure paths with the delivered v2 results; distinguished five-fold
  CV SE from SD, class decision scores from probabilities, and centred offline smoothing from
  coordinate denoising and causal score smoothing.
- Clarified linked-group paired bootstrap, present-class user/bootstrap F1, report-only all-user
  CV, and the distinction between legacy transition metrics and noise-study one-to-one matching.
- Documented separate Part 1/Part 2 environments, fixed-path overwrite risks and existence-only
  resume behaviour. Distinguished the separately reproduced noise-study model from the original
  v2 fitted artifact.
- Corrected the description of `acc_std` to signed-acceleration std and disclosed the inherited
  path-length/duration support mismatch. **No feature implementation was changed or fixed.**
- No experiment design, Python code, data, splits, seeds, model settings or result artifacts were
  changed. No training or literature search was performed for this documentation alignment.
- The rewritten report draft remains AI-assisted material requiring human verification, rewriting
  and a verified bibliography before submission; no new references were invented.
- Verification: a read-only check passed 132 numeric/artifact comparisons and checked 17 local
  Markdown links across the five edited documents. No training or unit tests were rerun.
  All 476 protected non-Markdown files under `src`, `tests`, `data` and `experiments` (excluding
  bytecode/cache files) retained their pre-edit aggregate SHA-256:
  `3c69810f8b1ca41bb33bdde0394d6676b6ed522ed17cde1cc3f163b22acff526`.

## 2026-11-23 - Noise Study Literature Review And Descriptive Checks

**Tool:** Codex (OpenAI), following the student's request for research and further review.

- Read the fixed noise protocol, existing diagnostics and paired-analysis handover; performed a
  targeted primary-source search and archived metadata, abstracts, selected full-text sections,
  retrieval failures and SHA-256 records under
  `experiments/noise_feature_smoothing/literature_review/`.
- Cross-checked seven study identities, including a January 2026 journal record; explicitly
  distinguished metadata-only verification, abstracts and selected full-text reading.
  Rejected an incorrect candidate MS-TCN DOI after checking its title. No unsupported
  claim of a complete literature review or full-text reading was made.
- Added deterministic local-plane stationary-noise calculations and reconstructed the original
  test-point sampling intervals. Clarified that linear coordinate derivative formulas do not
  directly give variances of this project's scalar acceleration and jerk.
- Added five constant-class descriptive references, reconstructed 144 existing classification
  metric rows from saved predictions and true point counts, and saved 720 reference contrasts.
  These references are interpretive calculations, not newly fitted or selected classifiers.
- Drafted a Chinese evidence review, bibliography with reading limits, and reporting suggestions.
  Used the `scientific-critical-thinking` skill for evidence appraisal and included its requested
  software citation. The draft requires student verification and rewriting before submission.
- No retraining, noise generation, new random seeds, new bootstrap draws, parameter search,
  feature correction or changes to the experimental protocol. No original experiment,
  diagnostic or paired-analysis artifact was overwritten. Existing figure visual-QA limitations
  remain explicit. Check `literature_review/acceptance.json`, `tests.log` and
  `delivery_verification.json` for the executed numeric verification scope and outcomes.

## 2026-11-23 - Noise Study Reporting Revision And Independent Verification

**Tool:** Codex (OpenAI), following the student's request to address the review's shortcomings.

- Created a supplementary reporting package under
  `experiments/noise_feature_smoothing/report_revision/`: a Chinese report draft, a
  shortcomings/evidence checklist, 36-condition summary tables, and three revised figures
  in PNG and vector PDF. Earlier reports and numerical snapshots remain intact.
- Distinguished 51 fixed training users from the 49 contributing eligible fitting windows.
  Separated perturbation seed SD from paired user-group intervals and displayed classification,
  boundary precision/recall/F1, isolated flips and prediction concentration together.
- Mapped the original +/-1-window boundary tolerance into timestamp support for all 529
  true window boundaries. Counted 632 mixed-label windows containing 31,253 points and
  8,109 non-majority points, without changing the original truth or scoring rules.
  Reused the original deterministic cases with actual time widths and a clean-raw comparison.
- Independent verification caught a timestamp-unit bug in the new descriptive table:
  integer timestamps were assumed to be nanoseconds although parsed data used microseconds.
  Replaced the conversion with explicit timedelta seconds and added a regression test for
  both resolutions. Original experiment metrics were unaffected.
- Clarified the inherited zero-boundary convention in 133 of the original 5,000 bootstrap
  draws. No new draws were generated. Reporting CSV provenance paths are relative to the
  CSV directory and are checked for existence.
- Verification independently reconstructed 144 saved-prediction F1/Accuracy rows and 720
  class F1 values; checked 36 summaries and 529 boundary timestamp records. Nine focused
  known-answer tests passed. All 543 protected files retained their contents and membership.
  See `report_revision/delivery_verification.json` and `tests.log` for final executed checks.
- Automated checks cover three new figure layouts and 12 original/new nonblank PNGs.
  The image tool could not provide visual inspection; complete manual figure review remains
  pending and is not reported as passed.
- No retraining, new coordinate perturbations, new bootstrap draws, feature corrections,
  model selection or changes to the frozen protocol. The scientific-critical-thinking skill
  informed evidence interpretation; its requested software citation is included. All prose
  remains AI-assisted draft material for team verification, rewriting and final citation checks.

## 2026-11-23 - Exploratory Noise Extensions And Group Sensitivity

**Tool:** Codex (OpenAI), following the student's request to run additional experiments
and consider the remaining limitations.

- Fixed a separate supplemental protocol before inspecting its new outcomes, explicitly
  after the original test results had been seen. This is exploratory work, not preregistration
  or independent validation. Outputs are isolated under
  `experiments/noise_feature_smoothing/supplementary_experiments/`.
- Generated 15 additional OU position-noise feature caches (tau=30s, original five seeds,
  sigma5/10/20m). Used actual irregular time intervals, stationary block resets, paired
  innovations and spherical E/N displacement; recomputed all original motion features.
  Kept the original frozen G3, point/window identities, feature definitions and scoring.
  Zero identity/recomputation and poisoning of stale motion columns were checked.
- Fitted two clean-training gamma controls for G1/G2 with gamma=1/27 on all 70,086 eligible
  fitting windows, keeping the remaining recipe unchanged. Reused original G3 and all
  16 original iid caches. Saved 47 new five-class decision-score/prediction files.
- Scored all three original smoothers and completed nine linked-group omissions without
  refitting. Exported 288 metric rows including 144 reused original rows, 3,744 per-user rows,
  1,440 class-F1 rows, 72 summaries, 504 paired contrasts and 2,880 LOO rows.
  Reused the original 5,000 paired nine-group bootstrap draws, with seedwise scoring first.
- Added explanatory error-increment diagnostics, regenerating and hash-checking all
  65 user/seed OU fields. Separated added-error vector increment/dt from measured scalar
  speed error; did not infer scalar acceleration/jerk variance from coordinate formulas.
- Findings: G3 raw20m F1 was 0.1370 for iid and 0.3234 for OU30; the paired OU-minus-iid
  interval was [-0.0968, 0.2097]. Omitting the largest linked group reversed this
  point-estimate difference to -0.0507 and left only three true train-class points.
  Gamma control did not remove substantial degradation. OU20m centred smoothing raised
  F1 from 0.3234 to 0.3384 while boundary recall fell from 0.7134 to 0.2178.
  Reports retain these limitations and do not select a winning model from test results.
- The first reporting attempt failed a canvas assertion on out-of-view tick labels.
  Preserved the original runner/protocol and failure log; a separate reporting recovery
  checked only rendered ticks plus titles/labels/legends. No numeric result or experiment
  was changed. Four PNG/PDF figures passed automated layout/nonblank checks.
  The image tool could not provide inspection; complete manual visual review remains pending.
- Seven focused tests passed. A separate implementation independently reconstructed all
  288 metric rows, 1,440 class-F1 values, 72 summary/interval rows, 504 contrast rows and
  2,880 LOO rows, checked 90 blockwise smoothed sequences, and re-inferred 19 sampled
  windows in each of the 47 new score files. All 572 protected original files retained
  their contents; the present log was the explicitly excluded documentation update.
- Read selected GPML chapter4 passages for exponential/OU covariance and official
  Nystroem API definitions; archived the retrieved files and hashes. An initial web-tool
  citation failure was corrected to the user; successful direct retrieval is recorded.
  No claim of a new complete literature review is made.
- Used scientific-critical-thinking for evidence appraisal and the PDF skill for source
  extraction. Included the scientific skill's requested software citation in the protocol,
  reports and findings. These are AI-assisted drafts for student verification and rewriting.
  Artificial OU30 does not calibrate real GPS, remove group imbalance or provide new users.

## 2026-11-23 - Investigation Of Noise Failure Mechanisms

**Tool:** Codex (OpenAI), following the student's request to investigate the
remaining problems using research and existing results.

- Added an explicitly post-hoc diagnostic package under
  `experiments/noise_feature_smoothing/problem_investigation/`. Fixed its scope
  before the new calculations; no claim of preregistration or new independent
  user validation is made.
- Reconstructed all stored G3 decision scores for 31 clean/iid/OU conditions
  from the frozen RBF landmarks and linear classifier. Maximum absolute
  discrepancy was 8.07e-12. Also checked 64 evenly spaced windows per condition
  against the actual saved Pipeline. Examined training-standardized feature
  shifts, landmark similarities and convergence toward the train-class intercept.
- Decomposed the largest-group-omission reversal by class and linked group.
  The bike F1 contribution, including false positives from bus/car/walk,
  explains the negative Macro-F1 difference; accuracy moves in the opposite
  direction. These descriptive results do not establish a universal ranking.
- Measured the actual temporal support of 11-window smoothing, short-segment
  label presence and matched boundary offsets. Distinguished the frozen
  class-blind boundary metric from diagnostic label-pair compatibility.
  Compatibility depends on the selected maximum-cardinality/minimum-lag
  matching and is not an optimal typed boundary metric.
- Six groups of known-answer checks passed. A separate implementation checked
  341 group confusion calculations, 93 main boundary counts and 1,024 pairs
  of binary sequences. All 755 protected file hashes matched; the runner also
  verified protected membership. The diagnostic directory and this disclosure
  log were explicitly excluded from the preservation inventory.
- Read official scikit-learn API definitions and the CVF MS-TCN abstract.
  Revisited archived Ranacher sections and previously extracted GPML passages,
  with reading limits recorded. A new PMC response was a browser challenge,
  not a retrieved paper, despite HTTP status 200. The web search tool supplied
  no usable results; successful direct retrievals and hashes are recorded.
- Used scientific-critical-thinking for evidence appraisal and included its
  requested software citation. Added a Chinese investigation draft and
  reproduction index. Student verification, rewriting and final citation
  review remain necessary.
- No retraining, new noise fields, seed changes, tuning, new bootstrap draws,
  feature corrections or replacement of main results. No new figures were
  produced; earlier manual visual-review limitations remain unresolved.

## 2026-11-23 - Three Follow-up Diagnostic Rounds

**Tool:** Codex (OpenAI), following the student's instruction to continue rounds
of work on the noise study.

- Added an explicitly post-hoc diagnostic specification and a finite three-round
  runner under `experiments/noise_feature_smoothing/followup_rounds/`. This work
  reused existing models, iid caches, predictions and original 5,000 paired
  bootstrap draws. No fitting, tuning, new noise or new resampling was performed.
  The two common-gamma controls were fitted in the earlier supplemental study.
- Round 1 checked all 80 model-condition combinations: original G1/G2/G3 and
  existing common-gamma G1/G2 across 16 conditions. Reconstructed every stored
  decision score, with maximum absolute discrepancy 1.3555e-8, and confirmed all
  raw prediction labels. Also checked 5,120 windows against actual Pipelines.
  Added signed distance-change decomposition relative to each window's fixed
  clean-nearest landmark, with additive coordinate and bucket assertions.
- Round 2 exported 20 raw summaries, 24 paired feature contrasts, 240 full/group-
  omission contrasts, 400 true-class error-flow rows and 2,000 predicted-pair
  flow rows. Mixed-window point class counts were retained. Seedwise statistics
  and across-linked-group intervals remain distinct.
- Findings distinguish model mechanisms: at iid20m original G2 concentrates
  81.43% of point predictions on its walk-preferred intercept class; G3 predicts
  train for 85.65%. G1 retains substantially larger kernel similarity despite
  poor F1, so a common kernel-collapse explanation is insufficient.
  G3's acceleration and jerk/turn buckets comprise 97.76% of mean fixed-reference
  distance change, explicitly not causal attribution of F1 loss.
  G3 has mean 545,559.0 correct-to-wrong and 17,362.8 wrong-to-correct points.
- Recorded paired and group-omission uncertainty rather than choosing a winner:
  original G3-minus-G2 at20m is -0.0169, changing to +0.1369 after omission of
  linked0. These exploratory results do not provide new independent-user evidence.
- All three rounds passed without retries. Three PNG/PDF figure pairs passed
  automatic nonblank/text-in-canvas checks. The image tool could not provide
  visual inspection; manual figure QA remains pending and is not claimed passed.
- A separate verifier, without importing the runner, independently checked all
  80 conditions, 720 linked-group confusions against both new and previous
  archives, 400 class error flows, 5,120 Pipeline/distance samples, 20 bootstrap
  summaries, 24 paired contrasts and 240 group-omission contrasts. All 794
  protected input hashes matched; the runner also checked inventory membership.
- Added Chinese findings, a reproduction index and explicit figure captions.
  Scientific-critical-thinking informed interpretation; its software citation
  and primary-source reading limits are linked from the prior source record.
  No new literature search is claimed this round. All prose remains AI-assisted
  draft material for student verification, rewriting and final citation review.

## G1 And Joint Sampling-Speed Diagnostic Follow-up

**Tool:** Codex (OpenAI), following the student's request for further exploration.
Execution timestamps are recorded with the Asia/Shanghai offset in the package.

- Added an isolated `g1_sampling_investigation/` package under the noise-study
  output directory. Its specification was written before computing this round's
  new joint tables, while explicitly acknowledging previous test-set inspection.
  The work reused all 16 original iid caches and 48 G1/G2/G3 raw predictions;
  no training, tuning, new perturbations, bootstrap, smoothing or protocol changes.
- Jointly stratified by original sampling interval and clean mean speed,
  retaining all 20 cells, including empty cells and missing-feature singleton
  windows. Exported pooled and all nine linked-group tables, class composition,
  point confusion matrices, damage/rescue counts and conditional denominators.
  Added 4,800 derived class rows with supports, predicted totals, TP, recall and F1.
- Examined G1 nearest-landmark identities and their frozen-model responses,
  explicitly distinguishing a model response from training ground-truth labels.
  At 20m, high max-kernel similarity (>=0.9) covers mean 153,026 points but
  accuracy is only 8.27%. G1's point predictions concentrate on car (49.54%)
  and train (41.87%), so its failure cannot be explained solely by the
  kernel-collapse pattern observed for G2/G3.
- Documented a composition counterexample: in the <=1s, clean mean speed
  (0.5,2]m/s cell, G2 predicts every point as walk both clean and at 20m.
  Accuracy remains 92.11% because 24,252 of 26,329 points are walk; the
  fixed-five-class Macro-F1 is 0.1918. This is insufficient evidence of
  retained multiclass robustness.
- Fixed NaN propagation in this new package's raw-feature median summaries;
  added finite-value denominators and reran the complete diagnostic and
  independent verifier. All points and model-space imputed inputs were retained.
  This correction did not modify inherited features or prior results.
- The independent implementation verified 9,600 joint/group confusion and flow
  rows, 624 nonempty pooled weighted sklearn F1 calculations, 1,536 Pipeline/
  direct-distance samples, all 1,000 G1 landmark model responses, 400 response
  transitions and 76,000 summary values. All 910 protected existing files and
  inventory membership matched. A separate delivery verifier covers derived
  class metrics, report claims, links and final hashes.
- Used scientific-critical-thinking to constrain inference. The Chinese draft
  links its requested software citation and prior source-reading limits.
  Attempted official webpage reading yielded no usable content; no new paper
  reading or current literature verification is claimed. No new figures or
  manual visual QA. Findings remain exploratory, noncausal and based on the
  same nine linked groups; student review and rewriting remain necessary.

## Part 3 sampling, windows, and smoothing extension (2026-11-24)

**Tool:** Codex (OpenAI). The student requested a joint experiment for topics 1, 4, and 5.

- Implemented the joint study under `experiments/sampling_window_smoothing/` using the Part 2
  recipe, user split and linked groups. The new choices were 15- and 30-second minimum sampling
  intervals, 60- and 120-second time windows, and a 120-second switch matching tolerance.
- Used training-user grouped validation to select the window, fitting rate, and causal smoother
  under a 0.02 mean validation macro-F1 margin. Compared predictions on the original dense test
  stream and estimated paired uncertainty by linked test group.
- Added isolated caches, model and prediction artifacts, synthesis scripts, and tests for
  time-based windows, point alignment, causal score smoothing, and switch matching. The replay
  assumptions and fitting eligibility rules are in the study README.
- Completed 520 validation records, 104 test settings, and six unit tests. The synthesis also
  reports release-time classification coverage and a supplementary training-fold choice
  restricted to 120-second windows for thinned input.
- Applied the scientific-critical-thinking workflow to the design. Attempts to retrieve external
  sources returned no usable content. The experimental results come from the local run.
