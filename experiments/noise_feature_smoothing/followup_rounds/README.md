# Three Follow-up Rounds

Three finite rounds of post-hoc diagnostics using existing frozen model,
feature-cache and prediction artifacts. No fitting, new noise, tuning or new
bootstrap draws. The original protocol and previous outputs are preserved.

- [Chinese findings draft](FINDINGS.md)
- [Scope fixed before these calculations](ANALYSIS_SPEC.md)
- [Figure captions and interpretation limits](FIGURE_CAPTIONS.md)
- [Round checkpoint](checkpoint.json)
- [Run acceptance and protected-file inventory check](acceptance.json)
- [Independent numeric verification](independent_verification.json)
- [Delivery, report claims and preservation checks](delivery_verification.json)
- [Original input manifest](input_manifest.json)
- [Model versions, feature order and intercept preferences](model_configs.json)
- [Automatic figure checks](figure_checks.json)

## Reproduction

From the repository root, using the existing audit environment:

```powershell
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/followup_rounds/run_rounds.py'
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/followup_rounds/verify_rounds.py'
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/followup_rounds/verify_delivery.py'
```

The runner checks the protected input inventory against the saved manifest.
It replaces outputs only inside this diagnostic directory; archive this directory
first if retaining its exact run timestamps is necessary.
`--start-round 2` or `--start-round 3` resumes only after prior acceptance files
exist. A failed attempt stops for diagnosis before a retry; inspect the failure
log rather than blindly repeating the command.

## Output Map

| Artifact | Meaning |
|---|---|
| `tables/raw_summary.csv` | Absolute raw F1/accuracy, seed SD, group-bootstrap intervals, positive clean-minus-noisy loss |
| `tables/paired_feature_contrasts.csv` | All 24 feature-pair comparisons and their change from clean |
| `tables/omitted_group_contrasts.csv` | 240 full/leave-one-linked-group-out comparisons; no refitting |
| `tables/kernel_summary.csv` | Kernel similarity, intercept concentration, score norm and distances |
| `tables/distance_buckets_summary.csv` | Signed geometric contributions at a fixed clean landmark |
| `tables/point_error_flows_summary.csv` | True-class damaged/rescued counts, averaged over noise seeds |
| `tables/prediction_pair_flows_by_seed.csv` | Full clean/noisy predicted-pair flows with true-class counts |
| `tables/*_by_seed.csv` | Unaggregated condition diagnostics |
| `diagnostics/*.npz` | Per-window numeric diagnostics and nine-group confusion matrices |
| `figures/` | Three PNG/PDF figure pairs |

`original` uses gamma=1/feature dimension; `fixed_gamma` identifies the two
previously fitted G1/G2 controls. The `common_gamma` comparison reuses original
G3, whose gamma already equals 1/27. It is not an additional G3 fit.
Kernel/distance medians are window based. Prediction shares and classification
metrics are point based. Seed SD and across-group bootstrap uncertainty answer
different questions.

Automated figure layout/nonblank checks passed. Manual image inspection remains
pending because the image tool was unavailable. All report prose is AI-assisted
draft material for student verification, rewriting and final citation checks.
