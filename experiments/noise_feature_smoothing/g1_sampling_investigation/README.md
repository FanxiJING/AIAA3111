# G1 and joint sampling-speed follow-up

Read [FINDINGS.md](FINDINGS.md) for the Chinese findings draft and caveats.
This package only analyzes frozen artifacts. It performs no fitting, noise
generation, tuning, new user resampling, smoothing or protocol change.

Run from the repository root:

```powershell
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/g1_sampling_investigation/run_analysis.py'
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/g1_sampling_investigation/verify_analysis.py'
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/g1_sampling_investigation/build_findings.py'
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/g1_sampling_investigation/verify_delivery.py'
```

The runner's preservation manifest is limited to the documented protected
scope. It does not hash virtual environments or every file on the computer.
The verifier deliberately does not import the analysis runner.

- [Analysis specification](ANALYSIS_SPEC.md)
- `membership.parquet` records fixed per-window membership in the full local run and can be regenerated with `run_analysis.py`.
- [All numerical tables](tables)
- [Runner acceptance](acceptance.json)
- [Independent verification](independent_verification.json)
- [Delivery verification](delivery_verification.json)
- [Input inventory and runtime versions](input_manifest.json)

`joint_summary.csv` includes all 20 cells, empty cells and all nine groups.
Class support/prediction/TP columns in the per-seed table retain exact integers.
`joint_class_metrics_by_seed.csv` is derived by the report builder; conditional
recall is missing where true-class support is zero. Fixed-five-class F1 uses
zero for a zero denominator, matching the original convention.

All reported SDs refer to the five fixed noise seeds. No new confidence
intervals are computed. No manual figure QA is claimed.
