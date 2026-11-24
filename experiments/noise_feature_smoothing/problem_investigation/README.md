# Noise Problem Investigation

This is a post-hoc diagnostic package dated 2026-11-23. It does not replace
the frozen experiment, constitute independent user validation, or introduce
a tuned model.

- [Chinese findings and limitations](INVESTIGATION_REPORT.md)
- [Diagnostic scope fixed before these calculations](ANALYSIS_SPEC.md)
- [Primary sources and actual reading limits](SOURCES.md)
- [Runner acceptance and preservation checks](acceptance.json)
- [Independent numerical verification](independent_verification.json)
- [Known-answer checks](known_answer_tests.json)

The analysis covers 31 G3 prediction conditions and 93 postprocessed sequences.
The independent verifier checked 341 group confusion calculations, 93 primary
boundary counts, 1,024 binary-sequence pairs, and 755 protected file hashes.
Verification does not establish that every line of the project is correct.

Run from the repository root:

```powershell
.venv-audit/Scripts/python.exe -B experiments/noise_feature_smoothing/problem_investigation/investigate.py
.venv-audit/Scripts/python.exe -B experiments/noise_feature_smoothing/problem_investigation/verify.py
```

No retraining, new perturbation fields, new bootstrap draws, or changes to
the original features and evaluation are part of this package.
