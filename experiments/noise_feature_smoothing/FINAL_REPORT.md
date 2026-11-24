# GeoLife noise / feature / smoothing execution report

All six stages passed their implemented acceptance checks.
Reference results were read only; all new artifacts are under this directory.

## Fixed protocol and interpretation

Added noise is independent Gaussian E/N noise, not a calibrated model of real GPS temporal errors. Noise settings are project-defined; Endo's noise study used KT. Smoothing averages decision scores, not GPS coordinates.
Sigma=0 means no added noise, not true coordinates. Clean training, noisy test.
Decision scores are not calibrated probabilities. Centered smoothing uses future windows; trailing smoothing still requires completed windows and is not a full real-time system.
Seeds describe perturbation variability; bootstrap CIs use 9 linked user groups.
Largest linked group contributes 78.8832% of test points.
Transition truth uses window majority labels (529 boundaries), losing some point-level changes.
Delta is noisy minus own clean score; negative values are degradation. Always inspect absolute scores.
Examples use deterministic error selection and are explanatory, not representative frequencies.

## Results: raw Macro-F1

- G1 sigma=0m: 0.6983 +/- 0.0000 seed SD; user CI [0.2785, 0.7135]; delta +0.0000.
- G1 sigma=5m: 0.2737 +/- 0.0011 seed SD; user CI [0.2043, 0.3099]; delta -0.4246.
- G1 sigma=10m: 0.2121 +/- 0.0021 seed SD; user CI [0.0746, 0.2654]; delta -0.4862.
- G1 sigma=20m: 0.1274 +/- 0.0011 seed SD; user CI [0.0598, 0.3021]; delta -0.5709.
- G2 sigma=0m: 0.7163 +/- 0.0000 seed SD; user CI [0.2987, 0.7331]; delta +0.0000.
- G2 sigma=5m: 0.2228 +/- 0.0015 seed SD; user CI [0.1020, 0.2317]; delta -0.4935.
- G2 sigma=10m: 0.1763 +/- 0.0006 seed SD; user CI [0.0354, 0.1837]; delta -0.5400.
- G2 sigma=20m: 0.1539 +/- 0.0016 seed SD; user CI [0.0732, 0.2277]; delta -0.5624.
- G3 sigma=0m: 0.7788 +/- 0.0000 seed SD; user CI [0.3502, 0.8001]; delta +0.0000.
- G3 sigma=5m: 0.2245 +/- 0.0031 seed SD; user CI [0.1675, 0.2941]; delta -0.5543.
- G3 sigma=10m: 0.1629 +/- 0.0009 seed SD; user CI [0.1143, 0.3732]; delta -0.6159.
- G3 sigma=20m: 0.1370 +/- 0.0009 seed SD; user CI [0.0898, 0.3640]; delta -0.6418.
