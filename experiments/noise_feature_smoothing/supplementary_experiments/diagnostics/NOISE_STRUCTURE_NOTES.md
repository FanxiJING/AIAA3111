# Noise structure diagnostic definitions

Valid edges match motion-feature time validity: same original block, 0<dt<=1200s.
No point is removed; this mask is only for error-increment diagnostics.

All 65 user/seed OU standard fields are independently regenerated and compared
with saved cache manifest hashes. Per-point iid innovations are regenerated from
the same original SeedSequence recipe. Values are not renormalised.

For a valid edge, per-axis additional-position-error increment variance is
2*sigma^2 for iid, and 2*sigma^2*(1-exp(-dt/30)) for OU30.
Vector RMS is sqrt(mean(E_increment^2 + N_increment^2)).
The per-dt variant divides each vector increment by that edge's actual seconds.
It is NOT the difference between noisy and clean scalar speeds, and its variance
does not give this project's scalar acceleration or jerk variance.

The original observed positions need not be true positions. These diagnostics
describe only the additional artificial error fields. They support an explanation
of why equal marginal position variance can produce different motion perturbation;
they do not establish a universal classification effect or a real GPS error model.
