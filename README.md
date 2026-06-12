# finite-validation-bridge

Simulation code and numerical protocols for:

**I. Denysov, *Physics from Finite Validation: The Bridge — How a Validation Network Becomes a Field*, United Field Initiative, 2026.**
DOI: [https://doi.org/10.5281/zenodo.20664785]

Series: [Paper 1](https://doi.org/10.5281/zenodo.20633926) · [Paper 2](https://doi.org/10.5281/zenodo.20634290) · [Paper 3](https://doi.org/10.5281/zenodo.20640511) · [Paper 4](https://doi.org/10.5281/zenodo.20657843) · this work (Paper 5).

This work closes the programme's load-bearing open debt: that its two model regimes — a Kuramoto-type phase network and a sine-Gordon field — are one object. The bridge is built in the linear and solitonic regime with a quantified domain of validity; the full nonlinear reduction is named as the one remaining analytic debt.

Requirements: Python 3.10+, numpy. `bridge_dde_truncation.py` additionally uses scipy (Lambert-W). Every script runs standalone; seeds and parameters are in the code.

---

# Numerical Companion: The Network–Field Bridge in Finite-Validation Models

*Ivan Denysov, Lead Researcher, United Field Initiative (UFI). Supplementary to: "Physics from Finite Validation: The Bridge" (this record). Code: https://github.com/ivan-denysov/finite-validation-bridge*

## Purpose and Status Discipline

This companion contains every numerical run supporting the main article, with full code, parameters, and per-result status labels. The discipline follows the series: confirmed in-model, observation, lesson/artefact (a result traced to the setup), open (analytic, pending). Runs are exploratory instruments within stated models, not measurements of nature. All scripts are standalone Python 3 (numpy; the delay-spectrum script uses scipy for the Lambert-W function).

## Block A — Inertia from Delay (main article §2)

- `bridge_inertia_from_delay.py` — a first-order Kuramoto-type network with *delayed* coupling, evaluated at θ(t−τ); no second derivative is written in. At τ = 0: no oscillations (clean relaxation — correct for a first-order network). At τ = 1, 2, 4: oscillations appear, period T = 3.20 / 6.06 / 10.53, ratio T/τ = 3.2 / 3.0 / 2.6 — period grows with delay, the signature of an inertial term born of the delay. The naive expansion gives an inertial term (τ²/2)F′·θ̈; the exact coefficient is in Block D. **Status: inertia from delay — confirmed in-model.**

## Block B — The Linear Bridge (main article §3)

- `bridge_dispersion.py`, `bridge_dispersion_v2.py` — dispersion from time-evolution FFT. High-k modes match the discrete sine-Gordon dispersion to <0.5%; low-k modes are resolution-limited near the gap (ω≈1) — a measurement artefact, not physics. Preserved as the route that motivated the exact method.
- `bridge_dispersion_v3.py` — the exact method: eigenvalues of the linearised operator M = c²L/a² − I. ω²_eig ≡ ω²_lattice = c²·(2/a²)(1−cos ka) + 1 to machine precision across all 33 modes, with ω² → c²k² + 1 as ka→0. **Status: the linearised phase network with inertia is identical to the *linearised* discrete sine-Gordon chain — confirmed in-model, exactly within the discretised formulation (the Frenkel–Kontorova continuum limit); the nonlinear identity is the §7 analytic debt.**

## Block C — The Kink (main article §4)

- `bridge_kink.py` — first attempt; the topological charge was measured at the array ends (th[-1]−th[0]) and corrupted by boundary effects (reported Q ≈ 0.16). Lesson: the charge must be the integrated phase winding, not an endpoint difference.
- `bridge_kink_v2.py` — corrected: Q = (1/2π)∮∂θ dx (the topological invariant), Neumann boundaries, np.sum·a (np.trapz is removed in current numpy). (i) Static kink: Q = 1.0000 → 1.0000, winding exactly 2π, centre fixed (Peierls–Nabarro does not tear it). (ii) Charge conserved through evolution. (iii) Moving kink: width/rest = 0.97 / 0.82 / 0.63 at v = 0.3 / 0.6 / 0.8 against √(1−v²) = 0.95 / 0.80 / 0.60; centre advances ∝ v. **Status: the network's soliton is a sine-Gordon kink in full — confirmed in-model.**

## Block D — The Delay Spectrum and Domain of Validity (main article §5)

- `bridge_dde_derivation.py` — test of the naive expansion's prediction (a fixed period ratio T/τ = π√2 independent of stiffness). Refuted: T/τ depends on the coupling K. Lesson: truncating the delay expansion at τ² loses the stiffness dependence; the exact characteristic equation is required.
- `bridge_dde_v2.py` — the exact characteristic equation λ = −S·e^(−λτ). The period depends only on the dimensionless p = Sτ. A Hopf threshold at p = π/2: below it the principal mode decays (Re λ < 0), above it it grows (Re λ > 0). **Observation: the finiteness of the tick sets a critical scale separating relaxation from sustained oscillation.**
- `bridge_dde_truncation.py` — the domain of the two-mode (sine-Gordon) reduction. Roots via Lambert-W branches λ_k = W_k(−Sτ)/τ; R = |Re λ₁ / Re λ₂| (small R = higher modes suppressed = clean sine-Gordon). R < 0.3 across p = 0.2…2.0, R = 0.000 exactly at the Hopf threshold p = π/2, R > 1 (reduction breaks) only beyond p ≈ 5. **Status: the two-mode sine-Gordon reduction is rigorous for Sτ ≲ 3, exact at p = π/2 — confirmed in-model as a quantified domain of validity. The full nonlinear reduction at arbitrary Sτ is an analytic debt, pending.**

## Block E — Isotropy from Amorphousness (main article §6)

- `bridge_3d_isotropy.py` — first attempt; three directions and one realisation are too noisy to separate cubic from amorphous. Lesson preserved.
- `bridge_3d_isotropy_v2.py` — coefficient of variation of ω²(k) over 24 Fibonacci-sphere directions, averaged over realisations. Long wavelengths: both isotropic. Short wavelengths (k = 1.0/1.5/2.0): cubic CV stays ≈ 0.04 (anisotropic — cubic axes), amorphous CV falls 0.037/0.034/0.023, advantage ×1.2/×1.3/×1.8. **Status: amorphousness suppresses the lattice's directional anisotropy, the more so the shorter the wavelength — confirmed in-model.**

## Reproducibility

Python 3.10+, numpy; Block D additionally scipy (Lambert-W). Every script runs standalone; seeds and parameters are in the code. Runtime: seconds to a few minutes per script.

## File → article map

| Script | Section | Status |
|---|---|---|
| bridge_inertia_from_delay.py | §2 | confirmed in-model (inertia from delay, T∝τ) |
| bridge_dispersion.py | §3 | lesson (FFT resolution-limited at low k) |
| bridge_dispersion_v2.py | §3 | intermediate (high-k match <0.5%) |
| bridge_dispersion_v3.py | §3 | confirmed in-model (ω²_eig ≡ ω²_lattice, exact) |
| bridge_kink.py | §4 | lesson (charge measured at ends — wrong) |
| bridge_kink_v2.py | §4 | confirmed in-model (Q=1, stable, Lorentz-contracts) |
| bridge_dde_derivation.py | §5 | lesson (naive τ² expansion refuted by numerics) |
| bridge_dde_v2.py | §5 | observation (Hopf threshold p=π/2) |
| bridge_dde_truncation.py | §5 | confirmed in-model (SG reduction rigorous for Sτ≲3) |
| bridge_3d_isotropy.py | §6 | lesson (too few directions/realisations) |
| bridge_3d_isotropy_v2.py | §6 | confirmed in-model (amorphousness → isotropy) |

## License
Code MIT; text and figures CC BY 4.0. © United Field Initiative.

## Research Programme
This document is one output of a broader, staged research programme of the United Field Initiative on cumulative complexity and validation dynamics. Details: ufi.observer/research-programme.
