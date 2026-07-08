# finite-validation-bridge
 
Simulation code and numerical protocols for:
 
**I. Denysov, *Physics from Finite Validation: The Bridge — How a Validation Network Becomes a Field*, United Field Initiative, 2026 (v2, July 2026).** DOI: https://doi.org/10.5281/zenodo.20664785
 
Series: [Paper 1](https://doi.org/10.5281/zenodo.20633926) · [Paper 2](https://doi.org/10.5281/zenodo.20634290) · [Paper 3](https://doi.org/10.5281/zenodo.20640511) · [Paper 4](https://doi.org/10.5281/zenodo.20657843) · this work (Paper 5, the Bridge).
 
This work closes the programme's load-bearing open debt — that its two model regimes (a Kuramoto-type phase network and a sine-Gordon field) are one object — in the linear and solitonic regime, and (v2) lifts the bridge to three dimensions, measures the boundary of the 3D scalar sector, states a set of dynamics postulates for the substrate, and exhibits a self-bound flow condensate from those postulates. Every result carries a status label (confirmed-in-model / observation / lesson / closed-negative / open); the full nonlinear reduction and several other gaps are named as analytic debts.
 
**Runs are exploratory instruments within stated models, not measurements of nature.**
 
## Requirements
 
```
python >= 3.10
numpy
scipy         # spatial (cKDTree, ConvexHull) for the net3d_* family; Lambert-W for bridge_dde_truncation.py
matplotlib    # figures only
```
 
```
pip install -r requirements.txt
```
 
Every script runs standalone; seeds and parameters are in the code. The 3D
substrate is built with `seed = 1` (`net3d_build.py` writes `net3d.npz`, which the
other `net3d_*` scripts load). Runtime: seconds to a few minutes per script; the
long condensate runs (`net3d_soliton_finish.py`, 20000 ticks) take a few minutes.
 
## Contents
 
**1D bridge (main article §§2–6), June:** `bridge_*.py` — inertia from delay,
linear dispersion, the kink, the delay spectrum, isotropy.
 
**3D bridge, scalar boundary, dynamics postulates, condensate (v2, July):**
`net3d_*.py` — substrate, 3D dispersion, scalar no-go, lock axiom, flow condensate;
`gen_*.py`, `make_figs.py` — figure generation; `fig_*.pdf/png` — the two v2 figures.
 
## Quick start
 
```
python net3d_build.py            # amorphous 3D substrate -> net3d.npz
python net3d_bridge3d.py         # 3D dispersion (fig_dispersion)
python net3d_soliton_finish.py A # condensate, low density  (fig_condensate)
python net3d_soliton_finish.py B # condensate, working density
python net3d_soliton_finish.py C # open-arc control
python make_figs.py              # rebuild both figures from logged series
```
 
## Script -> paper-section map
 
1D bridge (companion Blocks A–E):
 
| script | §  | status |
|---|---|---|
| `bridge_inertia_from_delay.py` | 2 | confirmed in-model (inertia from delay, T∝τ) |
| `bridge_dispersion.py` | 3 | lesson (FFT resolution-limited at low k) |
| `bridge_dispersion_v2.py` | 3 | intermediate (high-k match <0.5%) |
| `bridge_dispersion_v3.py` | 3 | confirmed in-model (ω²_eig ≡ ω²_lattice, exact) |
| `bridge_kink.py` | 4 | lesson (charge measured at ends — wrong) |
| `bridge_kink_v2.py` | 4 | confirmed in-model (Q=1, stable, Lorentz-contracts) |
| `bridge_dde_derivation.py` | 5 | lesson (naive τ² expansion refuted) |
| `bridge_dde_v2.py` | 5 | observation (Hopf threshold p=π/2) |
| `bridge_dde_truncation.py` | 5 | confirmed in-model (SG reduction rigorous for Sτ≲3) |
| `bridge_3d_isotropy.py` | 6 | lesson (too few directions) |
| `bridge_3d_isotropy_v2.py` | 6 | confirmed in-model (amorphousness → isotropy) |
 
3D bridge + substrate (companion Block F, §6):
 
| script | role |
|---|---|
| `net3d_build.py` | amorphous substrate N=8000, Z̄=5.48, isotropy → net3d.npz |
| `net3d_dyn.py` | discrete sine-Gordon, symplectic, energy drift −0.04% |
| `net3d_bridge3d.py` | dispersion ω²=0.986+0.00067k², axis isotropy 0.38% |
 
Scalar 3D boundary (companion Block G, §4):
 
| script | role |
|---|---|
| `net3d_soliton.py` | ball twist (dies) vs planar wall (stable) |
| `net3d_ring_core.py`, `net3d_ring_long.py` | vortex ring (collapses t≈50) |
| `net3d_unwind_check.py` | winding-number meter; line–antiline validation |
 
Vector point defect (companion Block G, §8 companion model):
 
| script | role |
|---|---|
| `net3d_hedgehog_single.py` | single hedgehog Q=+1, stable to t=480 |
| `net3d_hedgehog.py` | hedgehog–antihedgehog pair annihilation |
 
Dynamics postulates (companion Block H, §7):
 
| script | role |
|---|---|
| `net3d_lock_axiom.py`, `net3d_lock_metrics.py` | directed regime (A vs B control); Σ-dimerisation |
| `net3d_derived_penalty.py` | derived turn-delay D(α)=(α/2)/sin(α/2) |
| `net3d_walkers.py` | chains/channels from curvature |
| `net3d_dyncurv.py` | two-timescale curvature memory |
| `net3d_embed.py`, `net3d_embed_sweep.py` | dynamical embedding; ceiling 0.47→0.90 |
| `net3d_texture_locks.py` | texture on the fast variable (finite-lived) |
| `net3d_twist*.py` | boundary-twist transmission (Frank elasticity not reached — open) |
 
Flow condensate (companion Block I, §8):
 
| script | role |
|---|---|
| `net3d_loop_soliton.py`, `net3d_loop_v2.py`, `net3d_loop_v3.py` | seeding, binding-strength trials |
| `net3d_soliton_finish.py`, `net3d_sf.py` | final runs A/B/C (plateau 0.294, arc control) |
 
Figures:
 
| script | output |
|---|---|
| `gen_figure_data.py`, `gen_disp.py`, `make_figs.py` | `fig_condensate.*`, `fig_dispersion.*` |
 
## Key numbers (seed = 1)
 
- substrate: Z̄ = 5.479, edge anisotropy 0.0029, giant component 99.3%
- 3D dispersion: ω² = 0.986 + 0.00067 k², R² = 0.94, axis isotropy 0.38%
- affine bound c²_aff = 0.001647; measured c²/c²_aff ≈ 0.29–0.47 (k-systematic)
- condensate: straightness plateau 0.294–0.295 over t = 10000–20000; blocked ≈ 0.8; open-arc control self-sustains
## License
 
Code: MIT. Figures: CC BY 4.0. © United Field Initiative.
 
## Research Programme
 
One output of a broader, staged research programme of the United Field Initiative
on cumulative complexity and validation dynamics. Details: ufi.observer/research-programme.
