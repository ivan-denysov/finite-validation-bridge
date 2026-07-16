# finite-validation-bridge

Simulation code and numerical protocols for:

**I. Denysov, *Physics from Finite Validation: The Bridge — How a Validation Network Becomes a Field*, United Field Initiative, 2026 (v3, July 2026).** DOI (all versions): https://doi.org/10.5281/zenodo.20664784

Series: [Paper 1](https://doi.org/10.5281/zenodo.20604820) · [Paper 2](https://doi.org/10.5281/zenodo.20634289) · [Paper 3](https://doi.org/10.5281/zenodo.20640510) · [Paper 4](https://doi.org/10.5281/zenodo.20657842) · this work (Paper 5, the Bridge). *(concept DOIs — always resolve to the latest version)*

This work closes the programme's load-bearing open debt — that its two model regimes (a Kuramoto-type phase network and a sine-Gordon field) are one object — in the linear and solitonic regime; (v2) lifts the bridge to three dimensions, measures the boundary of the 3D scalar sector, states a set of dynamics postulates for the substrate, and exhibits a self-bound flow condensate; and (v3, §6d) reads the two branches of the 3D dispersion as **two sectors of one network** — a massless, symmetry-protected Goldstone (gravity) and a massive Klein–Gordon field (matter) — with a graded set of in-model checks. Every result carries a status label (confirmed-in-model / observation / lesson / closed-negative / open); the full nonlinear reduction, the tensor-sector coefficient, the c_T = c match (GW170817), and the map to the Einstein equations are named as analytic debts.

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

**Two sectors / Goldstone protection (v3, July, §6d):**
`goldstone_*.py`, `dilute_pin_scan.py`, `transverse_phonon_v1.py`, `bond_bending_v2.py`
— the massless (Goldstone) vs massive (Klein–Gordon) branches, what matter is, and
the tensor (spin-2) sector on the amorphous substrate.

## Quick start

```
python net3d_build.py            # amorphous 3D substrate -> net3d.npz
python net3d_bridge3d.py         # 3D dispersion (fig_dispersion)
python net3d_soliton_finish.py A # condensate, low density  (fig_condensate)
python goldstone_dispersion_v1.py   # massless vs massive branch (1D)
python bond_bending_v2.py           # tensor sector on the amorphous substrate
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
| `net3d_bridge3d.py` | dispersion ω²=0.986+0.00067k², axis isotropy 0.38%; also the m=0 massless run |

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

Two sectors / Goldstone protection (companion Block J, §6d) [v3]:

| script | role | status |
|---|---|---|
| `goldstone_dispersion_v1.py` | 1D massless (g=0) vs massive (g>0) branch | confirmed in-model |
| `goldstone_symmetry_v1.py` | masslessness is symmetry-protected (only absolute-phase terms gap it) | confirmed in-model |
| `goldstone_3d_v1.py` | 3D dispersion + exact uniform-mode zero (lattice-independent) | confirmed in-model |
| `dilute_pin_scan.py` | scattered pins would screen gravity √(gn) ⇒ matter couples via Δω, not pins | confirmed in-model |
| `transverse_phonon_v1.py` | transverse (spin-2) phonons on a rigid Z=18 stand-in, c_T<c_L | confirmed in-model |
| `bond_bending_v2.py` | bond-bending (B5) rigidifies the amorphous Z̄≈5.5 substrate; floppy = Maxwell + self-stress; c_T/c_L=0.57–0.71 same substrate | confirmed in-model (reduced N) |

Figures:

| script | output |
|---|---|
| `gen_figure_data.py`, `gen_disp.py`, `make_figs.py` | `fig_condensate.*`, `fig_dispersion.*` |

## Key numbers (seed = 1)

- substrate: Z̄ = 5.479, edge anisotropy 0.0029, giant component 99.3%
- 3D dispersion: ω² = 0.986 + 0.00067 k², R² = 0.94, axis isotropy 0.38%
- affine bound c²_aff = 0.001647; measured c²/c²_aff ≈ 0.29–0.47 (k-systematic, node-scale dispersion)
- condensate: straightness plateau 0.294–0.295 over t = 10000–20000; blocked ≈ 0.8; open-arc control self-sustains
- Goldstone: massless branch gapless (1D & 3D), exact uniform-mode zero; on-site term reopens gap √g
- tensor sector: amorphous Z̄=5.59 floppy (296 modes = Maxwell 140 + 156 self-stress); bond-bending k⊥ restores shear, c_T/c_L = 0.57–0.71 (reduced N=700)

## License

Code: MIT. Figures: CC BY 4.0. © United Field Initiative.

## Research Programme

One output of a broader, staged research programme of the United Field Initiative
on cumulative complexity and validation dynamics. Details: ufi.observer/research-programme.
