# -*- coding: utf-8 -*-
"""
IN-HOUSE PROBE (reviewer's load-bearing question): does a finite density of
symmetry-breaking PINS open a Goldstone gap that would screen gravity?

Context: the series has two 'mass' formalisms. In [4] matter mass is a
frequency shift d_omega that RESPECTS the phase-shift symmetry (a heavy node
sources the UNSCREENED delay field, [4] run 5.1). In §6d matter was
(over-)identified with an on-site PIN g*sin(theta) that BREAKS the symmetry.
If matter = pins, a vacuum with matter fraction n carries sparse pins, and by
Block J.2 those open a gap. If the collective-mode gap grows as sqrt(g*n),
gravity acquires a screening length in the presence of matter -- fatal to the
unscreened trough of [4].

Test: pin a random fraction n of sites (on-site g*sin) on the gapless chain,
measure the uniform-mode (Goldstone) frequency = the gap, scan n.
Prediction (analytic): the uniform component obeys
    d2/dt2 <theta> ~ -g*n*<theta>  =>  gap = sqrt(g*n).
So a FINITE pin fraction DOES gap the mode -> pins cannot be matter without
screening gravity. Contrast: d_omega is a symmetry-respecting SOURCE, not a
pin, and sources the field without gapping it.
"""
import numpy as np

N = 512; dx = 1.0; c = 1.0
dt = 0.02; STEPS = 120000; A = 1e-3; G = 0.05

def lap_sin(th):
    return (np.sin(np.roll(th, -1) - th) - np.sin(th - np.roll(th, 1))) / dx**2

def uniform_gap(n, seed=0):
    """Fraction n of random sites pinned; return the uniform-mode frequency."""
    rng = np.random.RandomState(seed)
    pin = (rng.uniform(size=N) < n).astype(float)      # pinned-site mask
    th = A * rng.standard_normal(N); v = np.zeros(N)
    M = np.empty(STEPS)
    for t in range(STEPS):
        v += dt * (c**2 * lap_sin(th) - G * pin * np.sin(th))
        th += dt * v
        M[t] = th.mean()                               # uniform-mode amplitude
    M -= M.mean()
    f = np.fft.rfftfreq(STEPS, d=dt)
    amp = np.abs(np.fft.rfft(M * np.hanning(STEPS)))
    return 2*np.pi*f[np.argmax(amp)], pin.mean()

print("DILUTE-PIN SCAN: Goldstone gap vs pinned fraction  (g = %.2f)\n" % G)
print(f"{'n_target':>9} {'n_actual':>9} {'gap_meas':>9} {'sqrt(g*n)':>10}")
for n in [0.0, 0.1, 0.25, 0.5, 1.0]:
    gap, na = uniform_gap(n)
    print(f"{n:>9.2f} {na:>9.3f} {gap:>9.4f} {np.sqrt(G*na):>10.4f}")
print()
print("Reading: if gap ~ sqrt(g*n) (opens with pin density) -> pins CANNOT be")
print("matter (they would screen gravity). Matter couples to gravity via the")
print("symmetry-respecting d_omega source of [4], which does not gap the medium;")
print("the on-site term is then the matter FIELD's internal mass, not vacuum pins.")
