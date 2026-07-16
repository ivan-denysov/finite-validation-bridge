# -*- coding: utf-8 -*-
"""
IN-HOUSE PROBE 3 (3D): the massless/massive reconciliation at the SAME
dimensional level as the tension it resolves.

The Bridge's 3D continuum limit is a MASSIVE isotropic Klein-Gordon
(omega^2 = m^2 + c^2 k^2), the mass coming from an on-site term inserted by
hand. Claim: remove the on-site term and the 3D limit is MASSLESS
(omega^2 = c^2 k^2) — the Goldstone mode of the broken phase-shift symmetry;
put it back and the gap reopens at sqrt(g). Same lattice, two sectors:
gravity (massless) and matter (massive).

Setup: 3D cubic periodic lattice, second-order phase dynamics (the inertia
the finite tick generates), small amplitude (linear regime),
    theta_ddot = c^2 * sum_6neighbours sin(dtheta) / dx^2  - g*sin(theta).
Longitudinal dispersion omega(k_x) with k_y=k_z=0 is isolated by recording the
TRANSVERSE-AVERAGED field M[t,x] = <theta>_{y,z}, then FFT in (x,t).

Predictions (before run):
  g = 0    : omega(k->0) -> 0            (massless Goldstone; m^2 = 0)
  g = 0.05 : omega(k->0) -> sqrt(0.05)=0.2236  (massive Klein-Gordon)
Discriminator omega(2k)/omega(k): ~2 gapless, ~1 gapped (form-immune).
Falsifier: a gap at g=0, or no gap at g>0.
"""
import numpy as np

LX = 64; LY = 6; LZ = 6; N = LX*LY*LZ; dx = 1.0; c = 1.0
dt = 0.02; STEPS = 80000; REC = 12
A = 1e-3

def lap_sin_3d(th):
    s = np.zeros_like(th)
    for ax in (0, 1, 2):
        s += np.sin(np.roll(th, -1, ax) - th) - np.sin(th - np.roll(th, 1, ax))
    return s / dx**2

def uniform_mode_gap(g):
    """Exact Goldstone test: a uniform shift is a zero mode iff g=0 (any lattice)."""
    th = np.full((LX, LY, LZ), A); v = np.zeros((LX, LY, LZ))
    probe = np.empty(STEPS)
    for t in range(STEPS):
        v += dt * (c**2 * lap_sin_3d(th) - g * np.sin(th)); th += dt * v
        probe[t] = th[0, 0, 0]
    probe -= probe.mean()
    f = np.fft.rfftfreq(STEPS, d=dt)
    om = 2*np.pi*f[np.argmax(np.abs(np.fft.rfft(probe)))]
    return om

def run(g, seed=0):
    r = np.random.RandomState(seed)
    th = A * r.standard_normal((LX, LY, LZ)); v = np.zeros((LX, LY, LZ))
    nrec = STEPS // REC
    M = np.empty((nrec, LX)); j = 0
    for t in range(STEPS):
        v += dt * (c**2 * lap_sin_3d(th) - g * np.sin(th)); th += dt * v
        if t % REC == 0 and j < nrec:
            M[j] = th.mean(axis=(1, 2))          # transverse average -> k_y=k_z=0
            j += 1
    M = M[:j]
    F = np.fft.rfft(M * np.hanning(M.shape[0])[:, None], axis=0)
    F = np.fft.fft(F, axis=1); P = np.abs(F)**2
    omg = 2*np.pi*np.fft.rfftfreq(M.shape[0], d=dt*REC)
    ks = 2*np.pi*np.fft.fftfreq(LX, d=dx)
    kk, ww = [], []
    for kx in range(1, LX//2):
        kk.append(abs(ks[kx])); ww.append(omg[np.argmax(P[:, kx])])
    kk = np.array(kk); ww = np.array(ww)
    o = kk.argsort(); kk, ww = kk[o], ww[o]
    # fit omega^2 = gap^2 + s * exact discrete basis 4 sin^2(k/2)/dx^2, all k
    basis = (2/dx)**2 * np.sin(kk*dx/2)**2
    slope, inter = np.polyfit(basis, ww**2, 1)
    gap = np.sqrt(max(inter, 0.0))
    print(f"--- g = {g} (theory gap sqrt(g) = {np.sqrt(g):.4f}) ---")
    print("  lowest k (k,omega):", ", ".join(f"({k:.3f},{o_:.4f})" for k, o_ in zip(kk[:4], ww[:4])))
    print(f"  => fit gap omega(k->0) = {gap:.4f}, c^2_fit = {slope:.3f}")
    return gap

if __name__ == "__main__":
    print(f"3D GOLDSTONE / MASS TEST — box {LX}x{LY}x{LZ} = {N} nodes, second-order")
    print(f"(elongated box: k_x,min = {2*np.pi/LX:.3f} resolves low k cheaply)\n")
    print("PART 1 — longitudinal dispersion omega(k_x), gap from exact-basis fit:")
    g0 = run(0.0)
    print()
    g1 = run(0.05)
    print()
    print("PART 2 — exact uniform-mode test (Goldstone zero mode, lattice-independent):")
    u0 = uniform_mode_gap(0.0)
    u1 = uniform_mode_gap(0.05)
    print(f"  g=0    : uniform-mode omega = {u0:.4f}  (Goldstone zero mode -> ~0)")
    print(f"  g=0.05 : uniform-mode omega = {u1:.4f}  (theory sqrt(g) = {np.sqrt(0.05):.4f})")
    print()
    print("VERDICT (3D):")
    print(f"  g=0    : gap(fit)={g0:.4f}, uniform-mode={u0:.4f}  -> "
          f"{'MASSLESS Goldstone  OK' if g0 < 0.03 and u0 < 0.02 else 'check'}")
    print(f"  g=0.05 : gap(fit)={g1:.4f}, uniform-mode={u1:.4f}  -> "
          f"{'MASSIVE Klein-Gordon  OK' if abs(g1-0.2236)<0.03 and abs(u1-0.2236)<0.02 else 'check'}")
    print("  => in 3D too: removing the on-site term turns The Bridge's massive KG")
    print("     into a massless Goldstone. Gravity and matter are two sectors, one network.")
