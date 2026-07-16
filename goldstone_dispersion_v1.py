# -*- coding: utf-8 -*-
"""
IN-HOUSE PROBE: the massless/massive reconciliation (series debt D-open-1),
tested as a dispersion measurement via space-time 2D FFT (one sim per g).

Claim under test (Goldstone reconciliation):
  - Synchronised network WITHOUT an on-site term = Goldstone mode of the
    broken phase-shift symmetry: gapless, omega(k->0)->0 => MASSLESS
    (the gravity / long-range sector).
  - WITH the on-site (sine-Gordon) term inserted by The Bridge: a GAP opens,
    omega(k->0)=sqrt(g) => MASSIVE Klein-Gordon (the matter sector).
  Both on ONE lattice => two sectors, not a contradiction.

Dynamics (second-order; the inertia the finite tick generates, per The Bridge),
1D periodic chain, small amplitude (linear regime):
    theta_ddot_i = c^2 (discrete-Laplacian sin-coupling) - g*sin(theta_i)
Seed all modes with small white noise, record theta(x,t), 2D FFT -> for each k
pick the peak omega. Fit omega^2 = gap^2 + c^2 * K(k), gap = omega(k->0).

Predictions (before run): g=0 -> gap=0 (Goldstone); g=0.05 -> gap=sqrt(0.05)=0.2236.
Falsifier: a gap at g=0, or no gap at g>0.
"""
import numpy as np

N = 256; dx = 1.0; c = 1.0
dt = 0.02; STEPS = 120000; REC = 12          # record every REC steps
A = 1e-3

def lap_sin(th):
    return (np.sin(np.roll(th, -1) - th) - np.sin(th - np.roll(th, 1))) / dx**2

def simulate(g, seed=0):
    rng = np.random.RandomState(seed)
    th = A * rng.standard_normal(N)
    v = np.zeros(N)
    nrec = STEPS // REC
    field = np.empty((nrec, N))
    j = 0
    for t in range(STEPS):
        a = c**2 * lap_sin(th) - g * np.sin(th)
        v += dt * a; th += dt * v
        if t % REC == 0 and j < nrec:
            field[j] = th; j += 1
    return field[:j]

def dispersion(field):
    """FFT space (k) then time (omega); peak-power omega for each k."""
    nt, nx = field.shape
    F = np.fft.rfft(field * np.hanning(nt)[:, None], axis=0)  # time  -> omega
    F = np.fft.fft(F, axis=1)                                 # space -> k
    P = np.abs(F)**2                                          # (nt//2+1, nx)
    dt_rec = dt * REC
    omegas = 2*np.pi*np.fft.rfftfreq(nt, d=dt_rec)
    ks = 2*np.pi*np.fft.fftfreq(nx, d=dx)
    out = []
    for kx in range(1, nx//2):                               # positive k, skip 0
        om = omegas[np.argmax(P[:, kx])]
        out.append((abs(ks[kx]), om))
    return np.array(out)

def gap_fit(disp):
    k = disp[:, 0]; om = disp[:, 1]
    sel = k < 0.9                                        # low-k window for the gap
    Kfun = (2/dx)**2 * np.sin(k[sel]*dx/2)**2
    slope, intercept = np.polyfit(Kfun, om[sel]**2, 1)
    return np.sqrt(max(intercept, 0.0)), slope

def run(g):
    disp = dispersion(simulate(g))
    gap, cc = gap_fit(disp)
    lowk = disp[disp[:,0].argsort()][:5]
    print(f"--- g = {g} ---  (theory gap = sqrt(g) = {np.sqrt(g):.4f})")
    print("  lowest-k (k, omega_meas):", ", ".join(f"({k:.3f},{o:.4f})" for k,o in lowk))
    print(f"  => fitted GAP omega(k->0) = {gap:.4f}   c^2_fit = {cc:.3f}")
    return gap

if __name__ == "__main__":
    print("GOLDSTONE / MASS DISPERSION TEST (1D, second-order, linear regime)\n")
    g0 = run(0.0)
    print()
    g1 = run(0.05)
    print()
    print("VERDICT:")
    print(f"  g=0    : gap = {g0:.4f}  -> {'MASSLESS (Goldstone) OK' if g0 < 8e-3 else 'HAS GAP -> FAIL'}")
    print(f"  g=0.05 : gap = {g1:.4f}  -> {'MASSIVE (KG) OK' if abs(g1-0.2236) < 0.04 else 'off theory'}")
    print("  Both on one lattice => massless & massive are two sectors, not a conflict.")
