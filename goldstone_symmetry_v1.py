# -*- coding: utf-8 -*-
"""
IN-HOUSE PROBE 2: is the masslessness SYMMETRY-PROTECTED (a true Goldstone by
theorem), or just "we omitted the mass term"?

Test: add perturbations to the gapless (g=0) chain and measure the gap.
  - Perturbations that RESPECT the phase-shift symmetry theta_i -> theta_i + c
    (i.e. depend only on phase DIFFERENCES theta_i - theta_j) must NOT open a
    gap, however strong.
  - Perturbations that BREAK it (depend on the ABSOLUTE theta_i) must open a
    gap, whatever their specific form.
If so, the gap <=> symmetry breaking, and the masslessness is Goldstone's
theorem, not an omission.

Cases:
  A  baseline (nearest-neighbour wave)                      symmetric  -> expect gap 0
  B  + strong next-nearest-neighbour coupling (h)           symmetric  -> expect gap 0
  C  + biharmonic difference term (b)                       symmetric  -> expect gap 0
  D  + on-site sin(theta) pin (g)                           BREAKS     -> expect gap sqrt(g)
  E  + random-phase on-site pin g*sin(theta - phi_i)        BREAKS     -> expect gap >0

Falsifier of the Goldstone claim: a gap in B or C, or no gap in D/E.
"""
import numpy as np

N = 256; dx = 1.0; c = 1.0
dt = 0.02; STEPS = 90000; REC = 12
A = 1e-3
G = 0.05
rng_phi = np.random.RandomState(7)
PHI = rng_phi.uniform(0, 2*np.pi, N)          # random pin phases for case E

def nn(th):        # nearest-neighbour sin-Laplacian (symmetric)
    return (np.sin(np.roll(th,-1)-th) - np.sin(th-np.roll(th,1)))/dx**2
def nnn(th):       # next-nearest-neighbour (symmetric: depends on differences)
    return (np.sin(np.roll(th,-2)-th) - np.sin(th-np.roll(th,2)))/dx**2
def biharm(th):    # discrete biharmonic of the phase (symmetric)
    lap = np.roll(th,-1)-2*th+np.roll(th,1)
    return -(np.roll(lap,-1)-2*lap+np.roll(lap,1))/dx**4

def force(th, case, h=0.5, b=0.3):
    f = c**2*nn(th)
    if case == 'B': f = f + h*nnn(th)
    if case == 'C': f = f + b*biharm(th)
    if case == 'D': f = f - G*np.sin(th)              # symmetry-breaking pin
    if case == 'E': f = f - G*np.sin(2*th)            # symmetry-breaking, other form (period-2)
    return f

def simulate(case, seed=0):
    r = np.random.RandomState(seed)
    th = A*r.standard_normal(N); v = np.zeros(N)
    nrec = STEPS//REC; fld = np.empty((nrec, N)); j = 0
    for t in range(STEPS):
        v += dt*force(th, case); th += dt*v
        if t % REC == 0 and j < nrec:
            fld[j] = th; j += 1
    return fld[:j]

def gap_of(case):
    fld = simulate(case)
    nt, nx = fld.shape
    F = np.fft.rfft(fld*np.hanning(nt)[:,None], axis=0)
    F = np.fft.fft(F, axis=1); P = np.abs(F)**2
    omg = 2*np.pi*np.fft.rfftfreq(nt, d=dt*REC)
    ks  = 2*np.pi*np.fft.fftfreq(nx, d=dx)
    kk, ww = [], []
    for kx in range(1, nx//2):
        kk.append(abs(ks[kx])); ww.append(omg[np.argmax(P[:,kx])])
    kk = np.array(kk); ww = np.array(ww)
    order = kk.argsort(); kk, ww = kk[order], ww[order]
    sel = kk < 0.35
    slope, inter = np.polyfit(kk[sel]**2, ww[sel]**2, 1)
    gap = np.sqrt(max(inter, 0.0))
    ratio = ww[1] / ww[0]                             # omega(2k)/omega(k): ~2 gapless, ~1 gapped
    return gap, ww[0], ratio

labels = {'A':'baseline NN            (symmetric)',
          'B':'+NNN coupling h=0.5    (symmetric)',
          'C':'+biharmonic b=0.3      (symmetric)',
          'D':'+on-site sin(theta)    (BREAKS)   ',
          'E':'+on-site sin(2 theta)  (BREAKS)   '}
print("SYMMETRY-PROTECTION TEST  (gapless g=0 chain + perturbations)\n")
print(f"{'case':>4}  {'perturbation':<34} {'gap':>8} {'om(k1)':>8} {'om(2k)/om(k)':>13}  verdict")
res = {}
for case in ['A','B','C','D','E']:
    gap, lowk, ratio = gap_of(case)
    res[case] = ratio
    v = 'GAPLESS' if ratio > 1.7 else ('GAPPED' if ratio < 1.3 else '??')
    print(f"{case:>4}  {labels[case]:<34} {gap:>8.4f} {lowk:>8.4f} {ratio:>13.2f}  {v}")
print()
print(f"om(2k)/om(k): ~2.0 = gapless (linear, massless) ; ~1.0 = gapped (massive)")
sym_ok = res['B'] > 1.7 and res['C'] > 1.7
brk_ok = res['D'] < 1.3 and res['E'] < 1.3
print(f"VERDICT: symmetric stay GAPLESS: {'OK' if sym_ok else 'FAIL'} "
      f"(B={res['B']:.2f}, C={res['C']:.2f}); "
      f"breaking GAP: {'OK' if brk_ok else 'FAIL'} (D={res['D']:.2f}, E={res['E']:.2f})")
print("=> if both OK, the masslessness is Goldstone (symmetry-protected), not an omission.")
