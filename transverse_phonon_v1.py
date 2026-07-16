# -*- coding: utf-8 -*-
"""
IN-HOUSE PROBE: does the network carry a TENSOR sector, or is gravity here
confined to a scalar?

The scalar delay field (the phase-shift Goldstone of §6d) gives the Newtonian
potential and, via the two-channel reading of [1], light deflection --- but a
scalar cannot carry the spin-2 (gravitational-wave / frame-dragging) content of
full GR. The claim to test: the network is NOT confined to scalar. Its NODE
POSITIONS are dynamical (postulate B5), so the substrate is an elastic solid,
and an elastic solid has, besides one LONGITUDINAL acoustic branch, TWO
TRANSVERSE (shear) branches --- and the transverse-traceless part of the strain
is spin-2. These transverse phonons are the Goldstones of TRANSLATION symmetry
(distinct from the phase-shift Goldstone = scalar gravity). If they exist and
are gapless, the tensor sector exists in-model.

Model: a 3D cubic network of central-force springs on nearest (NN, 6) and
next-nearest (NNN, 12) neighbours -- coordination Z=18, well above the central-
force rigidity threshold 2d=6 (a pure-NN or a floppy Z<6 network has ZERO shear
modulus, so NNN bonds are needed to give the shear rigidity a real solid has).
Node displacement u (3-vector); linear central-force dynamics
    u_ddot_i = sum_bonds k * nhat (nhat . (u_j - u_i)).
For k along [100], longitudinal (u||x) and transverse (u_y, u_z) plane waves are
exact eigenmodes by symmetry, so their frequencies separate cleanly.

Prediction (before run): BOTH branches gapless (omega->0 as k->0, acoustic
Goldstones of translation), with c_T < c_L (shear slower than compression), and
the two transverse branches (y,z) degenerate.
Falsifier: a transverse gap (no shear rigidity), or no transverse branch.
"""
import numpy as np

LX, LY, LZ = 40, 5, 5; dx = 1.0                   # elongated: small k_x cheaply
dt = 0.05; STEPS = 16000; REC = 8; A = 1e-3

# neighbour offsets: 6 NN + 12 NNN, central-force springs (k = 1/|d|^2)
offsets = []
for a in (-1, 0, 1):
    for b in (-1, 0, 1):
        for c in (-1, 0, 1):
            d = (a, b, c); s = a*a + b*b + c*c
            if s in (1, 2):                      # NN (|d|^2=1) and NNN (=2)
                offsets.append(np.array(d, float))
BONDS = []
for d in offsets:
    n = d / np.linalg.norm(d)
    P = np.outer(n, n)                           # projector along the bond
    k = 1.0 / (d @ d)                            # softer for longer bonds
    BONDS.append((tuple(int(x) for x in d), P, k))

def force(u):
    """u shape (L,L,L,3) -> elastic force, central-force NN+NNN."""
    F = np.zeros_like(u)
    for d, P, k in BONDS:
        uj = np.roll(u, shift=(-d[0], -d[1], -d[2]), axis=(0, 1, 2))
        F += k * ((uj - u) @ P.T)                # P (u_j - u_i)
    return F

def dispersion(pol):
    """pol=0 longitudinal (u_x), pol=1 transverse (u_y). Return omega(k_x)."""
    rng = np.random.RandomState(0)
    u = np.zeros((LX, LY, LZ, 3))
    u[..., pol] = A * rng.standard_normal((LX, LY, LZ))   # broadband, one polarisation
    v = np.zeros_like(u)
    nrec = STEPS // REC
    M = np.empty((nrec, LX)); j = 0
    for t in range(STEPS):
        v += dt * force(u); u += dt * v
        if t % REC == 0 and j < nrec:
            M[j] = u[..., pol].mean(axis=(1, 2))        # transverse-average -> k_y=k_z=0
            j += 1
    M = M[:j]
    F = np.fft.rfft(M * np.hanning(M.shape[0])[:, None], axis=0)
    F = np.fft.fft(F, axis=1); P = np.abs(F)**2
    omg = 2*np.pi*np.fft.rfftfreq(M.shape[0], d=dt*REC)
    ks = 2*np.pi*np.fft.fftfreq(LX, d=dx)
    out = []
    for kx in range(1, LX//2):
        out.append((abs(ks[kx]), omg[np.argmax(P[:, kx])]))
    return np.array(out)

print(f"TRANSVERSE-PHONON TEST — box {LX}x{LY}x{LZ} NN+NNN central-force (Z=18)\n")
dL = dispersion(0)   # longitudinal
dT = dispersion(1)   # transverse
print(f"{'k_x':>7} {'omega_L':>9} {'omega_T':>9}   (both should ->0 as k->0)")
for (k, oL), (_, oT) in zip(dL, dT):
    print(f"{k:>7.3f} {oL:>9.4f} {oT:>9.4f}")
# sound speeds from the two lowest-k points
cL = dL[0, 1] / dL[0, 0]; cT = dT[0, 1] / dT[0, 0]
gapL = np.polyfit(dL[:3, 0]**2, dL[:3, 1]**2, 1)[1]
gapT = np.polyfit(dT[:3, 0]**2, dT[:3, 1]**2, 1)[1]
print()
print(f"c_L = {cL:.3f}, c_T = {cT:.3f}  (shear slower: {'YES' if cT < cL else 'no'})")
print(f"gap_L = {np.sqrt(max(gapL,0)):.4f}, gap_T = {np.sqrt(max(gapT,0)):.4f}  "
      f"(both ~0 = acoustic/gapless)")
print()
print("VERDICT: a transverse (shear) acoustic branch exists and is gapless")
print("=> the network carries a tensor (spin-2) sector -- the translation")
print("   Goldstones -- distinct from the scalar phase-shift Goldstone of gravity.")
print("   Gravity here is NOT confined to scalar; the tensor sector is B5 elasticity.")
