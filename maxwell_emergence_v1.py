# -*- coding: utf-8 -*-
"""
IN-HOUSE PROBE (finish debt #2, the last hinge: 'is light a transverse mode?').
c_GW = c_light was reduced to a symmetry identity PROVIDED light is a transverse,
massless excitation of the substrate. That identification is the Maxwell/EM-
emergence debt. Here we exhibit the mechanism: a link (bond) vector field with a
magnetic (curl^2) energy -- the natural gauge field of a network -- reproduces
the three defining kinematic facts of the photon:

  (1) TWO transverse, gapless polarisations with omega = c_gamma |k| (light is a
      transverse wave, 2 states, no gap);
  (2) the LONGITUDINAL component is a non-propagating ZERO mode (pure gauge) --
      light has no longitudinal polarisation;
  (3) U(1) GAUGE INVARIANCE A -> A + grad(chi) leaves the magnetic energy exactly
      invariant -- the symmetry that PROTECTS masslessness (photon massless for
      the same structural reason the graviton-Goldstone is: a protected symmetry).

Lattice dynamical matrix (temporal gauge, lattice momenta K_mu = 2 sin(k_mu/2)):
    M(k) = |K|^2 I - K K^T,  eigenvalues { |K|^2, |K|^2, 0 }.
The two |K|^2 modes are transverse photons (omega = |K|); the 0 is longitudinal
(gauge). We diagonalise M(k) along a generic direction, evolve the amplitudes to
confirm the longitudinal one does not propagate, and test gauge invariance on a
full 3D field by FFT.

Honest residual after this run: the photon is now STRUCTURALLY transverse and
massless. The only thing left for c_GW = c_light is the UNIQUE-LIGHT-CONE
condition c_gamma = c_T (all massless sectors share one speed) -- the generic
emergent-Lorentz problem, which a substrate only approximates and which is NOT
claimed closed here.
"""
import numpy as np

# ---------- Part 1: dispersion of the link-Maxwell dynamical matrix ----------
def Kvec(k): return 2*np.sin(k/2)                      # lattice momentum
def M(k):
    K = Kvec(k); return (K@K)*np.eye(3) - np.outer(K, K)

nhat = np.array([1,1,2], float); nhat /= np.linalg.norm(nhat)
print("PHOTON DISPERSION (link field, magnetic energy), generic direction:")
print(f"{'|k|':>7} {'w_T1':>8} {'w_T2':>8} {'w_L':>8} {'c_gamma=w_T/|k|':>16}")
for kmag in [0.10, 0.20, 0.30, 0.40, 0.60]:
    k = kmag*nhat
    w = np.sqrt(np.clip(np.sort(np.linalg.eigvalsh(M(k)))[::-1], 0, None))  # desc
    Kn = np.linalg.norm(Kvec(k))
    print(f"{kmag:>7.2f} {w[0]:>8.4f} {w[1]:>8.4f} {w[2]:>8.2e} {w[0]/kmag:>16.4f}")
print("  -> two DEGENERATE transverse branches, gapless (omega -> c_gamma|k|);")
print("     the third (longitudinal) eigenvalue is exactly 0 at every k.\n")

# ---------- Part 2: longitudinal does not propagate (dynamics) ----------
k = 0.3*nhat; Mk = M(k)
K = Kvec(k)
eL = K/np.linalg.norm(K)                               # longitudinal polarisation
eT = np.cross(eL, [0,0,1.0]); eT /= np.linalg.norm(eT) # a transverse polarisation
def evolve(a0, steps=20000, dt=0.02):
    a = a0.copy(); v = np.zeros(3); rec = np.empty(steps)
    for t in range(steps):
        v += dt*(-(Mk@a)); a += dt*v; rec[t] = a@a0
    return rec
def omega(rec, dt=0.02):
    rec = rec - rec.mean()
    f = np.fft.rfftfreq(len(rec), d=dt)
    return 2*np.pi*f[np.argmax(np.abs(np.fft.rfft(rec*np.hanning(len(rec)))))]
wT = omega(evolve(eT)); wL = omega(evolve(eL))
print("DYNAMICS at |k|=0.3 (evolve the amplitude, read frequency):")
print(f"  transverse pol:  omega = {wT:.4f}  (propagates; expect |K|={np.linalg.norm(K):.4f})")
print(f"  longitudinal pol: omega = {wL:.4f}  (does NOT propagate -> pure gauge)\n")

# ---------- Part 3: U(1) gauge invariance on a full 3D field ----------
n = 16; rng = np.random.RandomState(1)
A = rng.randn(n, n, n, 3)
kx = 2*np.pi*np.fft.fftfreq(n)
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
Kc = [2*np.sin(KX/2), 2*np.sin(KY/2), 2*np.sin(KZ/2)]   # real lattice momenta magnitudes
# use complex phase factors for exact lattice curl: k_mu -> (e^{i k_mu} - 1)
Ph = [np.exp(1j*KX)-1, np.exp(1j*KY)-1, np.exp(1j*KZ)-1]
def mag_energy(A):
    Ah = [np.fft.fftn(A[...,mu]) for mu in range(3)]
    # B_{mu nu} = Ph_mu A_nu - Ph_nu A_mu ; energy = sum_{mu<nu} |B|^2
    E = 0.0
    for mu in range(3):
        for nu in range(mu+1, 3):
            B = Ph[mu]*Ah[nu] - Ph[nu]*Ah[mu]
            E += np.sum(np.abs(B)**2)
    return E / n**3
E0 = mag_energy(A)
chi = rng.randn(n, n, n)                                # gauge function
chih = np.fft.fftn(chi)
gA = A.copy()
for mu in range(3):
    gA[..., mu] += np.real(np.fft.ifftn(Ph[mu]*chih))   # A -> A + grad(chi)
E1 = mag_energy(gA)
print("U(1) GAUGE INVARIANCE (full 3D field):")
print(f"  magnetic energy before gauge shift: {E0:.6f}")
print(f"  after A -> A + grad(chi):           {E1:.6f}")
print(f"  relative change: {abs(E1-E0)/E0:.2e}  -> invariant (masslessness protected)\n")

print("VERDICT: the link-Maxwell sector reproduces the photon's defining kinematics")
print("-- 2 transverse gapless polarisations, no propagating longitudinal, U(1)")
print("gauge invariance protecting masslessness. So 'light is a transverse, massless")
print("substrate mode' is STRUCTURAL, closing the last hinge of the c_GW = c_light")
print("argument at the kinematic level. The sole remaining debt is the UNIQUE LIGHT")
print("CONE c_gamma = c_T (all massless sectors one speed) -- the emergent-Lorentz")
print("problem, named, not tuned.")
