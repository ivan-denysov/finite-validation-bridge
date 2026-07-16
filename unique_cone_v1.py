# -*- coding: utf-8 -*-
"""
IN-HOUSE PROBE (attempt to close the last debt: the UNIQUE LIGHT CONE, so that
c_gamma = c_T = c_GW coincide numerically).

The obstacle (honest): a substrate carries several massless sectors whose speeds
are set by DIFFERENT couplings -- longitudinal sound c_L=sqrt(k_par), transverse
sound / graviton c_T=sqrt(k_perp), a scalar c_s, and, if the photon is an
INDEPENDENT link field, c_gamma=sqrt(g_link). Generic couplings => different
speeds => no unique cone. We show this first.

The mechanism that closes it: ELASTICITY-GAUGE DUALITY (Kleinert 'world crystal';
Beekman-Zaanen). The emergent photon is NOT an independent field with its own
coupling -- it is the GAUGE DUAL of the transverse elastic sector. The dual
photon's stiffness IS the shear modulus mu; its inertia IS the mass density rho.
Hence its speed is sqrt(mu/rho) = c_T IDENTICALLY -- the SAME number that carries
the graviton (the transverse-traceless strain wave). Not tuned: forced by the
photon being the dual of the sector whose modulus is mu.

Consequence: c_light = c_GW = sqrt(mu/rho) at ALL k, because both are transverse
excitations governed by the ONE shear modulus. GW170817 closed -- conditional on
the physically-motivated identification 'light is the gauge dual of the
transverse substrate sector', which we state explicitly.

This run (simple-cubic dynamical matrices, clean transverse/longitudinal split):
  (A) generic independent photon (g_link = 1) vs graviton (k_perp) -> speeds DIFFER;
  (B) dual photon (g_link := shear modulus k_perp) -> c_gamma = c_T at all k;
  (C) verify the graviton wave speed (dynamic, transverse phonon) equals
      sqrt(shear modulus) from the SAME lattice -- one modulus, one cone.
"""
import numpy as np

kpar, kperp = 1.0, 0.30           # central (longitudinal) and bending (shear) stiffness
E = [np.outer(e, e) for e in np.eye(3)]

def D_elastic(k):
    """Simple-cubic elastic dynamical matrix; 6 neighbours, central+bending."""
    D = np.zeros((3,3))
    for mu in range(3):
        Kb = kpar*E[mu] + kperp*(np.eye(3)-E[mu])
        D += Kb * 2*(1-np.cos(k[mu]))
    return D

def Ktil(k): return np.array([2*np.sin(k[mu]/2) for mu in range(3)])
def D_photon(k, g):
    """Link-Maxwell: g*(|K|^2 I - K K^T); transverse eigenvalues g|K|^2, long. 0."""
    K = Ktil(k); return g*((K@K)*np.eye(3) - np.outer(K, K))

# high-symmetry axis: on a cubic lattice the modes split cleanly into
# longitudinal / transverse only along an axis (off-axis is cubic-anisotropy
# contaminated); the amorphous substrate is isotropic along ALL directions (v2).
nhat = np.array([1,0,0], float)

def speeds_elastic(kmag):
    k = kmag*nhat; w2 = np.sort(np.linalg.eigvalsh(D_elastic(k)))
    # two smallest = transverse, largest = longitudinal
    cT = np.sqrt(w2[:2].mean())/kmag; cL = np.sqrt(w2[2])/kmag
    return cT, cL
def speed_photon(kmag, g):
    k = kmag*nhat; w2 = np.sort(np.linalg.eigvalsh(D_photon(k, g)))[::-1]
    return np.sqrt(max(w2[1],0))/kmag        # transverse branch

cT_theory = np.sqrt(kperp)                    # small-k transverse elastic speed
print(f"lattice: k_par(longitudinal)={kpar}, k_perp(shear/bending)={kperp}")
print(f"transverse elastic speed  c_T = sqrt(k_perp) = {cT_theory:.4f}")
print(f"longitudinal (sound)      c_L = sqrt(k_par)  = {np.sqrt(kpar):.4f}\n")

print("(A) INDEPENDENT photon (generic link coupling g=1) vs graviton c_T:")
print(f"{'|k|':>7} {'c_gamma(g=1)':>13} {'c_T':>8} {'ratio c_g/c_T':>14}")
for km in [0.05, 0.15, 0.30]:
    cg = speed_photon(km, 1.0); cT,_ = speeds_elastic(km)
    print(f"{km:>7.2f} {cg:>13.4f} {cT:>8.4f} {cg/cT:>14.4f}")
print("   -> different cones (ratio ~1.8): an independent photon does NOT match.\n")

print("(B) DUAL photon: link coupling := shear modulus k_perp (elasticity-gauge duality):")
print(f"{'|k|':>7} {'c_gamma(dual)':>14} {'c_T':>8} {'ratio c_g/c_T':>14}")
for km in [0.05, 0.15, 0.30, 0.50]:
    cg = speed_photon(km, kperp); cT,_ = speeds_elastic(km)
    print(f"{km:>7.2f} {cg:>14.4f} {cT:>8.4f} {cg/cT:>14.5f}")
print("   -> c_gamma = c_T EXACTLY (ratio 1.00000 at ALL k, not just k->0, because")
print("      2(1-cos k) = |K_tilde|^2: elastic and dual-gauge share one lattice")
print("      dispersion factor). ONE modulus sets both. Unique cone.\n")

# (C) dynamic graviton (transverse phonon) speed = sqrt(shear modulus)
km = 0.05; cT_dyn,_ = speeds_elastic(km)
print("(C) consistency: the transverse phonon (graviton carrier) wave speed")
print(f"    c_T(dynamic, k->0) = {cT_dyn:.4f}  vs  sqrt(shear modulus) = {cT_theory:.4f}")
print(f"    match to {abs(cT_dyn-cT_theory)/cT_theory*100:.2f}% -> the graviton wave and")
print("    the modulus the photon inherits are the SAME number.\n")

print("VERDICT: an independent photon has its own cone (A) -- so a unique cone is")
print("NOT generic. But the emergent photon is the GAUGE DUAL of the transverse")
print("elastic sector: its stiffness is the shear modulus mu, so c_gamma = sqrt(mu/rho)")
print("= c_T = c_GW at all k (B,C), forced by ONE modulus, not tuned. GW170817 closes")
print("EXACTLY, conditional on the single identification 'light is the dual of the")
print("transverse substrate sector' (Kleinert world-crystal) -- now the lone premise.")
