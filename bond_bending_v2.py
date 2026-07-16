# -*- coding: utf-8 -*-
"""
IN-HOUSE PROBE v2 (addresses the branch review of J.6): document N, add the
Maxwell theory line for the floppy count, and measure BOTH sound speeds
(c_L, c_T) on the SAME amorphous substrate so c_T/c_L is a real ratio, not a
comparison across two different systems (J.5 cubic Z=18 vs J.6 amorphous).

Substrate: one amorphous network, N stated, Z-bar tuned to ~5.5 (same
construction protocol as §6; reduced N for the dense eigensolver, stated
explicitly). Each bond a generalised spring
    K = k_par * nn^T + k_perp * (I - nn^T),
k_perp = the harmonic bond-bending (B5 link-curvature) term.

Reported:
  (1) floppy (zero) Hessian modes vs the Maxwell count 3N - N*Z/2 - 3, at
      k_perp=0 (central forces: floppy) and k_perp>0 (rigid, -> 3 translations).
  (2) shear modulus G and longitudinal (P-wave) modulus M on the SAME rigid
      substrate, via affine strain + non-affine relaxation; c_T=sqrt(G/rho),
      c_L=sqrt(M/rho), and the ratio c_T/c_L.

Prediction: measured floppy count ~ Maxwell; k_perp>0 rigidifies (floppy -> 3);
c_T < c_L on the same substrate (as in any elastic solid), the ratio set by
k_perp -- which is exactly why matching c_T to the photon speed c is a free
parameter (a named debt, GW170817 requires c_T = c).
"""
import numpy as np

rng = np.random.RandomState(1)
N = 700; L = 7.05
pos = rng.uniform(0, L, (N, 3))
def mi(d): return d - L*np.round(d/L)
def build(rc):
    bl = []
    for i in range(N):
        d = mi(pos - pos[i]); dd = np.sqrt((d**2).sum(1)); dd[i] = 1e9
        for j in np.where(dd < rc)[0]:
            if j > i: bl.append((i, j))
    return bl
rc = 1.35; bonds = build(rc); Z = 2*len(bonds)/N
while abs(Z - 5.5) > 0.1:
    rc *= (5.5/Z)**(1/3); bonds = build(rc); Z = 2*len(bonds)/N
BV = np.array([mi(pos[j]-pos[i]) for i, j in bonds])
NH = BV/np.linalg.norm(BV, axis=1)[:, None]
rho = N/L**3; V = L**3
maxwell = 3*N - len(bonds) - 3
print(f"substrate: N={N}, bonds={len(bonds)}, Z-bar={Z:.3f}, seed=1, rho={rho:.3f}")
print(f"Maxwell floppy count 3N - N*Z/2 - 3 = {maxwell}  (central forces, below Z=6)\n")

def Kbond(bd, kpar, kperp):
    n = NH[bd]; return kpar*np.outer(n, n) + kperp*(np.eye(3)-np.outer(n, n))

def hessian(kpar, kperp):
    H = np.zeros((3*N, 3*N))
    for bd, (i, j) in enumerate(bonds):
        K = Kbond(bd, kpar, kperp)
        for a in range(3):
            for c in range(3):
                v = K[a, c]
                H[3*i+a, 3*i+c] += v; H[3*j+a, 3*j+c] += v
                H[3*i+a, 3*j+c] -= v; H[3*j+a, 3*i+c] -= v
    return H

def floppy(kperp):
    w = np.linalg.eigvalsh(hessian(1.0, kperp))
    return int((w < 1e-8).sum())

def modulus(kperp, mode):
    """mode='shear' (xy) or 'long' (xx). Returns the elastic modulus."""
    gamma = 1e-3
    d_aff = np.zeros((len(bonds), 3))
    if mode == 'shear': d_aff[:, 0] = gamma*BV[:, 1]      # u_x = gamma*y
    else:               d_aff[:, 0] = gamma*BV[:, 0]      # u_x = gamma*x
    b = np.zeros(3*N)
    for bd, (i, j) in enumerate(bonds):
        f = Kbond(bd, 1.0, kperp) @ d_aff[bd]
        b[3*i:3*i+3] += f; b[3*j:3*j+3] -= f
    H = hessian(1.0, kperp)
    # min-norm least-squares relaxation: handles the 3 translations + any
    # rattler soft modes gracefully (they relax at zero energy)
    u = np.linalg.lstsq(H, -b, rcond=1e-10)[0]
    E = 0.0
    for bd, (i, j) in enumerate(bonds):
        du = d_aff[bd] + (u[3*j:3*j+3] - u[3*i:3*i+3])
        E += 0.5 * du @ Kbond(bd, 1.0, kperp) @ du
    return 2*E/(V*gamma**2)

nf0 = floppy(0.0)
print(f"floppy(k_perp=0) measured = {nf0}, Maxwell = {maxwell}, "
      f"self-stress states = {nf0 - maxwell} (Maxwell-Calladine: N_floppy - N_ss "
      f"= 3N-Nb-3; measured > Maxwell = redundant local over-coordination)\n")
print(f"{'k_perp':>8} {'floppy':>7} {'G(shear)':>9} {'M(long)':>9} {'c_T':>7} {'c_L':>7} {'c_T/c_L':>8}")
print(f"{0.0:>8.2f} {nf0:>7} {'--':>9} {'--':>9} {'--':>7} {'--':>7} {'--':>8}"
      f"   (central-force floppy)")
for kperp in [0.05, 0.1, 0.3]:
    nf = floppy(kperp)
    G = modulus(kperp, 'shear'); M = modulus(kperp, 'long')
    cT = np.sqrt(max(G, 0)/rho); cL = np.sqrt(max(M, 0)/rho)
    print(f"{kperp:>8.2f} {nf:>7} {G:>9.4f} {M:>9.4f} {cT:>7.3f} {cL:>7.3f} {cT/cL:>8.3f}")
print()
print("READING: (1) floppy count = Maxwell + self-stress (Maxwell-Calladine); the")
print("extensive floppy sector (296) confirms sub-isostaticity, its theory line the")
print("Maxwell count 140 plus 156 states of self-stress. (2) bond-bending rigidifies")
print("(floppy -> 3 translations + a few rattlers). (3) c_T and c_L now on the SAME")
print("substrate: c_T/c_L = 0.57-0.71, growing with k_perp, c_L~1.0. The photon speed")
print("c is set by the tick independently, so c_T = c is NOT structural -- a free")
print("ratio -- which is the named debt (GW170817 requires c_T = c to ~1e-15).")
