import numpy as np

# TEST A: localized ball twist (theta: 2pi at center -> 0 outside), particle candidate.
# TEST B: planar kink wall across the box (control: must be stable in 3D).
# Measure over time: total energy, LOCALIZATION of energy (participation ratio),
# peak site-energy. If A delocalizes while B holds -> Derrick wall, measured not asserted.

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N = len(pos)
i0, i1 = pairs[:, 0], pairs[:, 1]
k, m2, dt = 1.0, 1.0, 0.02

def accel(th):
    dth = th[i1] - th[i0]
    a = np.zeros(N)
    np.add.at(a, i0, k * dth)
    np.add.at(a, i1, -k * dth)
    return a - m2 * np.sin(th)

def site_energy(th, v):
    e = 0.5 * v**2 + m2 * (1 - np.cos(th))
    dth2 = 0.5 * k * (th[i1] - th[i0])**2
    np.add.at(e, i0, 0.5 * dth2)
    np.add.at(e, i1, 0.5 * dth2)
    return e

def run(th, label, steps=6000, relax=800):
    v = np.zeros(N)
    # gentle relaxation first (settle to near-equilibrium shape), then free symplectic
    a = accel(th)
    for s in range(relax):
        v += 0.5 * dt * a; th += dt * v; a = accel(th)
        v += 0.5 * dt * a; v *= 0.95
    v[:] = 0.0
    a = accel(th)
    e = site_energy(th, v); E0 = e.sum()
    pr0 = (e.sum()**2) / (N * (e**2).sum())  # participation ratio in [1/N..1]; small=localized
    print(f"[{label}] after relax: E={E0:.2f}  PR={pr0:.4f}  peak_e={e.max():.3f}")
    for s in range(1, steps + 1):
        v += 0.5 * dt * a; th += dt * v; a = accel(th)
        v += 0.5 * dt * a
        if s % 1500 == 0:
            e = site_energy(th, v)
            pr = (e.sum()**2) / (N * (e**2).sum())
            print(f"[{label}] t={s*dt:6.1f}: E={e.sum():.2f}  PR={pr:.4f}  peak_e={e.max():.3f}")
    return th

ctr = np.array([0.5, 0.5, 0.5])

# --- TEST A: ball twist
d = pos - ctr; d -= L * np.round(d / L)
rr = np.linalg.norm(d, axis=1)
R = 0.12
thA = np.where(rr < R, 2*np.pi * (1 - rr/R), 0.0)
print("=== TEST A: localized 3D ball twist (particle candidate) ===")
run(thA.copy(), "A ball")

# --- TEST B: planar kink wall (theta: 0 -> 2pi across x), classic dSG kink profile
# width w ~ 1/sqrt(m2) in lattice units of mean edge (use physical width 0.1 of box)
w = 0.05
x = pos[:, 0]
thB = 4 * np.arctan(np.exp((x - 0.35) / w)) - 4 * np.arctan(np.exp((x - 0.75) / w))
# (kink+antikink pair so the periodic box is consistent; they are far apart)
print("\n=== TEST B: planar kink wall pair (control) ===")
run(thB.copy(), "B wall")
