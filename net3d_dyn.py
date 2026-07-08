import numpy as np

# Discrete sine-Gordon on the amorphous 3D graph (Bridge: network+inertia = dSG).
# theta_i'' = k * sum_j (theta_j - theta_i) - m2 * sin(theta_i)
# Symplectic leapfrog (fixed lesson: no explicit-Euler pumping).

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N = len(pos)
i0, i1 = pairs[:, 0], pairs[:, 1]

k, m2 = 1.0, 1.0
dt = 0.02

def accel(th):
    dth = th[i1] - th[i0]
    a = np.zeros(N)
    np.add.at(a, i0, k * dth)
    np.add.at(a, i1, -k * dth)
    return a - m2 * np.sin(th)

def energy(th, v):
    dth = th[i1] - th[i0]
    return 0.5 * (v**2).sum() + 0.5 * k * (dth**2).sum() + m2 * (1 - np.cos(th)).sum()

rng = np.random.default_rng(2)
th = rng.normal(0, 0.3, N)   # random small excitation around vacuum theta=0
v = np.zeros(N)

E0 = energy(th, v)
a = accel(th)
for step in range(4000):
    v += 0.5 * dt * a
    th += dt * v
    a = accel(th)
    v += 0.5 * dt * a
E1 = energy(th, v)
print(f"E0={E0:.4f}  E_final={E1:.4f}  drift={(E1-E0)/E0*100:.3f}%  (4000 steps, dt={dt})")
