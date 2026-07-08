import numpy as np, time
from scipy.spatial import cKDTree, ConvexHull

# SINGLE hedgehog, CLAMPED radial boundary (charge +1 enforced by boundary,
# no partner to annihilate with). The only escape is slipping through lattice
# discreteness. Core TRACKER: core = node of max neighbor misalignment;
# charge measured around the TRACKED core (ring lesson: meters must follow).
# Existence test = damped relaxation endpoint (stable static defect or not),
# then lightly-damped run for persistence.

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N = len(pos)
i0, i1 = pairs[:, 0], pairs[:, 1]
k, dt = 1.0, 0.02
tree = cKDTree(pos, boxsize=L)
ctr = np.array([0.5, 0.5, 0.5])

d = pos - ctr; d -= L * np.round(d / L)
r = np.linalg.norm(d, axis=1) + 1e-12
u = d / r[:, None]

Rb = 0.35
free = r < Rb            # evolving interior
clamp = ~free            # clamped shell: n = r_hat forever
n = u.copy()             # exact hedgehog ansatz

deg = np.zeros(N, int)
np.add.at(deg, i0, 1); np.add.at(deg, i1, 1)
deg = np.maximum(deg, 1)

def accel(n):
    a = np.zeros((N, 3))
    np.add.at(a, i0, k * n[i1])
    np.add.at(a, i1, k * n[i0])
    a -= (np.einsum('ij,ij->i', a, n))[:, None] * n
    a[clamp] = 0.0
    return a

def energy(n, v):
    return 0.5 * (v**2).sum() + k * (1 - np.einsum('ij,ij->i', n[i0], n[i1])).sum()

def find_core(n):
    # core = node where mean neighbor direction has minimal norm (max misalignment)
    s = np.zeros((N, 3))
    np.add.at(s, i0, n[i1]); np.add.at(s, i1, n[i0])
    a = np.linalg.norm(s, axis=1) / deg
    a[clamp] = 1.0
    a[r > Rb - 0.05] = 1.0   # ignore shell edge artefacts
    a[deg < 3] = 1.0         # exclude isolated/low-degree nodes (58 off giant comp.)
    j = np.argmin(a)
    return pos[j], a[j]

def sphere_sample(npts=400):
    g = (1 + 5**0.5) / 2
    idx = np.arange(npts)
    zz = 1 - 2*(idx + 0.5)/npts
    tt = 2*np.pi*idx/g
    ss = np.sqrt(1 - zz**2)
    return np.stack([ss*np.cos(tt), ss*np.sin(tt), zz], 1)

SPH = sphere_sample()
TRI = ConvexHull(SPH).simplices.copy()
for t in range(len(TRI)):
    a_, b_, c_ = SPH[TRI[t]]
    if np.dot(np.cross(b_ - a_, c_ - a_), (a_ + b_ + c_)) < 0:
        TRI[t] = TRI[t][::-1]

def charge_at(n, center, Rm):
    p = (center + Rm * SPH) % L
    _, idx = tree.query(p)
    f = n[idx]
    A, B, C = f[TRI[:, 0]], f[TRI[:, 1]], f[TRI[:, 2]]
    num = np.einsum('ij,ij->i', A, np.cross(B, C))
    den = 1 + np.einsum('ij,ij->i', A, B) + np.einsum('ij,ij->i', B, C) + np.einsum('ij,ij->i', C, A)
    return (2 * np.arctan2(num, den)).sum() / (4 * np.pi)

def report(tag, n, v):
    cpos, amin = find_core(n)
    q05 = charge_at(n, cpos, 0.05)
    q10 = charge_at(n, cpos, 0.10)
    dc = np.linalg.norm(mi := (cpos - ctr) - L*np.round((cpos - ctr)/L))
    print(f"{tag}: E={energy(n, v):8.2f}  core@|r|={dc:.3f} misalign={amin:.3f}  "
          f"Q(core,0.05)={q05:+.3f}  Q(core,0.10)={q10:+.3f}")

v = np.zeros((N, 3))
print(f"sanity: Q(ctr,0.05)={charge_at(n,ctr,0.05):+.3f}  Q(ctr,0.10)={charge_at(n,ctr,0.10):+.3f}")
report("init   ", n, v)

# STAGE 1: deep damped relaxation (existence = stable static endpoint)
a = accel(n)
for s in range(1, 4001):
    v += 0.5*dt*a
    n[free] += dt*v[free]; n /= np.linalg.norm(n, axis=1)[:, None]
    v -= (np.einsum('ij,ij->i', v, n))[:, None] * n
    a = accel(n)
    v += 0.5*dt*a; v *= 0.90
    if s % 1000 == 0:
        report(f"relax t={s*dt:5.1f}", n, v)

# STAGE 2: lightly damped persistence run
v[:] = 0.0
a = accel(n)
t0 = time.time()
for s in range(1, 12001):
    v += 0.5*dt*a
    n[free] += dt*v[free]; n /= np.linalg.norm(n, axis=1)[:, None]
    v -= (np.einsum('ij,ij->i', v, n))[:, None] * n
    a = accel(n)
    v += 0.5*dt*a; v *= 0.999
    if s % 2000 == 0:
        report(f"run   t={s*dt:6.1f}", n, v)
print(f"[{time.time()-t0:.0f}s]")
