import numpy as np, time
from scipy.spatial import cKDTree, ConvexHull

# VECTOR SECTOR from the 6-direction primitive: node field n in S^2
# (coarse-grained transition direction; |v|=1 directed regime).
# Coupling: E = k * sum_links (1 - n_i . n_j)  (leading order of ANY symmetric
# alignment of transition directions - no free structure inserted).
# TEST: hedgehog-antihedgehog pair (pi_2 charges +1/-1; total 0 for periodic box).
# HONEST charge meter: degree of map S^2 -> S^2 via signed solid angles
# (Van Oosterom-Strackee), at two radii. NOT an energy metric (ring lesson).

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N = len(pos)
i0, i1 = pairs[:, 0], pairs[:, 1]
k, dt = 1.0, 0.02
tree = cKDTree(pos, boxsize=L)

c1 = np.array([0.25, 0.5, 0.5])
c2 = np.array([0.75, 0.5, 0.5])
R = np.diag([1.0, 1.0, -1.0])   # reflection -> charge -1 for the anti-hedgehog

def mi(dv):
    return dv - L * np.round(dv / L)

d1 = mi(pos - c1); r1 = np.linalg.norm(d1, axis=1) + 1e-12
d2 = mi(pos - c2); r2 = np.linalg.norm(d2, axis=1) + 1e-12
u1 = d1 / r1[:, None]
u2 = (d2 / r2[:, None]) @ R.T
w1 = 1.0 / r1**2
w2 = 1.0 / r2**2
m = u1 * w1[:, None] + u2 * w2[:, None]
n = m / (np.linalg.norm(m, axis=1)[:, None] + 1e-12)

def accel(n):
    a = np.zeros((N, 3))
    np.add.at(a, i0, k * n[i1])
    np.add.at(a, i1, k * n[i0])
    a -= (np.einsum('ij,ij->i', a, n))[:, None] * n   # tangent projection
    return a

def energy(n, v):
    return 0.5 * (v**2).sum() + k * (1 - np.einsum('ij,ij->i', n[i0], n[i1])).sum()

# --- pi_2 charge meter: icosphere sample around center c, radius Rm ---
def sphere_sample(npts=400):
    g = (1 + 5**0.5) / 2
    idx = np.arange(npts)
    zz = 1 - 2*(idx + 0.5)/npts
    tt = 2*np.pi*idx/g
    ss = np.sqrt(1 - zz**2)
    return np.stack([ss*np.cos(tt), ss*np.sin(tt), zz], 1)

SPH = sphere_sample()
HULL = ConvexHull(SPH)   # triangulation of the sample sphere
TRI = HULL.simplices
# orient triangles outward
for t in range(len(TRI)):
    a, b, c = SPH[TRI[t]]
    if np.dot(np.cross(b - a, c - a), (a + b + c)) < 0:
        TRI[t] = TRI[t][::-1]

def charge(n, center, Rm):
    p = (center + Rm * SPH) % L
    _, idx = tree.query(p)
    f = n[idx]
    A, B, C = f[TRI[:, 0]], f[TRI[:, 1]], f[TRI[:, 2]]
    num = np.einsum('ij,ij->i', A, np.cross(B, C))
    den = 1 + np.einsum('ij,ij->i', A, B) + np.einsum('ij,ij->i', B, C) + np.einsum('ij,ij->i', C, A)
    om = 2 * np.arctan2(num, den)
    return om.sum() / (4 * np.pi)

print(f"init:  Q(c1,0.08)={charge(n,c1,0.08):+.3f}  Q(c1,0.15)={charge(n,c1,0.15):+.3f}  "
      f"Q(c2,0.08)={charge(n,c2,0.08):+.3f}")

v = np.zeros((N, 3))
a = accel(n)
for s in range(800):   # damped relax
    v += 0.5*dt*a
    n += dt*v; n /= np.linalg.norm(n, axis=1)[:, None]
    v -= (np.einsum('ij,ij->i', v, n))[:, None] * n
    a = accel(n)
    v += 0.5*dt*a; v *= 0.95
v[:] = 0.0
a = accel(n)
E0 = energy(n, v)
print(f"relaxed: E={E0:.2f}  Q(c1,0.08)={charge(n,c1,0.08):+.3f}  Q(c1,0.15)={charge(n,c1,0.15):+.3f}  "
      f"Q(c2,0.08)={charge(n,c2,0.08):+.3f}")

t0 = time.time()
for s in range(1, 12001):
    v += 0.5*dt*a
    n += dt*v; n /= np.linalg.norm(n, axis=1)[:, None]
    v -= (np.einsum('ij,ij->i', v, n))[:, None] * n
    a = accel(n)
    v += 0.5*dt*a
    if s % 1500 == 0:
        print(f"t={s*dt:6.1f}: E={energy(n,v):.2f}  Q(c1,0.08)={charge(n,c1,0.08):+.3f}  "
              f"Q(c1,0.15)={charge(n,c1,0.15):+.3f}  Q(c2,0.08)={charge(n,c2,0.08):+.3f}  [{time.time()-t0:.0f}s]")

np.save('/home/claude/hedgehog_n.npy', n)
