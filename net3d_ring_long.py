import numpy as np, time
from scipy.spatial import cKDTree

# LONG RUN: vortex ring, t=1200 (60000 steps). Two controls:
# 1) core radius (azimuthally averaged energy density in (rho,z));
# 2) WINDING along a closed loop THREADING the ring (linking=1 -> total 2*pi).
#    Winding = sum of wrapped phase diffs along a dense node path; topологический
#    инвариант — если держится 2pi, кольцо живо независимо от шума метрики ядра.

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N = len(pos)
i0, i1 = pairs[:, 0], pairs[:, 1]
k, dt = 1.0, 0.02
ctr = np.array([0.5, 0.5, 0.5])

d = pos - ctr; d -= L * np.round(d / L)
x, y, z = d[:, 0], d[:, 1], d[:, 2]
rho = np.sqrt(x**2 + y**2)
R0 = 0.15
th = np.arctan2(z, rho - R0)

def accel(t):
    s = k * np.sin(t[i1] - t[i0])
    a = np.zeros(N)
    np.add.at(a, i0, s); np.add.at(a, i1, -s)
    return a

def site_energy(t, v):
    e = 0.5 * v**2
    ee = 0.5 * k * (1 - np.cos(t[i1] - t[i0]))
    np.add.at(e, i0, ee); np.add.at(e, i1, ee)
    return e

# core tracker
nb = 40
rb = np.linspace(0, 0.5, nb+1); zb = np.linspace(-0.5, 0.5, nb+1)
ri = np.clip(np.digitize(rho, rb)-1, 0, nb-1)
zi = np.clip(np.digitize(z, zb)-1, 0, nb-1)
cnt = np.zeros((nb, nb)); np.add.at(cnt, (ri, zi), 1.0); cnt[cnt == 0] = 1.0
rc = 0.5*(rb[:-1]+rb[1:]); zc = 0.5*(zb[:-1]+zb[1:])
R_, Z_ = np.meshgrid(rc, zc, indexing='ij')

def core(e):
    H = np.zeros((nb, nb)); np.add.at(H, (ri, zi), e)
    Hd = H / cnt
    thr = np.quantile(Hd[Hd > 0], 0.98)
    m = Hd >= thr; w = Hd[m]
    return (w*R_[m]).sum()/w.sum(), (w*Z_[m]).sum()/w.sum()

# winding loop threading the ring: rectangle in xz-plane
# axis segment z: -0.35 -> +0.35 at rho~0 (through the ring hole),
# out to x=0.42, back down, return to axis. Dense waypoints -> nearest nodes.
def loop_waypoints(step=0.015):
    pts = []
    for zz in np.arange(-0.35, 0.35, step):        pts.append((0.0, 0.0, zz))
    for xx in np.arange(0.0, 0.42, step):          pts.append((xx, 0.0, 0.35))
    for zz in np.arange(0.35, -0.35, -step):       pts.append((0.42, 0.0, zz))
    for xx in np.arange(0.42, 0.0, -step):         pts.append((xx, 0.0, -0.35))
    return np.array(pts) + ctr

wp = loop_waypoints() % L
tree = cKDTree(pos, boxsize=L)
_, idx = tree.query(wp)
# dedupe consecutive repeats, close the loop
path = [idx[0]]
for j in idx[1:]:
    if j != path[-1]: path.append(j)
if path[0] != path[-1]: path.append(path[0])
path = np.array(path)

def winding(t):
    dp = t[path[1:]] - t[path[:-1]]
    dp = (dp + np.pi) % (2*np.pi) - np.pi
    return dp.sum() / (2*np.pi)

v = np.zeros(N)
a = accel(th)
for s in range(600):
    v += 0.5*dt*a; th += dt*v; a = accel(th)
    v += 0.5*dt*a; v *= 0.95
v[:] = 0.0
a = accel(th)
e = site_energy(th, v)
r0_, z0_ = core(e)
print(f"relaxed: E={e.sum():.2f}  core rho={r0_:.4f}  winding={winding(th):+.3f} (loop {len(path)} nodes)")

t0 = time.time()
for s in range(1, 60001):
    v += 0.5*dt*a; th += dt*v; a = accel(th)
    v += 0.5*dt*a
    if s % 6000 == 0:
        e = site_energy(th, v)
        rC, zC = core(e)
        print(f"t={s*dt:7.1f}: E={e.sum():.2f}  core rho={rC:.4f} z={zC:+.4f}  winding={winding(th):+.3f}  [{time.time()-t0:.0f}s]")
