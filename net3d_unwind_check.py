import numpy as np
from scipy.spatial import cKDTree

# 1) RING: fine-grained winding tracking t=0..120 (when exactly does it unwind?)
# 2) CONTROL: straight vortex line-antiline pair along z (periodic-consistent).
#    Lines cannot annihilate fast (separated by 0.5L); winding around ONE line
#    must hold -> validates the winding meter itself.

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N = len(pos)
i0, i1 = pairs[:, 0], pairs[:, 1]
k, dt = 1.0, 0.02
ctr = np.array([0.5, 0.5, 0.5])
d = pos - ctr; d -= L * np.round(d / L)
x, y, z = d[:, 0], d[:, 1], d[:, 2]
rho = np.sqrt(x**2 + y**2)
tree = cKDTree(pos, boxsize=L)

def accel(t):
    s = k * np.sin(t[i1] - t[i0])
    a = np.zeros(N); np.add.at(a, i0, s); np.add.at(a, i1, -s)
    return a

def make_path(waypoints):
    wp = (waypoints + ctr) % L
    _, idx = tree.query(wp)
    p = [idx[0]]
    for j in idx[1:]:
        if j != p[-1]: p.append(j)
    if p[0] != p[-1]: p.append(p[0])
    return np.array(p)

def winding(t, path):
    dp = t[path[1:]] - t[path[:-1]]
    dp = (dp + np.pi) % (2*np.pi) - np.pi
    return dp.sum() / (2*np.pi)

# ---------- 1) RING fine tracking ----------
R0 = 0.15
th = np.arctan2(z, rho - R0)
step = 0.015
pts = []
for zz in np.arange(-0.35, 0.35, step): pts.append((0.0, 0.0, zz))
for xx in np.arange(0.0, 0.42, step):   pts.append((xx, 0.0, 0.35))
for zz in np.arange(0.35, -0.35, -step):pts.append((0.42, 0.0, zz))
for xx in np.arange(0.42, 0.0, -step):  pts.append((xx, 0.0, -0.35))
ring_path = make_path(np.array(pts))

v = np.zeros(N); a = accel(th)
for s in range(600):
    v += 0.5*dt*a; th += dt*v; a = accel(th); v += 0.5*dt*a; v *= 0.95
v[:] = 0.0; a = accel(th)
print("=== RING: fine winding tracking ===")
print(f"t=   0.0  winding={winding(th, ring_path):+.3f}")
for s in range(1, 6001):
    v += 0.5*dt*a; th += dt*v; a = accel(th); v += 0.5*dt*a
    if s % 300 == 0:
        print(f"t={s*dt:6.1f}  winding={winding(th, ring_path):+.3f}")

# ---------- 2) CONTROL: line-antiline pair ----------
# vortex line at (x=0.25L,y=0.5L) winding +1, antiline at (0.75L,0.5L) winding -1
# (relative to box coords). Phase = sum of angles; periodic-consistent as a pair.
P = pos / L
a1 = np.arctan2(P[:,1]-0.5, P[:,0]-0.25)
a2 = np.arctan2(P[:,1]-0.5, P[:,0]-0.75)
th2 = a1 - a2

# loop around line 1 only: circle radius 0.12L in xy at z=0.5L
ang = np.arange(0, 2*np.pi, 0.06)
cpts = np.stack([0.25 + 0.12*np.cos(ang) - 0.5, 0.12*np.sin(ang), np.zeros_like(ang)], 1) * L
line_path = make_path(cpts)

v2 = np.zeros(N); a_ = accel(th2)
for s in range(600):
    v2 += 0.5*dt*a_; th2 += dt*v2; a_ = accel(th2); v2 += 0.5*dt*a_; v2 *= 0.95
v2[:] = 0.0; a_ = accel(th2)
print("\n=== CONTROL: line-antiline pair, winding around line 1 ===")
print(f"t=   0.0  winding={winding(th2, line_path):+.3f}")
for s in range(1, 6001):
    v2 += 0.5*dt*a_; th2 += dt*v2; a_ = accel(th2); v2 += 0.5*dt*a_
    if s % 600 == 0:
        print(f"t={s*dt:6.1f}  winding={winding(th2, line_path):+.3f}")
