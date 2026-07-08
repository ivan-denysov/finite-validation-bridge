import numpy as np

# Precise ring-core tracking: azimuthally averaged energy density in (rho, z),
# core = energy-weighted centroid of top-decile bins. A real vortex ring should
# SELF-PROPEL along z at ~constant rho (smoke ring). Silent shrink is impossible
# without dissipation; decay would show as core rho -> 0 + radiation.

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
th = np.arctan2(z, rho - R0)   # plain vortex ring, winding 1

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

# (rho, z) histogram grid
nb = 40
rb = np.linspace(0, 0.5, nb+1)
zb = np.linspace(-0.5, 0.5, nb+1)
ri = np.clip(np.digitize(rho, rb)-1, 0, nb-1)
zi = np.clip(np.digitize(z, zb)-1, 0, nb-1)
# node count per bin for density normalization (bins at small rho hold few nodes)
cnt = np.zeros((nb, nb)); np.add.at(cnt, (ri, zi), 1.0)
cnt[cnt == 0] = 1.0
rc = 0.5*(rb[:-1]+rb[1:]); zc = 0.5*(zb[:-1]+zb[1:])

def core(e):
    H = np.zeros((nb, nb)); np.add.at(H, (ri, zi), e)
    Hd = H / cnt                              # energy DENSITY per bin
    thr = np.quantile(Hd[Hd > 0], 0.98)
    m = Hd >= thr
    w = Hd[m]
    R, Z = np.meshgrid(rc, zc, indexing='ij')
    return (w*R[m]).sum()/w.sum(), (w*Z[m]).sum()/w.sum()

v = np.zeros(N)
a = accel(th)
for s in range(600):    # gentle relax
    v += 0.5*dt*a; th += dt*v; a = accel(th)
    v += 0.5*dt*a; v *= 0.95
v[:] = 0.0
a = accel(th)
e = site_energy(th, v)
r0, z0 = core(e)
print(f"relaxed: E={e.sum():.2f}  core=(rho={r0:.4f}, z={z0:+.4f})")
for s in range(1, 12001):
    v += 0.5*dt*a; th += dt*v; a = accel(th)
    v += 0.5*dt*a
    if s % 1200 == 0:
        e = site_energy(th, v)
        rC, zC = core(e)
        print(f"t={s*dt:6.1f}: E={e.sum():.2f}  core=(rho={rC:.4f}, z={zC:+.4f})")
