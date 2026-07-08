import numpy as np

# Compact phase (XY coupling sin(dtheta), no on-site): vortices protected by pi_1.
# TEST C: plain vortex ring, radius R0 -> expect shrink (line tension).
# TEST D: twisted ring (n_t phase turns along the ring, Hopf-like) -> does twist
#         slow/stop the shrink? Measure energy-weighted ring radius over time.

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N = len(pos)
i0, i1 = pairs[:, 0], pairs[:, 1]
k, dt = 1.0, 0.02
ctr = np.array([0.5, 0.5, 0.5])

d = pos - ctr; d -= L * np.round(d / L)
x, y, z = d[:, 0], d[:, 1], d[:, 2]
rho = np.sqrt(x**2 + y**2)          # cylindrical radius (ring in xy-plane)
alpha = np.arctan2(y, x)            # azimuth along the ring

R0 = 0.15

def ring_field(n_twist):
    th = np.arctan2(z, rho - R0)    # winding 1 around the ring core circle
    return th + n_twist * alpha     # twist along the ring

def accel(th):
    s = k * np.sin(th[i1] - th[i0])
    a = np.zeros(N)
    np.add.at(a, i0, s)
    np.add.at(a, i1, -s)
    return a

def site_energy(th, v):
    e = 0.5 * v**2
    ee = 0.5 * k * (1 - np.cos(th[i1] - th[i0]))
    np.add.at(e, i0, ee)
    np.add.at(e, i1, ee)
    return e

def ring_radius(e):
    # energy-weighted rho over the top-energy nodes near the ring plane
    m = e > np.quantile(e, 0.95)
    return (e[m] * rho[m]).sum() / e[m].sum()

def run(n_twist, label, steps=9000, relax=600):
    th = ring_field(n_twist)
    v = np.zeros(N)
    a = accel(th)
    for s in range(relax):
        v += 0.5*dt*a; th += dt*v; a = accel(th)
        v += 0.5*dt*a; v *= 0.95
    v[:] = 0.0
    a = accel(th)
    e = site_energy(th, v)
    print(f"[{label}] relaxed: E={e.sum():8.2f}  R_eff={ring_radius(e):.4f} (R0={R0})")
    for s in range(1, steps+1):
        v += 0.5*dt*a; th += dt*v; a = accel(th)
        v += 0.5*dt*a
        if s % 1500 == 0:
            e = site_energy(th, v)
            print(f"[{label}] t={s*dt:6.1f}: E={e.sum():8.2f}  R_eff={ring_radius(e):.4f}")

print("=== TEST C: plain vortex ring (no twist) ===")
run(0, "C plain")
print("\n=== TEST D: twisted vortex ring (n_twist=2) ===")
run(2, "D twist2")
