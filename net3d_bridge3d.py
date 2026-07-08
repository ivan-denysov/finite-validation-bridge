import numpy as np

# 3D LINEAR BRIDGE: Kuramoto-with-inertia + on-site sine (= discrete sine-Gordon)
# on the exact amorphous 3D network. June Bridge proved network=dSG in 1D exactly.
# Here: plane-wave dispersion in 3D must be omega^2 = m^2 + c^2 k^2, ISOTROPIC.
# 5 directions (3 axes + 2 diagonals) x 4 |k| values; frequency via projection
# amplitude zero-crossing count.

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N, E = len(pos), len(pairs)
i0, i1 = pairs[:, 0], pairs[:, 1]
k_spr, m2, dt = 1.0, 1.0, 0.01

def accel(th):
    dth = th[i1] - th[i0]
    a = np.zeros(N)
    np.add.at(a, i0, k_spr * dth)
    np.add.at(a, i1, -k_spr * dth)
    return a - m2 * np.sin(th)

dirs = {
    "x":   np.array([1, 0, 0.]),
    "y":   np.array([0, 1, 0.]),
    "z":   np.array([0, 0, 1.]),
    "xy":  np.array([1, 1, 0.]) / np.sqrt(2),
    "xyz": np.array([1, 1, 1.]) / np.sqrt(3),
}
print(f"{'dir':>4} {'n':>2} {'|k|':>6} {'omega':>7} {'omega^2':>8}")
res = {}
for name, dvec in dirs.items():
    for nharm in (1, 2, 3, 4):
        # periodic-consistent k: components must be 2*pi*int/L
        if name == "x":  kvec = 2*np.pi*nharm*np.array([1,0,0.])/L
        elif name == "y": kvec = 2*np.pi*nharm*np.array([0,1,0.])/L
        elif name == "z": kvec = 2*np.pi*nharm*np.array([0,0,1.])/L
        elif name == "xy": kvec = 2*np.pi*nharm*np.array([1,1,0.])/L
        else: kvec = 2*np.pi*nharm*np.array([1,1,1.])/L
        kmag = np.linalg.norm(kvec)
        phase = pos @ kvec
        mode = np.cos(phase)
        nrm = (mode**2).sum()
        th = 0.05 * mode
        v = np.zeros(N)
        a = accel(th)
        amp = []
        steps = 8000
        for s in range(steps):
            v += 0.5*dt*a; th += dt*v; a = accel(th); v += 0.5*dt*a
            amp.append((th*mode).sum()/nrm)
        amp = np.array(amp)
        # zero crossings -> frequency
        zc = np.where(np.diff(np.sign(amp)) != 0)[0]
        if len(zc) > 3:
            period = 2*(zc[-1]-zc[0])/ (len(zc)-1) * dt
            om = 2*np.pi/period
        else:
            om = np.nan
        res[(name, nharm)] = (kmag, om)
        print(f"{name:>4} {nharm:>2} {kmag:6.2f} {om:7.3f} {om*om:8.3f}")

# fit omega^2 = m2_fit + c2 * k^2 over all points
ks = np.array([v[0] for v in res.values()])
oms = np.array([v[1] for v in res.values()])
ok = np.isfinite(oms)
Afit = np.stack([np.ones(ok.sum()), ks[ok]**2], 1)
coef, *_ = np.linalg.lstsq(Afit, oms[ok]**2, rcond=None)
pred = Afit @ coef
r2 = 1 - ((oms[ok]**2-pred)**2).sum()/((oms[ok]**2-np.mean(oms[ok]**2))**2).sum()
print(f"\nfit: omega^2 = {coef[0]:.3f} + {coef[1]:.5f} * k^2   R^2={r2:.5f}   (m^2 input = {m2})")
# isotropy: compare omega across directions at same |k| (n=2: x,y,z share |k|)
oxx = [res[(d,2)][1] for d in ("x","y","z")]
print(f"isotropy check n=2 axes: omega x/y/z = {oxx[0]:.4f}/{oxx[1]:.4f}/{oxx[2]:.4f}  "
      f"spread={ (max(oxx)-min(oxx))/np.mean(oxx)*100:.2f}%")
