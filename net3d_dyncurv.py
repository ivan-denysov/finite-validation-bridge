import numpy as np, time

# DYNAMIC CURVATURE (Ivan's rule as dynamics, two-timescale):
# fast: lock potential u_e (synapse: +DELTA on passage, decay GAMMA)
# slow: link curvature c_e in [0,1] (traffic straightens: c -= ETA*c on passage;
#       gravity re-curves slowly: c += GC*(1-c)... toward c0=1). GC << GAMMA.
# link delay ~ 1 + c  => preference exp(-MU*c). Turn penalty as derived (LAM).
# TEST: seed radial texture in CURVATURE (radial links straight), u random.
# Does slow geometry hold the director order A(t)? Control: uniform curvature.

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N, E = len(pos), len(pairs)
i0, i1 = pairs[:, 0], pairs[:, 1]
dv = pos[i1] - pos[i0]; dv -= L * np.round(dv / L)
udir = dv / np.linalg.norm(dv, axis=1)[:, None]
ctr = np.array([0.5, 0.5, 0.5])
dc = pos - ctr; dc -= L * np.round(dc / L)
rr = np.linalg.norm(dc, axis=1) + 1e-12
rhat = dc / rr[:, None]
mid = pos[i0] + 0.5 * dv
dm = mid - ctr; dm -= L * np.round(dm / L)
rhat_m = dm / (np.linalg.norm(dm, axis=1) + 1e-12)[:, None]
align = np.einsum('ej,ej->e', udir, rhat_m)**2

deg = np.zeros(N, int); np.add.at(deg, i0, 1); np.add.at(deg, i1, 1)
Dmax = deg.max()
LID = np.full((N, Dmax), -1, int); ODIR = np.zeros((N, Dmax, 3)); NBR = np.full((N, Dmax), -1, int)
fill = np.zeros(N, int)
for e in range(E):
    a, b = i0[e], i1[e]
    LID[a, fill[a]] = e; ODIR[a, fill[a]] = udir[e];  NBR[a, fill[a]] = b; fill[a] += 1
    LID[b, fill[b]] = e; ODIR[b, fill[b]] = -udir[e]; NBR[b, fill[b]] = a; fill[b] += 1
MASK = LID >= 0
good = np.where(deg >= 3)[0]
shell = good[(rr[good] > 0.10) & (rr[good] < 0.32)]

DELTA, GAMMA, U0, LAM = 1.0, 0.05, 0.05, 10.0
ETA, GC, MU = 0.05, 0.001, 2.0
W, TICKS = 4000, 5000
rng = np.random.default_rng(13)

def Dfac(ca):
    al = np.arccos(np.clip(ca, -1, 1)); s = np.sin(al/2)
    out = np.ones_like(al); m = s > 1e-9
    out[m] = (al[m]/2)/s[m]
    return out

def radial_order(w_link):
    Q = np.zeros((N, 3, 3))
    for slot in range(Dmax):
        m = MASK[:, slot]
        w = np.zeros(N); w[m] = w_link[LID[m, slot]]
        o = ODIR[:, slot]
        Q += w[:, None, None] * (o[:, :, None]*o[:, None, :] - np.eye(3)/3)
    A = []
    for i in shell:
        ev, evec = np.linalg.eigh(Q[i])
        if ev[-1] > 1e-9:
            A.append(np.dot(evec[:, -1], rhat[i])**2)
    return np.mean(A)

def run(seeded, tag):
    u = 0.1 + rng.random(E) * 0.1
    c = (1.0 - align) if seeded else np.full(E, np.mean(1.0 - align))
    wnode = rng.choice(good, W)
    wdir = rng.normal(size=(W, 3)); wdir /= np.linalg.norm(wdir, axis=1)[:, None]
    # order measured on STRAIGHTNESS (1-c): the geometry field
    print(f"[{tag}] t=0: A_geom={radial_order(1.0 - c):.3f}  A_u={radial_order(u):.3f} (iso 0.333)")
    t0 = time.time()
    for t in range(1, TICKS + 1):
        lid = LID[wnode]; odir = ODIR[wnode]; mask = MASK[wnode]
        lidc = np.clip(lid, 0, E-1)
        w_syn = np.where(mask, (u[lidc] + U0) * np.exp(-MU * c[lidc]), 0.0)
        ca = np.einsum('wj,wdj->wd', wdir, odir)
        wgt = np.where(mask, w_syn * np.exp(-LAM*(Dfac(ca)-1.0)), 0.0) + 1e-300
        g = -np.log(-np.log(rng.random(wgt.shape)))
        pick = np.argmax(np.where(mask, np.log(wgt)+g, -np.inf), axis=1)
        rows = np.arange(W)
        e_out = lid[rows, pick]; d_out = odir[rows, pick]
        np.add.at(u, e_out, DELTA)
        u *= (1.0 - GAMMA)
        straighten = np.zeros(E); np.add.at(straighten, e_out, 1.0)
        c -= ETA * c * straighten            # traffic straightens (Ivan)
        c += GC * (1.0 - c)                  # gravity re-curves slowly
        np.clip(c, 0.0, 1.0, out=c)
        wnode = NBR[wnode, pick]; wdir = d_out
        if t in (500, 1500, 3000, 5000):
            print(f"[{tag}] t={t}: A_geom={radial_order(1.0 - c):.3f}  A_u={radial_order(u):.3f}  [{time.time()-t0:.0f}s]")

print("=== SEEDED: radial texture in CURVATURE (slow geometry) ===")
run(True, "seed")
print("=== CONTROL: uniform curvature ===")
run(False, "ctrl")
