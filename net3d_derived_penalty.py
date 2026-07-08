import numpy as np, time

# DERIVED turn penalty. Minimal-curvature arc turning by alpha over chord a:
# arc length l = a * (alpha/2)/sin(alpha/2)  =>  delay factor D(alpha) = (a/2)/sin(a/2).
# Lock selection = race-to-open with noise => softmax in delays:
# p ~ (u+u0) * exp(-lam*(D(alpha)-1)), lam = straightening strength (dynamical: mass).
# Test lam in {3, 10}: do derived-form chains + stiffness reproduce the beta results?

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N, E = len(pos), len(pairs)
i0, i1 = pairs[:, 0], pairs[:, 1]
dv = pos[i1] - pos[i0]; dv -= L * np.round(dv / L)
udir = dv / np.linalg.norm(dv, axis=1)[:, None]

deg = np.zeros(N, int)
np.add.at(deg, i0, 1); np.add.at(deg, i1, 1)
Dmax = deg.max()
LID = np.full((N, Dmax), -1, int)
ODIR = np.zeros((N, Dmax, 3))
NBR = np.full((N, Dmax), -1, int)
fill = np.zeros(N, int)
for e in range(E):
    a, b = i0[e], i1[e]
    LID[a, fill[a]] = e; ODIR[a, fill[a]] = udir[e];  NBR[a, fill[a]] = b; fill[a] += 1
    LID[b, fill[b]] = e; ODIR[b, fill[b]] = -udir[e]; NBR[b, fill[b]] = a; fill[b] += 1
MASK = LID >= 0

DELTA, GAMMA, U0 = 1.0, 0.05, 0.05
W, TICKS = 4000, 2500
rng = np.random.default_rng(5)
good = np.where(deg >= 3)[0]

def Dfac(ca):
    al = np.arccos(np.clip(ca, -1, 1))
    s = np.sin(al / 2)
    out = np.ones_like(al)
    m = s > 1e-9
    out[m] = (al[m] / 2) / s[m]
    return out

def run(lam, tag):
    u = rng.random(E) * 0.1
    wnode = rng.choice(good, W)
    wdir = rng.normal(size=(W, 3)); wdir /= np.linalg.norm(wdir, axis=1)[:, None]
    pers = []
    t0 = time.time()
    for t in range(1, TICKS + 1):
        lid = LID[wnode]; odir = ODIR[wnode]; mask = MASK[wnode]
        w_syn = np.where(mask, u[np.clip(lid, 0, E-1)] + U0, 0.0)
        ca = np.einsum('wj,wdj->wd', wdir, odir)
        w_turn = np.exp(-lam * (Dfac(ca) - 1.0))
        wgt = np.where(mask, w_syn * w_turn, 0.0) + 1e-300
        g = -np.log(-np.log(rng.random(wgt.shape)))
        pick = np.argmax(np.where(mask, np.log(wgt) + g, -np.inf), axis=1)
        rows = np.arange(W)
        e_out = lid[rows, pick]; d_out = odir[rows, pick]
        pers.append(np.einsum('wj,wj->w', wdir, d_out).mean())
        np.add.at(u, e_out, DELTA)
        u *= (1.0 - GAMMA)
        wnode = NBR[wnode, pick]; wdir = d_out
    # directors from Q-tensor of u
    Q = np.zeros((N, 3, 3))
    for slot in range(Dmax):
        m = MASK[:, slot]
        w = np.zeros(N); w[m] = u[LID[m, slot]]
        o = ODIR[:, slot]
        Q += w[:, None, None] * (o[:, :, None] * o[:, None, :] - np.eye(3)/3)
    n = np.zeros((N, 3)); ok = np.zeros(N, bool)
    for i in good:
        ev, evec = np.linalg.eigh(Q[i])
        n[i] = evec[:, -1]; ok[i] = ev[-1] > 1e-9
    m = ok[i0] & ok[i1]
    c2 = np.einsum('ij,ij->i', n[i0[m]], n[i1[m]])**2
    idx = np.where(ok)[0]
    a_, b_ = rng.choice(idx, m.sum()), rng.choice(idx, m.sum())
    c2r = np.einsum('ij,ij->i', n[a_], n[b_])**2
    top = np.sort(u)[::-1]
    print(f"[{tag}] <cos^2>={c2.mean():.3f} (rand {c2r.mean():.3f})  "
          f"persistence={np.mean(pers[-200:]):.3f}  top10%={top[:E//10].sum()/u.sum()*100:.0f}%  "
          f"[{time.time()-t0:.0f}s]")

print("derived penalty D(a)=(a/2)/sin(a/2), race-softmax p~exp(-lam(D-1)):")
run(3.0,  "lam=3 ")
run(10.0, "lam=10")
