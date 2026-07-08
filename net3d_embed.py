import numpy as np, time

# DYNAMIC EMBEDDING, phase 1 (sanity, no clamps).
# Node positions are variables. Traffic tension pulls link endpoints into line
# ("influence on space" taken literally); rest-length springs resist collapse.
# Compare embedding ON vs OFF (frozen):
#  (a) step persistence <cos>  — does the geometric ceiling (~0.55) lift?
#  (b) neighbor director <cos^2> — transverse order
#  (c) geometry health: mean edge len / rest len, mean node displacement.

D = np.load('/home/claude/net3d.npz')
P0, pairs, L = D['pos'].copy(), D['pairs'], float(D['L'])
N, E = len(P0), len(pairs)
i0, i1 = pairs[:, 0], pairs[:, 1]

deg = np.zeros(N, int); np.add.at(deg, i0, 1); np.add.at(deg, i1, 1)
Dmax = deg.max()
SLOT_E = np.full((N, Dmax), 0, int); SLOT_S = np.zeros((N, Dmax)); MASK = np.zeros((N, Dmax), bool)
NBR = np.full((N, Dmax), 0, int)
fill = np.zeros(N, int)
for e in range(E):
    a, b = i0[e], i1[e]
    SLOT_E[a, fill[a]] = e; SLOT_S[a, fill[a]] = +1; NBR[a, fill[a]] = b; MASK[a, fill[a]] = True; fill[a] += 1
    SLOT_E[b, fill[b]] = e; SLOT_S[b, fill[b]] = -1; NBR[b, fill[b]] = a; MASK[b, fill[b]] = True; fill[b] += 1
good = np.where(deg >= 3)[0]

DELTA, GAMMA, U0, LAM = 1.0, 0.05, 0.05, 10.0
ETA, GC, MU = 0.05, 0.0005, 2.0
KT, KR, MUP = 0.0008, 1.0, 0.05
W, TICKS = 8000, 5000
rng = np.random.default_rng(21)

def Dfac(ca):
    al = np.arccos(np.clip(ca, -1, 1)); s = np.sin(al/2)
    out = np.ones_like(al); m = s > 1e-9
    out[m] = (al[m]/2)/s[m]
    return out

def neighbor_c2(P, wlink):
    dv = P[i1] - P[i0]; dv -= L*np.round(dv/L)
    ud = dv / (np.linalg.norm(dv, axis=1) + 1e-12)[:, None]
    Q = np.zeros((N, 3, 3))
    for slot in range(Dmax):
        m = MASK[:, slot]
        w = np.zeros(N); w[m] = wlink[SLOT_E[m, slot]]
        o = SLOT_S[:, slot, None] * ud[SLOT_E[:, slot]]
        Q += (w * m)[:, None, None] * (o[:, :, None]*o[:, None, :] - np.eye(3)/3)
    n = np.zeros((N, 3)); ok = np.zeros(N, bool)
    for i in good:
        ev, evec = np.linalg.eigh(Q[i])
        n[i] = evec[:, -1]; ok[i] = ev[-1] > 1e-9
    m = ok[i0] & ok[i1]
    return (np.einsum('ij,ij->i', n[i0[m]], n[i1[m]])**2).mean()

def run(embed, tag):
    P = P0.copy()
    dv = P[i1] - P[i0]; dv -= L*np.round(dv/L)
    a_rest = np.linalg.norm(dv, axis=1)
    u = 0.1 + rng.random(E)*0.1
    c = np.full(E, 0.7)
    wnode = rng.choice(good, W)
    wdir = rng.normal(size=(W, 3)); wdir /= np.linalg.norm(wdir, axis=1)[:, None]
    pers = []
    t0 = time.time()
    for t in range(1, TICKS+1):
        dv = P[i1] - P[i0]; dv -= L*np.round(dv/L)
        ln = np.linalg.norm(dv, axis=1) + 1e-12
        ud = dv / ln[:, None]
        ODIR = SLOT_S[:, :, None] * ud[SLOT_E]
        lid = SLOT_E[wnode]; odir = ODIR[wnode]; mask = MASK[wnode]
        w_syn = np.where(mask, (u[lid] + U0)*np.exp(-MU*c[lid]), 0.0)
        ca = np.einsum('wj,wdj->wd', wdir, odir)
        wgt = np.where(mask, w_syn*np.exp(-LAM*(Dfac(ca)-1.0)), 0.0) + 1e-300
        g = -np.log(-np.log(rng.random(wgt.shape)))
        pick = np.argmax(np.where(mask, np.log(wgt)+g, -np.inf), axis=1)
        rows = np.arange(W)
        e_out = lid[rows, pick]; d_out = odir[rows, pick]
        perm = rng.permutation(W)
        _, first = np.unique(wnode[perm], return_index=True)
        movers = perm[first]
        stay = np.ones(W, bool); stay[movers] = False
        pers.append(np.einsum('wj,wj->w', wdir[movers], d_out[movers]).mean())
        np.add.at(u, e_out[movers], DELTA)
        u *= (1.0 - GAMMA)
        st = np.zeros(E); np.add.at(st, e_out[movers], 1.0)
        c -= ETA*c*st
        c += GC*(1.0 - c)
        np.clip(c, 0.0, 1.0, out=c)
        if embed:
            # tension ~ traffic memory pulls endpoints together; rest springs resist
            T = KT * u
            Fmag = (T - KR*np.maximum(a_rest - ln, -a_rest)*0.0) # tension part
            # forces: tension pulls i0 -> i1 and i1 -> i0; rest spring restores length
            f_t = T[:, None] * ud
            f_r = (KR * (ln - a_rest))[:, None] * ud
            F = np.zeros((N, 3))
            np.add.at(F, i0,  f_t + f_r)
            np.add.at(F, i1, -f_t - f_r)
            P += MUP * F
            P %= L
        newnode = NBR[wnode, pick]
        wnode = np.where(stay, wnode, newnode)
        wdir = np.where(stay[:, None], wdir, d_out)
        if t in (1000, 3000, 5000):
            dv2 = P[i1]-P[i0]; dv2 -= L*np.round(dv2/L)
            ln2 = np.linalg.norm(dv2, axis=1)
            disp = P - P0; disp -= L*np.round(disp/L)
            c2 = neighbor_c2(P, 1.0 - c)
            print(f"[{tag}] t={t}: persistence={np.mean(pers[-300:]):.3f}  "
                  f"nbr<cos^2>={c2:.3f}  len/rest={np.mean(ln2/a_rest):.3f}  "
                  f"<|disp|>/edge={np.linalg.norm(disp,axis=1).mean()/a_rest.mean():.2f}  [{time.time()-t0:.0f}s]")

print("=== EMBED OFF (frozen geometry, reference) ===")
run(False, "OFF")
print("=== EMBED ON (nodes move under traffic tension) ===")
run(True,  "ON ")
