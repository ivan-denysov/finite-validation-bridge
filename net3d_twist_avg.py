import numpy as np, time

# TIME-AVERAGED twist profile. Continue from saved twist_embed state (c, P).
# Warm-up 1000 ticks (fresh walkers), then accumulate per-bin Q-tensor over 3000
# ticks. Fluctuating channels average out; a static elastic twist, if present,
# survives averaging.

D = np.load('/home/claude/net3d.npz')
P0, pairs, L = D['pos'].copy(), D['pairs'], float(D['L'])
N, E = len(P0), len(pairs)
i0, i1 = pairs[:, 0], pairs[:, 1]
dv0 = P0[i1]-P0[i0]; dv0 -= L*np.round(dv0/L)
ud0 = dv0/np.linalg.norm(dv0, axis=1)[:, None]
xm = (P0[i0, 0]+0.5*dv0[:, 0]) % L

deg = np.zeros(N, int); np.add.at(deg, i0, 1); np.add.at(deg, i1, 1)
Dmax = deg.max()
SLOT_E = np.zeros((N, Dmax), int); SLOT_S = np.zeros((N, Dmax)); MASK = np.zeros((N, Dmax), bool)
NBR = np.zeros((N, Dmax), int)
fill = np.zeros(N, int)
for e in range(E):
    a, b = i0[e], i1[e]
    SLOT_E[a, fill[a]] = e; SLOT_S[a, fill[a]] = +1; NBR[a, fill[a]] = b; MASK[a, fill[a]] = True; fill[a] += 1
    SLOT_E[b, fill[b]] = e; SLOT_S[b, fill[b]] = -1; NBR[b, fill[b]] = a; MASK[b, fill[b]] = True; fill[b] += 1
good = np.where(deg >= 3)[0]

s1 = (xm >= 0.00) & (xm < 0.08); s2 = (xm >= 0.46) & (xm < 0.54)
clamped = s1 | s2
az = np.abs(ud0[:, 2]); ay = np.abs(ud0[:, 1])
u_cl = np.zeros(E); c_cl = np.ones(E)
u_cl[s1] = 6.0*az[s1]**2; c_cl[s1] = 1.0-az[s1]**2
u_cl[s2] = 6.0*ay[s2]**2; c_cl[s2] = 1.0-ay[s2]**2
frozen = np.zeros(N, bool)
frozen[P0[:, 0] < 0.09] = True
frozen[(P0[:, 0] >= 0.45) & (P0[:, 0] < 0.55)] = True

DELTA, GAMMA, U0, LAM = 1.0, 0.05, 0.05, 10.0
ETA, GC, MU = 0.05, 0.0005, 2.0
KT, KR, MUP = 0.006, 1.0, 0.05
W = 8000
rng = np.random.default_rng(29)

P = np.load('/home/claude/twist_embed_P.npy')
c = np.load('/home/claude/twist_embed_c.npy')
a_rest = np.linalg.norm(dv0, axis=1)
u = 0.1 + rng.random(E)*0.1
u[clamped] = u_cl[clamped]; c[clamped] = c_cl[clamped]
wnode = rng.choice(good, W)
wdir = rng.normal(size=(W, 3)); wdir /= np.linalg.norm(wdir, axis=1)[:, None]

def Dfac(ca):
    al = np.arccos(np.clip(ca, -1, 1)); s = np.sin(al/2)
    out = np.ones_like(al); m = s > 1e-9
    out[m] = (al[m]/2)/s[m]
    return out

nb = 12
Qacc = np.zeros((nb, 3, 3)); Wacc = np.zeros(nb)

t0 = time.time()
WARM, ACC = 1000, 3000
for t in range(1, WARM+ACC+1):
    dv = P[i1]-P[i0]; dv -= L*np.round(dv/L)
    ln = np.linalg.norm(dv, axis=1)+1e-12
    ud = dv/ln[:, None]
    ODIR = SLOT_S[:, :, None]*ud[SLOT_E]
    lid = SLOT_E[wnode]; odir = ODIR[wnode]; mask = MASK[wnode]
    w_syn = np.where(mask, (u[lid]+U0)*np.exp(-MU*c[lid]), 0.0)
    ca = np.einsum('wj,wdj->wd', wdir, odir)
    wgt = np.where(mask, w_syn*np.exp(-LAM*(Dfac(ca)-1.0)), 0.0)+1e-300
    g = -np.log(-np.log(rng.random(wgt.shape)))
    pick = np.argmax(np.where(mask, np.log(wgt)+g, -np.inf), axis=1)
    rows = np.arange(W)
    e_out = lid[rows, pick]; d_out = odir[rows, pick]
    perm = rng.permutation(W)
    _, first = np.unique(wnode[perm], return_index=True)
    movers = perm[first]
    stay = np.ones(W, bool); stay[movers] = False
    np.add.at(u, e_out[movers], DELTA)
    u *= (1.0-GAMMA)
    st = np.zeros(E); np.add.at(st, e_out[movers], 1.0)
    c -= ETA*c*st
    c += GC*(1.0-c)
    np.clip(c, 0.0, 1.0, out=c)
    u[clamped] = u_cl[clamped]; c[clamped] = c_cl[clamped]
    f_t = (KT*u)[:, None]*ud
    f_r = (KR*(ln-a_rest))[:, None]*ud
    F = np.zeros((N, 3))
    np.add.at(F, i0, f_t+f_r); np.add.at(F, i1, -f_t-f_r)
    F[frozen] = 0.0
    P += MUP*F; P %= L
    newnode = NBR[wnode, pick]
    wnode = np.where(stay, wnode, newnode)
    wdir = np.where(stay[:, None], wdir, d_out)
    if t > WARM and t % 10 == 0:
        # accumulate per-bin Q from straightness-weighted link axes (link-based, cheap)
        s_w = 1.0 - c
        bx = np.clip((((P[i0, 0]+0.5*dv[:, 0]) % L)*nb).astype(int), 0, nb-1)
        o = ud
        Qe = s_w[:, None, None]*(o[:, :, None]*o[:, None, :]-np.eye(3)/3)
        for b in range(nb):
            m = bx == b
            Qacc[b] += Qe[m].sum(0); Wacc[b] += s_w[m].sum()

print(f"time-averaged profile over {ACC} ticks (every 10), [{time.time()-t0:.0f}s]:")
print("   x-bin    angle(z->y, deg)   S")
for b in range(nb):
    lo, hi = b/nb, (b+1)/nb
    if Wacc[b] <= 0: continue
    ev, evec = np.linalg.eigh(Qacc[b]/Wacc[b])
    n = evec[:, -1]; S = ev[-1]*1.5
    ang = np.degrees(np.arctan2(abs(n[1]), abs(n[2])))
    tag = " <clampZ" if hi <= 0.10 else (" <clampY" if (lo >= 0.44 and hi <= 0.56) else "")
    print(f"   [{lo:.2f},{hi:.2f})   {ang:6.1f}          {S:.3f}{tag}")
