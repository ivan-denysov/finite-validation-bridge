import numpy as np, time

# D2 BRIDGE TEST: twist transmission through the lock medium.
# Clamp slab1 x in [0.00,0.08]: director axis = z (aligned links: straight c=0, high u;
#   others: curved c=1, low u; held fixed every tick).
# Clamp slab2 x in [0.46,0.54]: axis = y.
# Interior evolves by the full axiom (derived arc penalty, fast u, slow curvature).
# ELASTIC medium => interior director rotates gradually z->y along x.
# NO stiffness => boundary layers only, disorder mid-gap.
# Profile: per x-bin Q-tensor (straightness-weighted), principal axis angle in (y,z).

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N, E = len(pos), len(pairs)
i0, i1 = pairs[:, 0], pairs[:, 1]
dv = pos[i1] - pos[i0]; dv -= L * np.round(dv / L)
udir = dv / np.linalg.norm(dv, axis=1)[:, None]
xm = (pos[i0, 0] + 0.5 * dv[:, 0]) % L      # link midpoint x

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

# link neighbors (links sharing an endpoint) for curvature spillover
from collections import defaultdict
node_links = defaultdict(list)
for e in range(E):
    node_links[i0[e]].append(e); node_links[i1[e]].append(e)
lnbr = [ [f for f in node_links[i0[e]] + node_links[i1[e]] if f != e] for e in range(E) ]
Kmax = max(len(a) for a in lnbr)
LNBR = np.full((E, Kmax), -1, int)
for e in range(E):
    LNBR[e, :len(lnbr[e])] = lnbr[e]
LNMASK = LNBR >= 0
LNDIR = udir[np.clip(LNBR, 0, E-1)] * LNMASK[:, :, None]

s1 = (xm >= 0.00) & (xm < 0.08)     # clamp axis z
s2 = (xm >= 0.46) & (xm < 0.54)     # clamp axis y
clamped = s1 | s2
az = np.abs(udir[:, 2])             # |cos| with z
ay = np.abs(udir[:, 1])
u_clamp = np.zeros(E); c_clamp = np.ones(E)
u_clamp[s1] = 6.0 * az[s1]**2; c_clamp[s1] = 1.0 - az[s1]**2
u_clamp[s2] = 6.0 * ay[s2]**2; c_clamp[s2] = 1.0 - ay[s2]**2

DELTA, GAMMA, U0, LAM = 1.0, 0.05, 0.05, 10.0
ETA, GC, MU = 0.05, 0.0005, 2.0
ETA2 = 0.03   # spillover: straightening of SPACE around the flow (anisotropic, ~cos^2)
W, TICKS = 8000, 6000
rng = np.random.default_rng(17)

def Dfac(ca):
    al = np.arccos(np.clip(ca, -1, 1)); s = np.sin(al/2)
    out = np.ones_like(al); m = s > 1e-9
    out[m] = (al[m]/2)/s[m]
    return out

def profile(wlink, nb=12):
    print("   x-bin    angle(deg from z, in yz)   order S   (clamp: z at x<0.08, y at 0.46-0.54)")
    for b in range(nb):
        lo, hi = b/nb, (b+1)/nb
        sel = good[(pos[good,0] >= lo) & (pos[good,0] < hi)]
        Q = np.zeros((3,3)); tot = 0.0
        for i in sel:
            for slot in range(deg[i]):
                e = LID[i, slot]
                w = wlink[e]
                o = ODIR[i, slot]
                Q += w * (np.outer(o, o) - np.eye(3)/3); tot += w
        if tot <= 0: continue
        ev, evec = np.linalg.eigh(Q/tot)
        n = evec[:, -1]; S = ev[-1] * 1.5
        ang = np.degrees(np.arctan2(abs(n[1]), abs(n[2])))   # 0=z, 90=y
        tagc = " <clampZ" if hi <= 0.10 else (" <clampY" if (lo >= 0.44 and hi <= 0.56) else "")
        print(f"   [{lo:.2f},{hi:.2f})   {ang:6.1f}                 {S:.3f}{tagc}")

u = 0.1 + rng.random(E) * 0.1
c = np.full(E, 0.7)
u[clamped] = u_clamp[clamped]; c[clamped] = c_clamp[clamped]
wnode = rng.choice(good, W)
wdir = rng.normal(size=(W, 3)); wdir /= np.linalg.norm(wdir, axis=1)[:, None]

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
    # B2 UNITARITY: one transition per NODE per tick — node exclusion
    perm = rng.permutation(W)
    _, first = np.unique(wnode[perm], return_index=True)
    movers = perm[first]                    # one winner per occupied node
    stay = np.ones(W, bool); stay[movers] = False
    np.add.at(u, e_out[movers], DELTA)
    u *= (1.0 - GAMMA)
    st = np.zeros(E); np.add.at(st, e_out[movers], 1.0)
    c -= ETA * c * st
    # spillover: traversed link straightens neighboring links ALONG the flow direction
    em = e_out[movers]; dm = d_out[movers]
    nb = LNBR[em]                                    # (M,K)
    aligns = (np.einsum('mkj,mj->mk', LNDIR[em], dm))**2 * LNMASK[em]
    flat = nb.ravel(); fa = aligns.ravel()
    vmask = flat >= 0
    dec = np.zeros(E); np.add.at(dec, flat[vmask], fa[vmask])
    c *= np.exp(-ETA2 * dec)
    c += GC * (1.0 - c)
    np.clip(c, 0.0, 1.0, out=c)
    u[clamped] = u_clamp[clamped]; c[clamped] = c_clamp[clamped]   # hold clamps
    newnode = NBR[wnode, pick]; wnode = np.where(stay, wnode, newnode)
    wdir = np.where(stay[:,None], wdir, d_out)
    if t in (1500, 6000):
        print(f"\n=== t={t}  [{time.time()-t0:.0f}s] — director profile (straightness-weighted) ===")
        profile(1.0 - c)
np.save('/home/claude/twist_c.npy', c); np.save('/home/claude/twist_u.npy', u)
