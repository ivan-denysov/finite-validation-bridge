import numpy as np, time

# SOLITON FINISH. MU=8 throughout.
# A: low carrier density (60 on ~59 nodes, ~1/node -> no B2 jams), t=20000:
#    does coherent circulation survive? plateau?
# B: density 300, t=20000: geometric plateau at working density.
# C: open arc 240deg, t=8000: closure control at MU=8.

D = np.load('/home/claude/net3d.npz')
P0, pairs, L = D['pos'].copy(), D['pairs'], float(D['L'])
N, E = len(P0), len(pairs)
i0, i1 = pairs[:, 0], pairs[:, 1]
ctr = np.array([0.5, 0.5, 0.5])
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
R0, TUBE = 0.28, 0.035

def channel_masks(arc_frac):
    dv = P0[i1]-P0[i0]; dv -= L*np.round(dv/L)
    mid = (P0[i0]+0.5*dv)-ctr; mid -= L*np.round(mid/L)
    rho = np.sqrt(mid[:, 0]**2+mid[:, 1]**2)+1e-12
    dist = np.sqrt((rho-R0)**2+mid[:, 2]**2)
    phi = np.arctan2(mid[:, 1], mid[:, 0])
    tang = np.stack([-mid[:, 1]/rho, mid[:, 0]/rho, np.zeros(E)], 1)
    ud = dv/np.linalg.norm(dv, axis=1)[:, None]
    ta = np.abs(np.einsum('ej,ej->e', ud, tang))
    in_arc = (phi >= -np.pi) & (phi <= -np.pi+2*np.pi*arc_frac)
    ch = (dist < TUBE) & (ta > 0.6) & in_arc
    dn = P0-ctr; dn -= L*np.round(dn/L)
    rn = np.sqrt(dn[:, 0]**2+dn[:, 1]**2)
    nd = np.sqrt((rn-R0)**2+dn[:, 2]**2)
    phin = np.arctan2(dn[:, 1], dn[:, 0])
    in_arc_n = (phin >= -np.pi) & (phin <= -np.pi+2*np.pi*arc_frac)
    nodes = np.where((nd < TUBE) & in_arc_n & (deg >= 3))[0]
    tangn = np.stack([-dn[nodes, 1], dn[nodes, 0], np.zeros(len(nodes))], 1)
    tangn /= np.linalg.norm(tangn, axis=1)[:, None]+1e-12
    return np.where(ch)[0], nodes, tangn

DELTA, GAMMA, U0, LAM = 1.0, 0.05, 0.05, 10.0
ETA, GC, MU = 0.05, 0.0005, 8.0
KT, KR, MUP = 0.006, 1.0, 0.05

def Dfac(ca):
    al = np.arccos(np.clip(ca, -1, 1)); s = np.sin(al/2)
    out = np.ones_like(al); m = s > 1e-9
    out[m] = (al[m]/2)/s[m]
    return out

def run(arc_frac, Wc, TICKS, tag, seed, checkpoints):
    rng = np.random.default_rng(seed)
    ch, cn, cntang = channel_masks(arc_frac)
    P = P0.copy()
    dv = P[i1]-P[i0]; dv -= L*np.round(dv/L)
    a_rest = np.linalg.norm(dv, axis=1)
    u = 0.1+rng.random(E)*0.1
    c = np.full(E, 0.7)
    u[ch] = 6.0; c[ch] = 0.05
    Wc = min(Wc, 6*len(cn))
    idxn = rng.choice(len(cn), Wc)
    wnode = cn[idxn]; wdir = cntang[idxn].copy()
    W = Wc
    print(f"[{tag}] links={len(ch)} nodes={len(cn)} carriers={W}")
    t0 = time.time()
    for t in range(1, TICKS+1):
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
        if t <= 500:
            u[ch] = np.maximum(u[ch], 6.0); c[ch] = np.minimum(c[ch], 0.05)
        f_t = (KT*u)[:, None]*ud
        f_r = (KR*(ln-a_rest))[:, None]*ud
        F = np.zeros((N, 3))
        np.add.at(F, i0, f_t+f_r); np.add.at(F, i1, -f_t-f_r)
        P += MUP*F; P %= L
        newnode = NBR[wnode, pick]
        wnode = np.where(stay, wnode, newnode)
        wdir = np.where(stay[:, None], wdir, d_out)
        if t in checkpoints:
            s_ch = (1.0-c[ch]).mean(); u_ch = u[ch].mean()
            dn = P[wnode]-ctr; dn -= L*np.round(dn/L)
            rn = np.sqrt(dn[:, 0]**2+dn[:, 1]**2)+1e-12
            nd = np.sqrt((rn-R0)**2+dn[:, 2]**2)
            intube = nd < TUBE*1.5
            tw = np.stack([-dn[:, 1]/rn, dn[:, 0]/rn, np.zeros(W)], 1)
            circ = np.einsum('wj,wj->w', wdir[intube], tw[intube]).mean() if intube.sum() > 0 else 0.0
            blocked = 1.0 - len(movers)/W
            print(f"[{tag}] t={t}: straight={s_ch:.3f} u_ch={u_ch:5.2f} "
                  f"in={intube.sum()}/{W} circ={circ:+.3f} blocked={blocked:.2f} "
                  f"<rho>={rn[intube].mean() if intube.sum()>0 else 0:.3f} [{time.time()-t0:.0f}s]")

import sys
which = sys.argv[1]
cp_long = (1000, 3000, 6000, 10000, 14000, 20000)
if which == "A": run(1.00, 60, 20000, "A-60 ", 41, cp_long)
elif which == "B": run(1.00, 300, 20000, "B-300", 43, cp_long)
else: run(0.667, 200, 8000, "C-arc", 47, (1000, 3000, 6000, 8000))
