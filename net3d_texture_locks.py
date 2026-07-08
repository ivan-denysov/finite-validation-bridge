import numpy as np, time

# END-TO-END: director hedgehog ON THE LOCK DYNAMICS ITSELF.
# Seed link potentials radially around center: u_e ~ (dir_e . r_hat)^2.
# Run full walker+lock dynamics (derived arc penalty, lam=10).
# Order parameter A(t) = <(n_i . r_hat_i)^2> over shell 0.10<r<0.32 (iso = 1/3).
# CONTROL: same dynamics from random seed (A must stay ~1/3).
# Honest scope: persistence of the radial director texture, not full pi_2 charge proof.

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
rm = np.linalg.norm(dm, axis=1) + 1e-12
rhat_m = dm / rm[:, None]
align = np.einsum('ej,ej->e', udir, rhat_m)**2   # radial alignment of link

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
W, TICKS = 4000, 2500
rng = np.random.default_rng(9)

def Dfac(ca):
    al = np.arccos(np.clip(ca, -1, 1)); s = np.sin(al/2)
    out = np.ones_like(al); m = s > 1e-9
    out[m] = (al[m]/2)/s[m]
    return out

def radial_order(u):
    Q = np.zeros((N, 3, 3))
    for slot in range(Dmax):
        m = MASK[:, slot]
        w = np.zeros(N); w[m] = u[LID[m, slot]]
        o = ODIR[:, slot]
        Q += w[:, None, None] * (o[:, :, None]*o[:, None, :] - np.eye(3)/3)
    A = []
    for i in shell:
        ev, evec = np.linalg.eigh(Q[i])
        if ev[-1] > 1e-9:
            A.append(np.dot(evec[:, -1], rhat[i])**2)
    return np.mean(A)

def run(seeded, tag):
    if seeded:
        u = 0.1 + 6.0 * align       # radial texture in the locks
    else:
        u = 0.1 + 6.0 * rng.random(E)   # random, same scale (control)
    wnode = rng.choice(good, W)
    wdir = rng.normal(size=(W, 3)); wdir /= np.linalg.norm(wdir, axis=1)[:, None]
    print(f"[{tag}] t=0: A={radial_order(u):.3f} (iso 0.333)")
    t0 = time.time()
    for t in range(1, TICKS + 1):
        lid = LID[wnode]; odir = ODIR[wnode]; mask = MASK[wnode]
        w_syn = np.where(mask, u[np.clip(lid, 0, E-1)] + U0, 0.0)
        ca = np.einsum('wj,wdj->wd', wdir, odir)
        wgt = np.where(mask, w_syn * np.exp(-LAM*(Dfac(ca)-1.0)), 0.0) + 1e-300
        g = -np.log(-np.log(rng.random(wgt.shape)))
        pick = np.argmax(np.where(mask, np.log(wgt)+g, -np.inf), axis=1)
        rows = np.arange(W)
        e_out = lid[rows, pick]; d_out = odir[rows, pick]
        np.add.at(u, e_out, DELTA)
        u *= (1.0 - GAMMA)
        wnode = NBR[wnode, pick]; wdir = d_out
        if t in (250, 500, 1000, 1750, 2500):
            print(f"[{tag}] t={t}: A={radial_order(u):.3f}  [{time.time()-t0:.0f}s]")
    return u

print("=== SEEDED: radial (hedgehog-director) texture in locks ===")
uS = run(True, "seed")
print("=== CONTROL: random seed, same dynamics ===")
uC = run(False, "ctrl")
np.save('/home/claude/texture_u_seeded.npy', uS)
