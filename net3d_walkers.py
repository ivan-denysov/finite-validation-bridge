import numpy as np, time

# PROPAGATION MODULE. Transition = walker (flow conserved at node: enters, continues).
# Locks = synapses on links (reinforced by traffic, closed by gravity decay).
# IVAN'S CURVATURE RULE: link up to the lock is curved; mass/traffic straightens it.
# -> turning costs delay -> transition prefers straight continuation:
#    p(e_out) ~ (u+u0) * ((1+cos a)/2)^beta,  a = turn angle. beta=1 leading order.
# VARIANTS: C-full (beta=1), C-noturn (beta=0, curvature OFF) - isolates the rule.
# TARGET METRIC (was 0.336=isotropic without propagation): neighbor DIRECTOR
# correlation <(n_i . n_j)^2> vs 1/3. Directors from per-node Q-tensor of u-weights.

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N, E = len(pos), len(pairs)
i0, i1 = pairs[:, 0], pairs[:, 1]
dv = pos[i1] - pos[i0]; dv -= L * np.round(dv / L)
udir = dv / np.linalg.norm(dv, axis=1)[:, None]

# padded incidence: for each node, incident link ids and outgoing unit dirs
deg = np.zeros(N, int)
np.add.at(deg, i0, 1); np.add.at(deg, i1, 1)
Dmax = deg.max()
LID = np.full((N, Dmax), -1, int)          # link ids
ODIR = np.zeros((N, Dmax, 3))              # direction away from node
NBR = np.full((N, Dmax), -1, int)          # neighbor node
fill = np.zeros(N, int)
for e in range(E):
    a, b = i0[e], i1[e]
    LID[a, fill[a]] = e; ODIR[a, fill[a]] = udir[e];  NBR[a, fill[a]] = b; fill[a] += 1
    LID[b, fill[b]] = e; ODIR[b, fill[b]] = -udir[e]; NBR[b, fill[b]] = a; fill[b] += 1
MASK = LID >= 0

DELTA, GAMMA, U0 = 1.0, 0.05, 0.05
W = 4000
TICKS = 2500
rng = np.random.default_rng(5)
good = np.where(deg >= 3)[0]

def run(beta, tag):
    u = rng.random(E) * 0.1
    wnode = rng.choice(good, W)                       # walker positions
    wdir = rng.normal(size=(W, 3))
    wdir /= np.linalg.norm(wdir, axis=1)[:, None]     # arrival directions
    pers = []
    t0 = time.time()
    for t in range(1, TICKS + 1):
        lid = LID[wnode]; odir = ODIR[wnode]; mask = MASK[wnode]
        w_syn = np.where(mask, u[np.clip(lid, 0, E-1)] + U0, 0.0)
        ca = np.einsum('wj,wdj->wd', wdir, odir)
        w_turn = ((1.0 + ca) / 2.0) ** beta if beta > 0 else 1.0
        wgt = w_syn * w_turn * mask + 1e-12 * mask
        # vectorized categorical sampling (Gumbel argmax)
        g = -np.log(-np.log(rng.random(wgt.shape)))
        pick = np.argmax(np.where(mask, np.log(wgt) + g, -np.inf), axis=1)
        rows = np.arange(W)
        e_out = lid[rows, pick]
        d_out = odir[rows, pick]
        pers.append(np.einsum('wj,wj->w', wdir, d_out).mean())
        np.add.at(u, e_out, DELTA)
        u *= (1.0 - GAMMA)
        wnode = NBR[wnode, pick]
        wdir = d_out
    # per-node Q-tensor from u-weights -> director
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
    a = rng.choice(idx, m.sum()); b = rng.choice(idx, m.sum())
    c2r = np.einsum('ij,ij->i', n[a], n[b])**2
    top = np.sort(u)[::-1]
    print(f"[{tag}] neighbor <cos^2>={c2.mean():.3f} (random {c2r.mean():.3f}, iso 0.333)   "
          f"path persistence <cos step>={np.mean(pers[-200:]):.3f}   "
          f"traffic concentration: top10% links hold {top[:E//10].sum()/u.sum()*100:.0f}% of u   "
          f"[{time.time()-t0:.0f}s]")
    return u, n, ok

print("=== C-noturn: propagation + synapse, curvature OFF (beta=0) ===")
u0_, n0_, ok0_ = run(0.0, "C-noturn")
print("=== C-full: propagation + synapse + Ivan's curvature rule (beta=1) ===")
u1_, n1_, ok1_ = run(1.0, "C-full  ")
print("=== C-sharp: stronger straightening (beta=3) ===")
u3_, n3_, ok3_ = run(3.0, "C-sharp ")
np.save('/home/claude/chan_u_beta1.npy', u1_)
np.save('/home/claude/chan_n_beta1.npy', n1_)
