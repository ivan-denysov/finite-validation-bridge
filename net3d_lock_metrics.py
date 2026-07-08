import numpy as np

# Clean observables, threshold-free:
# 1) per-node participation ratio PR = (sum u)^2 / (deg * sum u^2) over incident links:
#    1.0 = uniform (ball), 1/deg = single winner. Degree-robust.
# 2) u-weighted node vector |v| = |sum u_e * dir_e| / sum u_e, vs UNIFORM-weight
#    baseline on the same star geometry (isotropy residual ~ 1/sqrt(Z)).
# 3) winner persistence in A: does the argmax link per node stay put (direction memory)?

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N, E = len(pos), len(pairs)
i0, i1 = pairs[:, 0], pairs[:, 1]
dv = pos[i1] - pos[i0]; dv -= L * np.round(dv / L)
udir = dv / np.linalg.norm(dv, axis=1)[:, None]

inc = [[] for _ in range(N)]; sgn = [[] for _ in range(N)]
for e in range(E):
    inc[i0[e]].append(e); sgn[i0[e]].append(+1.0)
    inc[i1[e]].append(e); sgn[i1[e]].append(-1.0)
inc = [np.array(a) for a in inc]; sgn = [np.array(a) for a in sgn]
deg = np.array([len(a) for a in inc])
active = np.where(deg >= 3)[0]

def metrics(u, tag):
    prs, vmags = [], []
    for i in active:
        e = inc[i]; w = u[e]
        s = w.sum()
        if s <= 0: continue
        prs.append(s * s / (len(e) * (w * w).sum()))
        v = (udir[e] * (w * sgn[i])[:, None]).sum(0)
        vmags.append(np.linalg.norm(v) / s)
    print(f"[{tag}] PR={np.mean(prs):.3f} (1=ball, {1/deg[active].mean():.3f}=winner)   "
          f"|v|_u={np.mean(vmags):.3f}")

# uniform baseline on same geometry
u_unif = np.ones(E)
metrics(u_unif, "BASELINE uniform")
uA = np.load('/home/claude/locks_uA.npy')
uB = np.load('/home/claude/locks_uB.npy')
metrics(uB, "MODEL B (smeared)")
metrics(uA, "MODEL A (unitary)")

# winner persistence in A: continue dynamics 600 ticks, check argmax stability
DELTA, GAMMA, U0 = 1.0, 0.05, 0.05
rng = np.random.default_rng(11)
u = uA.copy()
win0 = np.array([inc[i][np.argmax(u[inc[i]])] for i in active])
checks = []
for t in range(1, 601):
    fired = np.zeros(E)
    for i in active:
        e = inc[i]; w = u[e] + U0
        j = e[rng.choice(len(e), p=w / w.sum())]
        fired[j] += DELTA
    u += fired
    u *= (1.0 - GAMMA)
    if t % 150 == 0:
        win = np.array([inc[i][np.argmax(u[inc[i]])] for i in active])
        same = (win == win0).mean()
        checks.append(same)
        print(f"A persistence t=+{t}: same winner as t0 = {same*100:.1f}%")
