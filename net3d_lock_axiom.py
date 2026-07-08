import numpy as np

# VECTOR AXIOM TEST. Ivan's lock rule: lock = synapse on the LINK (intersection).
# Opens at threshold potential (accumulate, or at once if enough); closes by decay
# (gravity pulls it shut). Plus the programme's own law: differentiation is UNITARY
# (one transition per node per tick, indivisible).
# MODEL A (axiom): each node fires ONE transition per tick through one incident link,
#   preferring lower delay = more open lock (p ~ u + u0). Firing feeds the lock's
#   potential; gravity decays all.
# MODEL B (control): same synapse, but the transition is SMEARED equally over all
#   incident links (unitarity OFF). Everything else identical.
# QUESTION: does A spontaneously break the ball (|v|>0, directed regime) while B stays
# isotropic? If yes: the S^2 vacuum (gap of the closed-ball state) is DERIVED from
# lock-as-synapse + unitarity, not inserted.

D = np.load('/home/claude/net3d.npz')
pos, pairs, L = D['pos'], D['pairs'], float(D['L'])
N = len(pos)
E = len(pairs)
i0, i1 = pairs[:, 0], pairs[:, 1]

# unit direction of each link (min image)
dv = pos[i1] - pos[i0]; dv -= L * np.round(dv / L)
udir = dv / np.linalg.norm(dv, axis=1)[:, None]

# incidence lists
inc = [[] for _ in range(N)]
sgn = [[] for _ in range(N)]
for e in range(E):
    inc[i0[e]].append(e); sgn[i0[e]].append(+1.0)
    inc[i1[e]].append(e); sgn[i1[e]].append(-1.0)
inc = [np.array(a) for a in inc]
sgn = [np.array(a) for a in sgn]
deg = np.array([len(a) for a in inc])
active = np.where(deg >= 3)[0]          # skip isolated

DELTA, GAMMA, THETA, U0 = 1.0, 0.05, 8.0, 0.05
TICKS = 3000
rng = np.random.default_rng(7)

def order_params(u, tag):
    open_ = u >= THETA
    # per-node vector from OPEN incident locks
    vmag = np.zeros(N); nopen = np.zeros(N)
    for i in active:
        e = inc[i]; o = open_[e]
        nopen[i] = o.sum()
        if o.any():
            v = (udir[e[o]] * sgn[i][o, None]).sum(0)
            vmag[i] = np.linalg.norm(v) / o.sum()
    m = active
    print(f"[{tag}] open links: {open_.mean()*100:5.1f}%  "
          f"<n_open/node>={nopen[m].mean():.2f}  "
          f"<|v|> (open nodes)={vmag[m][nopen[m]>0].mean() if (nopen[m]>0).any() else 0:.3f}  "
          f"nodes with exactly 1 open: {(nopen[m]==1).mean()*100:.1f}%")
    return open_

def run(unitary, tag):
    u = rng.random(E) * 0.1            # small random potentials (noise seed)
    for t in range(1, TICKS + 1):
        if unitary:
            # each node: ONE transition through one incident link, p ~ u + U0
            fired = np.zeros(E)
            for i in active:
                e = inc[i]
                w = u[e] + U0
                j = e[rng.choice(len(e), p=w / w.sum())]
                fired[j] += DELTA
            u += fired
        else:
            # smeared: each node feeds ALL incident links equally (same total input)
            add = np.zeros(E)
            for i in active:
                e = inc[i]
                add[e] += DELTA / len(e)
            u += add
        u *= (1.0 - GAMMA)             # gravity pulls locks shut
        if t in (100, 500, 1500, 3000):
            order_params(u, f"{tag} t={t}")
    return u

print(f"network: N={N}, E={E}, <Z>={deg.mean():.2f}; DELTA={DELTA} GAMMA={GAMMA} "
      f"THETA={THETA} (steady u for always-fired link = {DELTA/GAMMA:.1f}, "
      f"for uniform sharing = {DELTA/(GAMMA*deg[active].mean()):.1f})")
print("\n=== MODEL A: unitary transition (the axiom) ===")
uA = run(True, "A")
print("\n=== MODEL B: smeared transition (unitarity OFF, control) ===")
uB = run(False, "B")

# persistence of chosen direction in A: correlation of winner link over time
np.save('/home/claude/locks_uA.npy', uA)
np.save('/home/claude/locks_uB.npy', uB)
