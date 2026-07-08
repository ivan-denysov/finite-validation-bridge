import numpy as np
from scipy.spatial import cKDTree

# Exact 3D amorphous network, PERIODIC box (torus) -> true bulk, no edges.
# Target mean degree Z ~ 5.5 (fixed spec). Verify isotropy structurally.

rng = np.random.default_rng(1)
N = 8000
L = 1.0
pos = rng.random((N, 3)) * L

# radius for target Z: expected Z = N * (4/3 pi r^3) / L^3
Z_target = 5.5
r = (Z_target * 3 / (4 * np.pi * N)) ** (1/3)
tree = cKDTree(pos, boxsize=L)
pairs = tree.query_pairs(r, output_type='ndarray')

deg = np.zeros(N, int)
np.add.at(deg, pairs[:, 0], 1)
np.add.at(deg, pairs[:, 1], 1)
Z = deg.mean()

# edge vectors with minimum-image (periodic)
d = pos[pairs[:, 1]] - pos[pairs[:, 0]]
d -= L * np.round(d / L)
dl = np.linalg.norm(d, axis=1)
u = d / dl[:, None]

# isotropy: mean |components|, plus chi2 on cos(theta), phi
comp = np.abs(u).mean(axis=0)
aniso = comp.max() - comp.min()
ct = u[:, 2]
h_ct, _ = np.histogram(ct, bins=20, range=(-1, 1))
chi2_ct = (((h_ct - h_ct.mean())**2) / h_ct.mean()).sum()
phi = np.arctan2(u[:, 1], u[:, 0])
h_p, _ = np.histogram(phi, bins=20, range=(-np.pi, np.pi))
chi2_p = (((h_p - h_p.mean())**2) / h_p.mean()).sum()

# connectivity: giant component
import collections
adj = collections.defaultdict(list)
for a, b in pairs:
    adj[a].append(b); adj[b].append(a)
seen = np.zeros(N, bool)
stack = [0]; seen[0] = True; cnt = 1
while stack:
    x = stack.pop()
    for y in adj[x]:
        if not seen[y]:
            seen[y] = True; cnt += 1; stack.append(y)

print(f"N={N}  edges={len(pairs)}  Z={Z:.3f} (target {Z_target})")
print(f"isotropy: mean|ux,uy,uz| = {comp.round(4)}  anisotropy={aniso:.4f}")
print(f"chi2 cos(theta) [19 dof] = {chi2_ct:.1f}   chi2 phi [19 dof] = {chi2_p:.1f}")
print(f"giant component: {cnt}/{N} = {cnt/N:.3f}")
print(f"mean edge length = {dl.mean():.4f}, r_cut = {r:.4f}")

np.savez('/home/claude/net3d.npz', pos=pos, pairs=pairs, L=L, r=r)
