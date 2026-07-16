# -*- coding: utf-8 -*-
"""
IN-HOUSE PROBE v2 (finish debt #2, c_T = c / GW170817): is the transverse
degeneracy EXACT (a symmetry statement) or just the ~1% coincidence v1 saw?

Claim: in an ISOTROPIC linear-elastic medium the transverse speed c_T is
EXACTLY independent of polarisation and propagation direction -- it follows from
rotational symmetry (the isotropic elastic tensor has a single shear modulus mu).
An amorphous substrate restores rotational symmetry only on AVERAGE; the residual
anisotropy is a finite-size fluctuation that must VANISH as N -> infinity
(self-averaging). If the measured spread of c_T across random shear orientations
falls with N, the degeneracy is exact in the continuum -- so c_GW = c_light is a
symmetry identity, not a tuned ratio, once both are transverse.

Test: for several system sizes N, measure c_T over many RANDOM shear orientations
(random orthonormal n, m; strain A = gamma * m n^T -> shear wave prop n, pol m).
Report the anisotropy std(c_T)/mean(c_T) vs N.

Prediction: anisotropy decreases with N (toward 0). Longitudinal c_L stays
distinct. Conclusion: transverse degeneracy is symmetry-exact; the SINGLE
remaining hinge for GW170817 is 'is light a transverse substrate mode?' (the
Maxwell/EM-emergence debt) -- no free 1e-15 tuning anywhere.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

def make_substrate(N, seed):
    rng = np.random.RandomState(seed); L = N**(1/3)
    pos = rng.uniform(0, L, (N,3))
    def nb(rc):
        nc=max(1,int(L/rc)); cs=L/nc; grid={}; idx=(pos//cs).astype(int)%nc
        for i in range(N): grid.setdefault(tuple(idx[i]),[]).append(i)
        I,J=[],[]
        for i in range(N):
            ci=idx[i]; cand=[]
            for a in(-1,0,1):
                for b in(-1,0,1):
                    for c in(-1,0,1):
                        cand+=grid.get(((ci[0]+a)%nc,(ci[1]+b)%nc,(ci[2]+c)%nc),[])
            for j in cand:
                if j>i:
                    d=pos[j]-pos[i]; d-=L*np.round(d/L)
                    if d@d<rc*rc: I.append(i); J.append(j)
        return np.array(I),np.array(J)
    rc=1.6
    for _ in range(8):
        I,J=nb(rc); Z=2*len(I)/N
        if abs(Z-5.5)<0.12: break
        rc*=(5.5/Z)**(1/3)
    bonds=np.stack([I,J],1); BV=pos[J]-pos[I]; BV-=L*np.round(BV/L)
    NH=BV/np.linalg.norm(BV,axis=1)[:,None]
    return bonds,BV,NH,L,Z

KPERP=0.1
def solve_modulus(bonds,BV,NH,L,N,A):
    Kb=NH[:,:,None]*NH[:,None,:]+KPERP*(np.eye(3)[None]-NH[:,:,None]*NH[:,None,:])
    Nb=len(bonds); ii=bonds[:,0]; jj=bonds[:,1]
    loc=np.zeros((Nb,6,6)); loc[:,:3,:3]=Kb; loc[:,3:,3:]=Kb; loc[:,:3,3:]=-Kb; loc[:,3:,:3]=-Kb
    gi=np.stack([3*ii,3*ii+1,3*ii+2,3*jj,3*jj+1,3*jj+2],1)
    rows=np.broadcast_to(gi[:,:,None],(Nb,6,6)).ravel(); cols=np.broadcast_to(gi[:,None,:],(Nb,6,6)).ravel()
    H=sparse.coo_matrix((loc.ravel(),(rows,cols)),shape=(3*N,3*N)).tocsr(); H=(H+H.T)*0.5
    g=1e-3; d_aff=BV@(g*A).T
    f=np.einsum('bij,bj->bi',Kb,d_aff); b=np.zeros(3*N)
    np.add.at(b,np.stack([3*ii,3*ii+1,3*ii+2],1),f); np.add.at(b,np.stack([3*jj,3*jj+1,3*jj+2],1),-f)
    u,_=cg(H,-b,rtol=1e-6,maxiter=6000)
    du=d_aff+(u[np.stack([3*jj,3*jj+1,3*jj+2],1)]-u[np.stack([3*ii,3*ii+1,3*ii+2],1)])
    E=0.5*np.einsum('bi,bij,bj->',du,Kb,du); return 2*E/(L**3*g**2)

rngO=np.random.RandomState(7)
def rand_shear():
    n=rngO.randn(3); n/=np.linalg.norm(n)
    m=rngO.randn(3); m-=m@n*n; m/=np.linalg.norm(m)
    return np.outer(m,n)                 # shear: pol m, propagation n

print("finite-size scaling of the transverse-speed anisotropy (KPERP=0.1):\n")
print(f"{'N':>6} {'Z':>6} {'<c_T>':>8} {'std/mean %':>11} {'c_L':>7}")
for N in [1000, 2500, 5000]:
    bonds,BV,NH,L,Z=make_substrate(N,1); rho=N/L**3
    cs=[]
    for _ in range(6):
        G=solve_modulus(bonds,BV,NH,L,N,rand_shear()); cs.append(np.sqrt(max(G,0)/rho))
    cs=np.array(cs)
    Along=solve_modulus(bonds,BV,NH,L,N,np.diag([1.0,0,0])); cL=np.sqrt(max(Along,0)/rho)
    print(f"{N:>6} {Z:>6.3f} {cs.mean():>8.4f} {cs.std()/cs.mean()*100:>11.2f} {cL:>7.3f}")
print()
print("READING: if std/mean falls with N, the transverse speed becomes")
print("polarisation- and direction-INDEPENDENT in the continuum -- c_T is a single")
print("number by rotational symmetry, exactly. Then c_GW = c_light is a SYMMETRY")
print("IDENTITY (both transverse waves of one isotropic substrate), not a tuned")
print("1e-15 ratio. The one remaining hinge is 'is light a transverse mode?' --")
print("the Maxwell/EM-emergence debt, now the sole load-bearing assumption.")
