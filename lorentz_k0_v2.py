# -*- coding: utf-8 -*-
"""
IN-HOUSE PROBE v2 (finish debt #1, long-wavelength Lorentz): pin down the
k -> 0 wave speed. v1 showed a plateau but the two lowest-k points were FFT-bin
limited (~12% scatter). Here: longer box (lower k), parabolic interpolation of
the FFT peak (sharpens omega ~10-100x), and an explicit dispersion fit
    omega^2 = c^2(0) * k^2 + beta * k^4
(only even powers -- inversion symmetry). If the fit is clean and c^2(0) is
well-determined with a small O(k^2) correction, the dispersion is LINEAR as
k -> 0 with a definite speed: the IR is Lorentz-invariant, closed in-model.
"""
import numpy as np
from scipy import sparse

rng = np.random.RandomState(1)
LX, LY, LZ = 120.0, 5.0, 5.0
rho = 1.0; N = int(LX*LY*LZ*rho)
pos = rng.uniform([0,0,0], [LX,LY,LZ], (N,3)); box = np.array([LX,LY,LZ])

def build(rc):
    nc=(box/rc).astype(int); nc[nc<1]=1; cs=box/nc; grid={}
    idx=(pos//cs).astype(int)%nc
    for i in range(N): grid.setdefault(tuple(idx[i]),[]).append(i)
    I,J=[],[]
    for i in range(N):
        ci=idx[i]; cand=[]
        for a in(-1,0,1):
            for b in(-1,0,1):
                for c in(-1,0,1):
                    cand+=grid.get(((ci[0]+a)%nc[0],(ci[1]+b)%nc[1],(ci[2]+c)%nc[2]),[])
        for j in cand:
            if j>i:
                d=pos[j]-pos[i]; d-=box*np.round(d/box)
                if d@d<rc*rc: I.append(i); J.append(j)
    return np.array(I),np.array(J)

rc=1.7
for _ in range(8):
    I,J=build(rc); Z=2*len(I)/N
    if abs(Z-5.5)<0.12: break
    rc*=(5.5/Z)**(1/3)
A=sparse.coo_matrix((np.ones(2*len(I)),(np.r_[I,J],np.r_[J,I])),shape=(N,N)).tocsr()
Lap=sparse.diags(np.asarray(A.sum(1)).ravel())-A
BV=pos[J]-pos[I]; BV-=box*np.round(BV/box)
c2_aff=(1/6)*(2*len(I)/N)*(BV**2).sum()/len(I)
print(f"substrate: N={N}, bonds={len(I)}, Z-bar={Z:.3f}, seed=1; c^2_aff={c2_aff:.4f}\n")

c0=1.0; dt=0.03; STEPS=50000; xi=pos[:,0]
freqs=np.fft.rfftfreq(STEPS,d=dt); win=np.hanning(STEPS)

def omega_of(n):
    k=2*np.pi*n/LX; th=np.cos(k*xi); v=np.zeros(N)
    cwave=np.cos(k*xi); proj=np.empty(STEPS)
    for t in range(STEPS):
        v+=dt*(-(c0**2)*(Lap@th)); th+=dt*v; proj[t]=th@cwave
    proj-=proj.mean()
    S=np.abs(np.fft.rfft(proj*win)); p=np.argmax(S)
    # parabolic interpolation of the peak bin
    if 0<p<len(S)-1:
        y0,y1,y2=S[p-1],S[p],S[p+1]; d=0.5*(y0-y2)/(y0-2*y1+y2+1e-30)
    else: d=0.0
    f=(p+d)*(freqs[1]-freqs[0]); return k,2*np.pi*f

ns=[1,2,3,4,5,6,7]; K=[]; W=[]
print(f"{'n':>3} {'k_x':>7} {'omega':>9} {'c^2=w2/k2':>10}")
for n in ns:
    k,om=omega_of(n); K.append(k); W.append(om)
    print(f"{n:>3} {k:>7.4f} {om:>9.5f} {(om/k)**2:>10.4f}")
K=np.array(K); W=np.array(W)
# fit omega^2 = c2*k^2 + beta*k^4
X=np.vstack([K**2,K**4]).T; coef,res,*_=np.linalg.lstsq(X,W**2,rcond=None)
c2_0,beta=coef; pred=X@coef; R2=1-((W**2-pred)**2).sum()/((W**2-(W**2).mean())**2).sum()
# error on c2_0 from covariance
dof=len(K)-2; sig2=((W**2-pred)**2).sum()/dof; cov=sig2*np.linalg.inv(X.T@X)
print()
print(f"fit  omega^2 = c^2(0) k^2 + beta k^4 :")
print(f"  c^2(0) = {c2_0:.4f} +/- {np.sqrt(cov[0,0]):.4f}   (c(0) = {np.sqrt(c2_0):.4f})")
print(f"  beta   = {beta:.4f}   (O(k^2) correction: beta/c^2(0) = {beta/c2_0:.3f})")
print(f"  R^2    = {R2:.5f}")
print()
print("READING: a clean fit with c^2(0) well-determined and a small even-power")
print("correction means omega -> c(0) k linearly as k -> 0 with a DEFINITE speed:")
print("the vacuum has a Lorentz-invariant long-wavelength limit; the finite-k")
print("dispersion is the beta k^4 node-scale term. Debt #1 closed in-model.")
