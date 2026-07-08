import numpy as np, time, json

# Re-run with FULL time-series logging for two figures:
#  FIG 1: condensate straightness plateau (runs A 60, B 300, control arc), every 100 ticks.
#  FIG 2: 3D dispersion omega^2 vs k^2 with fit + axis isotropy (from net3d_bridge3d logic).
# Saves arrays to /home/claude for plotting; figures written in the next step.

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
    rho = np.sqrt(mid[:,0]**2+mid[:,1]**2)+1e-12
    dist = np.sqrt((rho-R0)**2+mid[:,2]**2)
    phi = np.arctan2(mid[:,1], mid[:,0])
    tang = np.stack([-mid[:,1]/rho, mid[:,0]/rho, np.zeros(E)],1)
    ud = dv/np.linalg.norm(dv,axis=1)[:,None]
    ta = np.abs(np.einsum('ej,ej->e', ud, tang))
    in_arc = (phi>=-np.pi)&(phi<=-np.pi+2*np.pi*arc_frac)
    ch = (dist<TUBE)&(ta>0.6)&in_arc
    dn = P0-ctr; dn -= L*np.round(dn/L)
    rn = np.sqrt(dn[:,0]**2+dn[:,1]**2)
    nd = np.sqrt((rn-R0)**2+dn[:,2]**2)
    phin = np.arctan2(dn[:,1], dn[:,0])
    in_arc_n = (phin>=-np.pi)&(phin<=-np.pi+2*np.pi*arc_frac)
    nodes = np.where((nd<TUBE)&in_arc_n&(deg>=3))[0]
    tangn = np.stack([-dn[nodes,1], dn[nodes,0], np.zeros(len(nodes))],1)
    tangn /= np.linalg.norm(tangn,axis=1)[:,None]+1e-12
    return np.where(ch)[0], nodes, tangn

DELTA,GAMMA,U0,LAM = 1.0,0.05,0.05,10.0
ETA,GC,MU = 0.05,0.0005,8.0
KT,KR,MUP = 0.006,1.0,0.05

def Dfac(ca):
    al=np.arccos(np.clip(ca,-1,1)); s=np.sin(al/2)
    out=np.ones_like(al); m=s>1e-9; out[m]=(al[m]/2)/s[m]; return out

def run(arc_frac, Wc, TICKS, seed):
    rng=np.random.default_rng(seed)
    ch,cn,cntang=channel_masks(arc_frac)
    P=P0.copy()
    dv=P[i1]-P[i0]; dv-=L*np.round(dv/L)
    a_rest=np.linalg.norm(dv,axis=1)
    u=0.1+rng.random(E)*0.1; c=np.full(E,0.7)
    u[ch]=6.0; c[ch]=0.05
    Wc=min(Wc,6*len(cn)); idxn=rng.choice(len(cn),Wc)
    wnode=cn[idxn]; wdir=cntang[idxn].copy(); W=Wc
    ts=[]; straight=[]; pop=[]; blocked=[]
    for t in range(1,TICKS+1):
        dv=P[i1]-P[i0]; dv-=L*np.round(dv/L)
        ln=np.linalg.norm(dv,axis=1)+1e-12; ud=dv/ln[:,None]
        ODIR=SLOT_S[:,:,None]*ud[SLOT_E]
        lid=SLOT_E[wnode]; odir=ODIR[wnode]; mask=MASK[wnode]
        w_syn=np.where(mask,(u[lid]+U0)*np.exp(-MU*c[lid]),0.0)
        ca=np.einsum('wj,wdj->wd',wdir,odir)
        wgt=np.where(mask,w_syn*np.exp(-LAM*(Dfac(ca)-1.0)),0.0)+1e-300
        g=-np.log(-np.log(rng.random(wgt.shape)))
        pick=np.argmax(np.where(mask,np.log(wgt)+g,-np.inf),axis=1)
        rows=np.arange(W); e_out=lid[rows,pick]; d_out=odir[rows,pick]
        perm=rng.permutation(W); _,first=np.unique(wnode[perm],return_index=True)
        movers=perm[first]; stay=np.ones(W,bool); stay[movers]=False
        np.add.at(u,e_out[movers],DELTA); u*=(1.0-GAMMA)
        st=np.zeros(E); np.add.at(st,e_out[movers],1.0)
        c-=ETA*c*st; c+=GC*(1.0-c); np.clip(c,0.0,1.0,out=c)
        if t<=500: u[ch]=np.maximum(u[ch],6.0); c[ch]=np.minimum(c[ch],0.05)
        f_t=(KT*u)[:,None]*ud; f_r=(KR*(ln-a_rest))[:,None]*ud
        F=np.zeros((N,3)); np.add.at(F,i0,f_t+f_r); np.add.at(F,i1,-f_t-f_r)
        P+=MUP*F; P%=L
        newnode=NBR[wnode,pick]; wnode=np.where(stay,wnode,newnode)
        wdir=np.where(stay[:,None],wdir,d_out)
        if t%100==0:
            dn=P[wnode]-ctr; dn-=L*np.round(dn/L)
            rn=np.sqrt(dn[:,0]**2+dn[:,1]**2)+1e-12
            nd=np.sqrt((rn-R0)**2+dn[:,2]**2); intube=nd<TUBE*1.5
            ts.append(t); straight.append((1.0-c[ch]).mean())
            pop.append(int(intube.sum())); blocked.append(1.0-len(movers)/W)
    return dict(t=ts, straight=straight, pop=pop, blocked=blocked, W=W, nlinks=len(ch))

print("run A (60)..."); A=run(1.00,60,20000,41)
print("run B (300).."); B=run(1.00,300,20000,43)
print("run arc......"); C=run(0.667,200,8000,47)
json.dump(dict(A=A,B=B,C=C), open('/home/claude/fig_condensate.json','w'))
print("condensate series saved")

# ---- dispersion series ----
k_spr,m2,dt=1.0,1.0,0.01
def accel(th):
    dth=th[i1]-th[i0]; a=np.zeros(N)
    np.add.at(a,i0,k_spr*dth); np.add.at(a,i1,-k_spr*dth); return a-m2*np.sin(th)
dirs={"x":[1,0,0],"y":[0,1,0],"z":[0,0,1],"xy":[1,1,0],"xyz":[1,1,1]}
pts=[]
for name,base in dirs.items():
    for n in (1,2,3,4):
        kvec=2*np.pi*n*np.array(base,float)/L; kmag=np.linalg.norm(kvec)
        mode=np.cos(P0@kvec); nrm=(mode**2).sum()
        th=0.05*mode; v=np.zeros(N); a=accel(th); amp=[]
        for s in range(8000):
            v+=0.5*dt*a; th+=dt*v; a=accel(th); v+=0.5*dt*a
            amp.append((th*mode).sum()/nrm)
        amp=np.array(amp); zc=np.where(np.diff(np.sign(amp))!=0)[0]
        om=2*np.pi/(2*(zc[-1]-zc[0])/(len(zc)-1)*dt) if len(zc)>3 else np.nan
        pts.append(dict(dir=name,n=n,k=float(kmag),om=float(om)))
json.dump(pts, open('/home/claude/fig_dispersion.json','w'))
print("dispersion series saved")
