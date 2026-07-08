import numpy as np, json
D=np.load('/home/claude/net3d.npz')
P0,pairs,L=D['pos'].copy(),D['pairs'],float(D['L'])
N,E=len(P0),len(pairs); i0,i1=pairs[:,0],pairs[:,1]
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
        om=2*np.pi/(2*(zc[-1]-zc[0])/(len(zc)-1)*dt) if len(zc)>3 else float('nan')
        pts.append(dict(dir=name,n=int(n),k=float(kmag),om=float(om)))
json.dump(pts, open('/home/claude/fig_dispersion.json','w'))
print("dispersion saved",len(pts))
