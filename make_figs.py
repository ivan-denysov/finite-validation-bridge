import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A=json.load(open('fig_cond_A.json')); B=json.load(open('fig_cond_B.json')); C=json.load(open('fig_cond_C.json'))

# ---------- FIG 1: condensate plateau ----------
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.2))
ax1.plot(A['t'],A['straight'],lw=1.4,label=f"closed ring, 60 carriers")
ax1.plot(B['t'],B['straight'],lw=1.4,label=f"closed ring, 300 carriers")
ax1.plot(C['t'],C['straight'],lw=1.4,ls='--',color='gray',label="open arc 240° (control)")
# no-flow analytic decay reference from seed level 0.95 at GC=0.0005
t=np.array(A['t']); ax1.plot(t,0.95*np.exp(-0.0005*t),lw=1.0,ls=':',color='k',label="no-flow decay (ref)")
ax1.axhline(0.294,color='C0',lw=0.8,alpha=0.5)
ax1.axvline(500,color='k',lw=0.6,alpha=0.4); ax1.text(560,0.86,"seed released",fontsize=8,alpha=0.7)
ax1.set_xlabel("tick"); ax1.set_ylabel("channel straightness  (1 − curvature)")
ax1.set_title("(a) Self-bound flow condensate: straightness")
ax1.legend(fontsize=8,loc='upper right'); ax1.set_ylim(0,1)

ax2.plot(A['t'],A['pop'],lw=1.4,label="60 carriers → plateau")
ax2.plot(B['t'],B['pop'],lw=1.4,label="300 carriers → plateau")
ax2.plot(C['t'],C['pop'],lw=1.4,ls='--',color='gray',label="open arc (control)")
ax2.set_xlabel("tick"); ax2.set_ylabel("carriers inside the tube")
ax2.set_title("(b) Carrier population (steady)")
ax2.legend(fontsize=8,loc='center right')
plt.tight_layout(); plt.savefig('/mnt/user-data/outputs/fig_condensate.pdf',dpi=150)
plt.savefig('/mnt/user-data/outputs/fig_condensate.png',dpi=150); plt.close()

# ---------- FIG 2: dispersion ----------
P=json.load(open('fig_dispersion.json'))
k=np.array([p['k'] for p in P]); om=np.array([p['om'] for p in P]); nm=[p['dir'] for p in P]
ok=np.isfinite(om)
Afit=np.stack([np.ones(ok.sum()),k[ok]**2],1)
coef,*_=np.linalg.lstsq(Afit,om[ok]**2,rcond=None)
pred=Afit@coef
r2=1-((om[ok]**2-pred)**2).sum()/((om[ok]**2-np.mean(om[ok]**2))**2).sum()

fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
colors={'x':'C0','y':'C1','z':'C2','xy':'C3','xyz':'C4'}
for d in set(nm):
    m=[i for i in range(len(P)) if nm[i]==d and np.isfinite(om[i])]
    a1.scatter(k[m]**2,om[m]**2,s=28,color=colors[d],label=d)
kk=np.linspace(0,k[ok].max()*1.02,50)
a1.plot(kk**2,coef[0]+coef[1]*kk**2,'k-',lw=1.2,
        label=f"fit ω²={coef[0]:.3f}+{coef[1]:.5f}k²  (R²={r2:.3f})")
a1.set_xlabel("k²"); a1.set_ylabel("ω²")
a1.set_title("(a) 3D dispersion: ω² = m² + c²k²  (m²ᵢₙ=1)")
a1.legend(fontsize=8)

# isotropy: omega at n=2 across x,y,z (same |k|)
axes3=['x','y','z']
o2={d:[p['om'] for p in P if p['dir']==d and p['n']==2][0] for d in axes3}
o1={d:[p['om'] for p in P if p['dir']==d and p['n']==1][0] for d in axes3}
xpos=np.arange(3)
a2.bar(xpos-0.2,[o1[d] for d in axes3],0.4,label="n=1 (|k|=6.28)")
a2.bar(xpos+0.2,[o2[d] for d in axes3],0.4,label="n=2 (|k|=12.57)")
sp2=(max(o2.values())-min(o2.values()))/np.mean(list(o2.values()))*100
a2.set_xticks(xpos); a2.set_xticklabels(axes3)
a2.set_ylabel("ω"); a2.set_title(f"(b) Axis isotropy (n=2 spread {sp2:.2f}%)")
a2.legend(fontsize=8); a2.set_ylim(0,1.4)
plt.tight_layout(); plt.savefig('/mnt/user-data/outputs/fig_dispersion.pdf',dpi=150)
plt.savefig('/mnt/user-data/outputs/fig_dispersion.png',dpi=150); plt.close()
print(f"figures done. fit ω²={coef[0]:.3f}+{coef[1]:.5f}k² R²={r2:.3f}, axis spread {sp2:.2f}%")
print(f"plateau A last10 mean={np.mean(A['straight'][-10:]):.4f}  B={np.mean(B['straight'][-10:]):.4f}")
