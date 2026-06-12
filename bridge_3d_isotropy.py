"""
МОСТ-4: 3D аморфная сеть — изотропна ли дисперсия (тот же SG-предел без выделенных осей)?
Сравниваем РЕГУЛЯРНУЮ кубическую решётку (ожидаем анизотропию на больших k)
и АМОРФНУЮ сеть (ожидаем изотропию). Линеаризованный оператор M = c²·L_graph - I,
плоская волна exp(ik·r): ω²(k) = c²·(набег по графу) + 1. Мерим ω²(k) вдоль РАЗНЫХ направлений.
Изотропия = ω²(k) зависит только от |k|, не от направления.
"""
import numpy as np
def build(kind, n, seed=0, box=None):
    rng=np.random.RandomState(seed)
    if kind=='cubic':
        pts=np.array([(i,j,k) for i in range(n) for j in range(n) for k in range(n)],float)
        box=n
    else: # amorphous
        N=n**3; pts=rng.uniform(0,n,(N,3)); box=n
    N=len(pts); deg=6
    # k ближайших соседей (с периодикой по box)
    A=np.zeros((N,N))
    for i in range(N):
        d=pts-pts[i]
        d-=box*np.round(d/box)  # периодика
        dist=np.sqrt((d**2).sum(1)); dist[i]=1e9
        nb=np.argsort(dist)[:deg]
        for j in nb: A[i,j]=1; A[j,i]=1
    D=np.diag(A.sum(1)); Lg=A-D  # граф-лапласиан (норм. на среднюю степень)
    Lg/= (A.sum(1).mean())
    return pts,Lg,box
def disp_along(pts,Lg,box,khat,kmags,c=1.0):
    # ω²(k) проекцией плоской волны exp(ik·r) на оператор: Rayleigh quotient
    out=[]
    for km in kmags:
        kvec=km*np.array(khat)/np.linalg.norm(khat)
        ph=np.exp(1j*(pts@kvec))
        # ω² = -<ph|M|ph>/<ph|ph>, M=c²Lg - I  => ω² = -c²<Lg> + 1
        num=np.real(np.conj(ph)@(Lg@ph)); den=np.real(np.conj(ph)@ph)
        w2=-c**2*(num/den)+1
        out.append(w2)
    return np.array(out)
kmags=np.array([0.2,0.5,1.0,1.5,2.0])
dirs={'[100]':[1,0,0],'[110]':[1,1,0],'[111]':[1,1,1]}
print("=== РЕГУЛЯРНАЯ кубическая решётка (ожидаем АНИЗОТРОПию на больших k) ===")
pts,Lg,box=build('cubic',8)
print(f"{'|k|':>5} "+" ".join(f"{d:>9}" for d in dirs))
for i,km in enumerate(kmags):
    vals=[disp_along(pts,Lg,box,h,[km])[0] for h in dirs.values()]
    spread=max(vals)-min(vals)
    print(f"{km:>5.2f} "+" ".join(f"{v:>9.4f}" for v in vals)+f"   разброс={spread:.4f}")
print("\n=== АМОРФНАЯ 3D сеть (ожидаем ИЗОТРОПию: разброс мал) ===")
pts,Lg,box=build('amorph',10,seed=1)
print(f"{'|k|':>5} "+" ".join(f"{d:>9}" for d in dirs))
for km in kmags:
    vals=[disp_along(pts,Lg,box,h,[km])[0] for h in dirs.values()]
    spread=max(vals)-min(vals)
    print(f"{km:>5.2f} "+" ".join(f"{v:>9.4f}" for v in vals)+f"   разброс={spread:.4f}")
print("\nЕсли у аморфной разброс по направлениям << чем у кубической => аморфность даёт ИЗОТРОПНЫЙ")
print("sine-Gordon-предел. Изотропия пространства = следствие аморфности сети (тезис вакуумной работы).")
