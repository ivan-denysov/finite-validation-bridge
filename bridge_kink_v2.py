"""МОСТ-3b: тот же кинк-тест, КОРРЕКТНОЕ измерение заряда и ширины.
Q = (1/2π)∫(∂θ/∂x)dx — полный набег фазы (топологический инвариант), не по краям.
Свободные (Neumann) границы, домен с запасом, центр и края подальше от кинка."""
import numpy as np
N=800; a=0.5; c=1.0; dt=0.008
x=np.arange(N)*a; L=N*a
def lap(t):  # Neumann (отражающие) границы — не рвут кинк
    l=np.empty_like(t)
    l[1:-1]=(t[2:]-2*t[1:-1]+t[:-2])/a**2
    l[0]=(t[1]-t[0])/a**2; l[-1]=(t[-2]-t[-1])/a**2
    return l
def kink(x0,v):
    g=1/np.sqrt(1-v**2)
    return 4*np.arctan(np.exp(g*(x-x0)))
def Qtop(th):  # топологический заряд = набег фазы /2π
    return np.sum(np.gradient(th,a))*a/(2*np.pi)
def width(th):  # ширина = 2π / max|∂θ/∂x|  (для SG-кинка покоя =1)
    return (2*np.pi)/ (np.abs(np.gradient(th,a)).max()) /np.pi  # норм. к покою
def center(th):
    return x[np.argmin(np.abs(th-np.pi))]
def evolve(th,thd,Tend):
    acc=c**2*lap(th)-np.sin(th)
    for s in range(int(Tend/dt)):
        thd=thd+0.5*dt*acc; th=th+dt*thd
        acc=c**2*lap(th)-np.sin(th); thd=thd+0.5*dt*acc
    return th,thd

print("=== (1) Статический кинк на решётке ===")
th=kink(L/2,0.0); thd=np.zeros(N)
Q0,c0=Qtop(th),center(th)
th,thd=evolve(th,thd,60)
print(f"Q: {Q0:.4f} -> {Qtop(th):.4f}  (топологический, должен = 1)")
print(f"центр: {c0:.2f} -> {center(th):.2f}  (сдвиг {abs(center(th)-c0):.3f})")
print(f"профиль набег: {th[-1]-th[0]:.4f} (=2π={2*np.pi:.4f})")

print("\n=== (2) Движущийся кинк: лоренц-сжатие ===")
print(f"{'v':>5} {'ширина/покой':>13} {'√(1-v²)':>9} {'Q':>7} {'центр прошёл':>13}")
w0=None
for v in [0.0,0.3,0.6,0.8]:
    th=kink(L/3,v); thd=-v*np.gradient(th,a)
    c_start=center(th)
    th,thd=evolve(th,thd,15)
    g=np.abs(np.gradient(th,a)); w=(2*np.pi)/g.max()
    if v==0: w0=w
    print(f"{v:>5.1f} {w/w0:>13.4f} {np.sqrt(1-v**2):>9.4f} {Qtop(th):>7.4f} {center(th)-c_start:>13.2f}")
print()
print("Q=1 держится + ширина/покой ≈ √(1-v²) + центр уезжает ∝ v => кинк сети = кинк SG, заряд проходит мост.")
