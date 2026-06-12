"""МОСТ-1b: по одной моде за прогон, длинный ряд — чистая дисперсия.
Линеаризуем (малая амплитуда, sin θ ≈ θ): тогда ω² = c²·(2/a²)(1-cos ka) + 1 ТОЧНО.
Проверяем и линейный режим (точное совпадение ожидается), и слабо-нелинейный."""
import numpy as np
N=1024; a=1.0; c=1.0; dt=0.01; steps=20000
def lap(th): return (np.roll(th,-1)+np.roll(th,1)-2*th)/a**2
def measure(k_idx, amp, nonlin):
    x=np.arange(N)*a; k=2*np.pi*k_idx/(N*a)
    th=amp*np.cos(k*x); thd=np.zeros(N)
    f=(np.sin if nonlin else (lambda z:z))
    acc=c**2*lap(th)-f(th)
    series=[]
    for t in range(steps):
        thd=thd+0.5*dt*acc; th=th+dt*thd
        acc=c**2*lap(th)-f(th); thd=thd+0.5*dt*acc
        if t%2==0: series.append(th[0])
    series=np.array(series)
    sp=np.abs(np.fft.rfft(series-series.mean()))
    fr=2*np.pi*np.fft.rfftfreq(len(series),d=2*dt)
    return fr[np.argmax(sp[1:])+1], k
print("LINEAR regime (sin θ→θ): ω должно ТОЧНО = √(c²·2/a²(1-cos ka)+1)")
print(f"{'k_idx':>6} {'k':>8} {'ω_meas':>9} {'ω_lattice':>10} {'ω_cont':>9} {'Δ%latt':>7}")
for ki in [1,2,4,8,16,32,64,128]:
    w,k=measure(ki,1e-3,False)
    wl=np.sqrt(c**2*(2/a**2)*(1-np.cos(k*a))+1)
    wc=np.sqrt(c**2*k**2+1)
    print(f"{ki:>6} {k:>8.4f} {w:>9.4f} {wl:>10.4f} {wc:>9.4f} {100*abs(w-wl)/wl:>7.2f}")
