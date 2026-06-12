"""
МОСТ-1: попадает ли дискретная фазовая сеть с инерцией в класс sine-Gordon?
Тест через ДИСПЕРСИЮ. 1D-цепь фаз с инерцией:
  m·θ̈_i = -Γ·θ̇_i + K(θ_{i+1}+θ_{i-1}-2θ_i) - sin(θ_i) + ...
(второй порядок = инерция; связь = дискретный лапласиан фаз; on-site sin = washboard).
Континуальный предел такой цепи = sine-Gordon: φ_tt - c²φ_xx + sin φ = 0,
с дисперсией малых колебаний вокруг φ=0:  ω² = c²k² + 1  (массивная ветвь, "щель"=1).
Дискретная решётка даёт:  ω² = c²·(2/a²)(1-cos(ka)) + 1  ->  при ka→0 совпадает с континуумом,
при больших k отклоняется (решёточная дисперсия). Мерим ω(k) ЧИСЛЕННО из эволюции и фитим.
"""
import numpy as np

N=512; a=1.0; c=1.0; m=1.0; Gamma=0.0; K=c**2/a**2  # бездиссипативно для чистой дисперсии
dt=0.02; steps=4000

def laplacian(th):
    return (np.roll(th,-1)+np.roll(th,1)-2*th)/a**2

# малое возмущение: суперпозиция мод, смотрим как каждая осциллирует
x=np.arange(N)*a
ks=2*np.pi*np.array([1,2,4,8,16,32,64])/(N*a)   # волновые числа
amp=1e-3
th=np.zeros(N); 
for k in ks: th+=amp*np.cos(k*x)
thd=np.zeros(N)

# записываем временные ряды амплитуд фурье-мод
hist=[]
th_prev=th.copy()
# leapfrog
acc=K*laplacian(th)-np.sin(th)
for t in range(steps):
    thd=thd+0.5*dt*acc/m
    th=th+dt*thd
    acc=K*laplacian(th)-np.sin(th)-Gamma*thd
    thd=thd+0.5*dt*acc/m
    if t%4==0:
        F=np.fft.rfft(th)
        hist.append(F)
hist=np.array(hist)
tt=np.arange(len(hist))*4*dt

print("k (1/a)   ω_measured   ω_SG_cont=√(c²k²+1)   ω_lattice=√(c²·2/a²(1-cos ka)+1)")
print("-"*78)
for k in ks:
    idx=int(round(k*N*a/(2*np.pi)))
    series=np.real(hist[:,idx])
    # частота из FFT временного ряда
    sp=np.abs(np.fft.rfft(series-series.mean()))
    freqs=2*np.pi*np.fft.rfftfreq(len(series),d=4*dt)
    w_meas=freqs[np.argmax(sp[1:])+1]
    w_cont=np.sqrt(c**2*k**2+1)
    w_latt=np.sqrt(c**2*(2/a**2)*(1-np.cos(k*a))+1)
    print(f"{k:.4f}    {w_meas:.4f}       {w_cont:.4f}              {w_latt:.4f}")
print()
print("Если ω_measured ≈ ω_lattice везде, а ω_lattice→ω_cont при малых k —")
print("дискретная фазовая сеть с инерцией ЕСТЬ дискретный sine-Gordon. Мост = известный континуальный предел.")
