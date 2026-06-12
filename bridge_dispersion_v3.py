"""МОСТ-1c: дисперсия АНАЛИТИЧЕСКИ из собственных значений линеаризованной системы.
Линеаризация цепи θ̈_i = c²(θ_{i+1}+θ_{i-1}-2θ_i)/a² - θ_i вокруг 0:
матрица M = c²·L/a² - I, где L — циклический лапласиан. Собственные моды = плоские волны,
ω²(k) = -eigenvalue = c²·(2/a²)(1-cos ka) + 1. Считаем eig напрямую — без FFT, без шума."""
import numpy as np
N=64; a=1.0; c=1.0
L=np.zeros((N,N))
for i in range(N):
    L[i,i]=-2; L[i,(i+1)%N]=1; L[i,(i-1)%N]=1
M=c**2*L/a**2 - np.eye(N)        # линеаризованный оператор
ev=np.linalg.eigvalsh(M)
w2=-ev                            # ω² = -собственное значение
w2.sort()
ks=2*np.pi*np.arange(N//2+1)/(N*a)
print("Точная дисперсия из eig vs решёточная формула vs континуум sine-Gordon:")
print(f"{'k':>8} {'ω²_eig':>10} {'ω²_lattice':>11} {'ω²_cont(c²k²+1)':>16} {'match':>7}")
w2_modes=sorted(set(np.round(w2,6)))
for j,k in enumerate(ks):
    wl=c**2*(2/a**2)*(1-np.cos(k*a))+1
    wc=c**2*k**2+1
    we=w2_modes[j] if j<len(w2_modes) else float('nan')
    ok='OK' if abs(we-wl)<1e-6 else f'Δ{abs(we-wl):.1e}'
    print(f"{k:>8.4f} {we:>10.5f} {wl:>11.5f} {wc:>16.5f} {ok:>7}")
print()
print("ВЫВОД: ω²_eig === ω²_lattice ТОЧНО (машинная точность) для всех k —")
print("линеаризованная фазовая цепь с инерцией ЕСТЬ дискретный sine-Gordon аналитически.")
print("При ka→0: ω²→c²k²+1 (континуальный sine-Gordon). Мост = классический предел Френкеля-Конторовы.")
