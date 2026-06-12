"""МОСТ-5b: где ошибка вывода? Берём ЧИСТЫЙ скаляр (одна мода, без пространства):
DDE: θ̇(t) = -S·θ(t-τ),  S>0 (жёсткость моды). Это точная линейная DDE.
Точное характеристическое ур-е: λ = -S·e^{-λτ}. Осцилляции, если есть комплексные λ.
Сравним: (а) точный корень DDE; (б) моё разложение 2-го порядка; (в) численность.
Цель: найти ПРАВИЛЬНУЮ зависимость периода от S и τ."""
import numpy as np
from scipy.optimize import brentq
import cmath

def exact_omega(S,tau):
    # λ = -S e^{-λτ}, ищем λ=α+iβ. Осцилляция: β>0. Решаем итерацией.
    # подстановка: λτ = -Sτ e^{-λτ}. Пусть z=λτ, p=Sτ: z=-p e^{-z}.
    p=S*tau
    z=complex(-0.1,1.0)
    for _ in range(200):
        z=z-(z+p*cmath.exp(-z))/(1+(-p*cmath.exp(-z))*(-1))  # Newton: f=z+p e^{-z}, f'=1-p e^{-z}
        f=z+p*cmath.exp(-z); fp=1-p*cmath.exp(-z)
        z=z-f/fp
    lam=z/tau
    return lam   # период T=2π/Im(lam)
print("Точная DDE θ̇=-Sθ(t-τ): период и устойчивость vs p=Sτ")
print(f"{'S':>5} {'τ':>5} {'p=Sτ':>6} {'Im(λ)':>9} {'T=2π/Im':>9} {'T/τ':>8} {'Re(λ)':>9} {'режим':>10}")
for S in [0.5,1.0,2.0]:
    for tau in [1.0,2.0,3.0]:
        lam=exact_omega(S,tau)
        b=lam.imag; T=2*np.pi/abs(b) if abs(b)>1e-6 else np.inf
        mode='осцил' if abs(b)>1e-6 else 'релакс'
        grow='раст' if lam.real>1e-4 else ('зат' if lam.real<-1e-4 else 'нейтр')
        print(f"{S:>5.1f} {tau:>5.1f} {S*tau:>6.2f} {b:>9.4f} {T:>9.3f} {T/tau:>8.3f} {lam.real:>9.4f} {mode+'/'+grow:>10}")
print()
print("T/τ зависит ТОЛЬКО от p=Sτ (безразмерный) — вот правильная переменная.")
print("При p=π/2 граница устойчивости (Re λ=0): классический порог DDE. Период там T=4τ.")
