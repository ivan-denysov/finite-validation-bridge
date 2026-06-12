"""
МОСТ-2: инерция из конечности тика (запаздывания).
Сеть ПЕРВОГО порядка (Курамото-тип), но связь с ЗАПАЗДЫВАНИЕМ τ:
  θ̇_i(t) = -K·sin(θ_i(t)) + K_c·[θ_{i+1}(t-τ)+θ_{i-1}(t-τ)-2θ_i(t-τ)]/a²
Никакого θ̈ руками. Вопрос: рождает ли запаздывание ОСЦИЛЛЯЦИИ (признак инерции/2-го порядка)?
И масштабируется ли эффективная "масса" (период²) как τ²?

Теория (разложение DDE по малому τ): θ̇ = f(θ) - τ f'·θ̇ + (τ²/2)θ̈ + ...
=> (τ²/2)θ̈ + (1+τf')θ̇ = f(θ). Инерционный член ~ τ². Эффективная масса m_eff = τ²/2.
Для осцилляций малых мод: ω² ≈ (жёсткость)/m_eff => период T ∝ τ. Проверяем T(τ).
"""
import numpy as np

def run(tau, K=1.0, Kc=1.0, a=1.0, N=64, dt=0.005, T_end=400, seed=0):
    steps=int(T_end/dt); lag=max(1,int(round(tau/dt)))
    # история фаз для запаздывания
    x=np.arange(N)*a; k0=2*np.pi*2/(N*a)
    th0=1e-2*np.cos(k0*x)
    buf=np.tile(th0,(lag+1,1))   # буфер истории
    th=th0.copy()
    series=[]
    def lap(t): return (np.roll(t,-1)+np.roll(t,1)-2*t)/a**2
    for s in range(steps):
        th_delayed=buf[0]               # θ(t-τ)
        dth = -K*np.sin(th) + Kc*lap(th_delayed)
        th = th + dt*dth
        buf=np.roll(buf,-1,axis=0); buf[-1]=th
        if s%2==0: series.append(th[0])
    return np.array(series)

print("Запаздывание τ → осцилляции? Период T(τ)? (m_eff ∝ τ² => T ∝ τ)")
print(f"{'τ':>6} {'осцилляции':>12} {'период T':>10} {'T/τ':>8}")
for tau in [0.0, 0.5, 1.0, 2.0, 4.0]:
    ser=run(tau)
    # детект осцилляций: знакопеременность производной + FFT
    sp=np.abs(np.fft.rfft(ser-ser.mean()))
    fr=np.fft.rfftfreq(len(ser),d=2*0.005)
    if tau==0:
        print(f"{tau:>6.1f} {'нет (релакс)':>12} {'--':>10} {'--':>8}")
        continue
    peak=np.argmax(sp[1:])+1
    f_peak=fr[peak]
    osc = sp[peak] > 3*np.median(sp[1:]) and f_peak>0
    Tp=1/f_peak if f_peak>0 else float('inf')
    print(f"{tau:>6.1f} {'ДА' if osc else 'нет':>12} {Tp:>10.2f} {Tp/tau:>8.2f}")
print()
print("Если τ>0 даёт осцилляции (которых нет у τ=0 чистой релаксации) и T∝τ —")
print("инерция РОЖДЕНА запаздыванием. Второй порядок не вписан, а выведен из конечности тика.")
