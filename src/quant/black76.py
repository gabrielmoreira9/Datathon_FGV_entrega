"""Modelo 1, 2 e 4: Black-76 (USD e inverso em moeda), inversao hibrida de IV e vetor de Gregas.

Convencao Deribit (verificada empiricamente no snapshot): opcoes precificadas por Black-76
sobre o forward F da maturidade, r = 0, e preco em moeda = V_USD / F.
Todas as funcoes sao vetorizadas (numpy broadcasting).
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

N, n = norm.cdf, norm.pdf
SIG_MIN, SIG_MAX = 0.05, 5.0


def _d1d2(F, K, tau, sig):
    F, K, tau, sig = map(np.asarray, (F, K, tau, sig))
    st = sig * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * st**2) / st
    return d1, d1 - st


def price_usd(F, K, tau, sig, is_call, r=0.0):
    """Preco Black-76 em USD. is_call: bool ou array de bool."""
    d1, d2 = _d1d2(F, K, tau, sig)
    df = np.exp(-r * np.asarray(tau))
    c = df * (F * N(d1) - K * N(d2))
    p = df * (K * N(-d2) - F * N(-d1))
    return np.where(is_call, c, p)


def price_coin(F, K, tau, sig, is_call, r=0.0):
    """Preco da opcao inversa em unidades da moeda (BTC/ETH): V_USD / F."""
    return price_usd(F, K, tau, sig, is_call, r) / np.asarray(F)


def greeks_usd(F, K, tau, sig, is_call, r=0.0):
    """Vetor analitico de Gregas (Black-76, USD). Theta por ano; vega por 1.00 de vol."""
    F, K, tau, sig = map(np.asarray, (F, K, tau, sig))
    d1, d2 = _d1d2(F, K, tau, sig)
    df, sq = np.exp(-r * tau), np.sqrt(tau)
    V = price_usd(F, K, tau, sig, is_call, r)
    delta = np.where(is_call, df * N(d1), -df * N(-d1))
    gamma = df * n(d1) / (F * sig * sq)
    vega = F * df * sq * n(d1)
    theta = -F * df * n(d1) * sig / (2 * sq) + r * V           # dV/dt (t calendario), por ano
    vanna = -df * n(d1) * d2 / sig
    volga = vega * d1 * d2 / sig
    rho = -tau * V
    return dict(price=V, delta=delta, gamma=gamma, vega=vega, theta=theta,
                vanna=vanna, volga=volga, rho=rho)


def delta_coin(F, K, tau, sig, is_call):
    """Delta em moeda da opcao inversa (Deribit 'delta' ajustado ao premio):
    d(V_USD/F)/dF * F  ->  Delta_USD - V_USD/F."""
    g = greeks_usd(F, K, tau, sig, is_call)
    return g["delta"] - g["price"] / np.asarray(F)


# ---------------------------- Modelo 2: IV hibrida ----------------------------
def implied_vol(V_usd, F, K, tau, is_call, r=0.0, tol=1e-8, max_newton=50):
    """Newton-Raphson (rapido, onde o vega e alto) com fallback global de Brent em [0.05, 5]."""
    V_usd, F, K, tau = (np.atleast_1d(np.asarray(x, float)) for x in (V_usd, F, K, tau))
    is_call = np.broadcast_to(np.atleast_1d(is_call), V_usd.shape)
    F, K, tau = (np.broadcast_to(x, V_usd.shape) for x in (F, K, tau))
    out = np.full(V_usd.shape, np.nan)
    method = np.zeros(V_usd.shape, int)                 # 1 = Newton, 2 = Brent
    for i in range(V_usd.size):
        f, k, t, c, v = F[i], K[i], tau[i], bool(is_call[i]), V_usd[i]
        intrinsic = max(f - k, 0) if c else max(k - f, 0)
        if not np.isfinite(v) or t <= 0 or v <= intrinsic * np.exp(-r * t) + 1e-12:
            continue
        s = np.sqrt(2 * abs(np.log(f / k)) / t) if f != k else 0.6   # chute (Brenner-Subrahmanyam/ Manaster-Koehler)
        s = float(np.clip(s if s > 0.05 else 0.6, 0.1, 3.0))
        ok = False
        for _ in range(max_newton):
            g = greeks_usd(f, k, t, s, c, r)
            diff = float(g["price"]) - v
            if abs(diff) < tol * max(f, 1):
                ok = True; break
            vg = float(g["vega"])
            if vg < 1e-8 * f: break
            s_new = s - diff / vg
            if not (SIG_MIN < s_new < SIG_MAX): break
            s = s_new
        if ok:
            out[i], method[i] = s, 1
            continue
        fn = lambda x: float(price_usd(f, k, t, x, c, r)) - v
        try:
            if fn(SIG_MIN) * fn(SIG_MAX) < 0:
                out[i], method[i] = brentq(fn, SIG_MIN, SIG_MAX, xtol=1e-10), 2
        except ValueError:
            pass
    return out, method


def strike_from_delta(F, tau, sig_fn, target_delta, is_call, lo=0.2, hi=5.0):
    """Encontra K tal que Delta_USD(K, sigma(K)) = alvo, usando a vol do sorriso sig_fn(K)."""
    f = lambda K: float(greeks_usd(F, K, tau, sig_fn(K), is_call)["delta"]) - target_delta
    return brentq(f, lo * F, hi * F, xtol=1e-6 * F)
