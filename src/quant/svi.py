"""Modelo 3: Raw SVI (Gatheral, 2004) com penalizacao de arbitragem estatica.

w(k) = a + b * ( rho*(k-m) + sqrt((k-m)^2 + s^2) ),  k = ln(K/F),  w = sigma^2 * tau
Restricoes:
  b >= 0, |rho| < 1, s > 0, a + b*s*sqrt(1-rho^2) >= 0           (variancia minima >= 0)
  b*(1+|rho|) <= 2                                                (Roger Lee: inclinacao das asas de w <= 2)
  g(k) >= 0  (densidade de Gatheral-Jacquier >= 0 -> sem arbitragem de butterfly)
  w(k, tau_i) >= w(k, tau_{i-1})  (sem arbitragem de calendario; calibracao sequencial)
"""
import numpy as np
from scipy.optimize import least_squares

KGRID = np.linspace(-1.5, 1.5, 301)


def svi_w(k, p):
    a, b, rho, m, s = p
    x = np.asarray(k) - m
    return a + b * (rho * x + np.sqrt(x * x + s * s))


def svi_derivs(k, p):
    a, b, rho, m, s = p
    x = np.asarray(k) - m
    r = np.sqrt(x * x + s * s)
    return svi_w(k, p), b * (rho + x / r), b * s * s / r**3


def g_density(k, p):
    """Funcao g(k) de Gatheral & Jacquier (2014); g < 0 => arbitragem de butterfly."""
    w, w1, w2 = svi_derivs(k, p)
    w = np.maximum(w, 1e-12)
    return (1 - k * w1 / (2 * w))**2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2


def calibrate_slice(k, iv, tau, weights=None, w_floor=None, x0=None, lam=50.0):
    """Minimos quadrados nao-lineares (scipy least_squares) ponderados por vega.
    k, iv: arrays; tau: anos; w_floor: (KGRID, w_prev) da maturidade anterior p/ calendario."""
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    wt = np.ones_like(k) if weights is None else np.asarray(weights, float)
    wt = np.sqrt(wt / wt.mean())
    w_mkt = iv**2 * tau
    atm = np.interp(0.0, np.sort(k), w_mkt[np.argsort(k)])
    kg = KGRID[(KGRID > k.min() - 0.5) & (KGRID < k.max() + 0.5)]

    def resid(p):
        a, b, rho, m, s = p
        fit = svi_w(k, p)
        r_fit = wt * (np.sqrt(np.maximum(fit, 1e-10) / tau) - iv)          # erro em pontos de vol
        pen = [lam * max(0.0, b * (1 + abs(rho)) - 2.0),
               lam * max(0.0, -(a + b * s * np.sqrt(1 - rho**2)))]
        g = g_density(kg, p)
        pen.append(10 * lam * np.sqrt(np.mean(np.minimum(g, 0.0)**2)))
        if w_floor is not None:
            kgf, wprev = w_floor
            pen.append(lam * np.sqrt(np.mean(np.minimum(svi_w(kgf, p) - wprev, 0.0)**2)) / max(tau, 1e-3))
        return np.concatenate([r_fit, pen])

    lb = [-1.0, 0.0, -0.999, -1.0, 1e-3]
    ub = [max(2 * w_mkt.max(), 0.05), 2.0, 0.999, 1.0, 2.0]
    starts = [x0] if x0 is not None else []
    starts += [[atm * 0.5, 0.1, -0.2, 0.0, 0.1], [atm * 0.8, 0.05, 0.0, 0.05, 0.2],
               [atm * 0.3, 0.3, -0.5, -0.1, 0.3]]
    best = None
    for s0 in starts:
        s0 = np.clip(s0, np.array(lb) + 1e-6, np.array(ub) - 1e-6)
        try:
            r = least_squares(resid, s0, bounds=(lb, ub), method="trf", max_nfev=600)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    fit_iv = np.sqrt(np.maximum(svi_w(k, p), 1e-10) / tau)
    ss_res = np.sum((iv - fit_iv)**2); ss_tot = np.sum((iv - iv.mean())**2)
    return dict(params=p, tau=tau, rmse_vol=float(np.sqrt(np.mean((iv - fit_iv)**2))),
                r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                min_g=float(g_density(KGRID, p).min()), n=len(k))


class SVISurface:
    """Superficie: fatias SVI por maturidade; interpolacao linear em variancia total ao longo de tau
    (preserva ausencia de arbitragem de calendario entre fatias livres de arbitragem)."""

    def __init__(self, slices):
        self.slices = sorted(slices, key=lambda s: s["tau"])
        self.taus = np.array([s["tau"] for s in self.slices])

    def w(self, k, tau):
        k = np.asarray(k, float)
        T = self.taus
        if tau <= T[0]:
            return svi_w(k, self.slices[0]["params"]) * tau / T[0]
        if tau >= T[-1]:
            return svi_w(k, self.slices[-1]["params"]) * tau / T[-1]
        j = np.searchsorted(T, tau)
        t0, t1 = T[j - 1], T[j]
        a = (tau - t0) / (t1 - t0)
        return (1 - a) * svi_w(k, self.slices[j - 1]["params"]) + a * svi_w(k, self.slices[j]["params"])

    def iv(self, k, tau):
        return np.sqrt(np.maximum(self.w(k, tau), 1e-10) / tau)


def calibrate_surface(df, min_points=6):
    """df: colunas tau, k, iv, vega (uma linha por instrumento). Calibracao sequencial curta->longa."""
    slices, prev = [], None
    for tau, g in sorted(df.groupby("tau"), key=lambda x: x[0]):
        g = g[(g.k.abs() < 1.2)]
        if len(g) < min_points or tau < 2 / 365:
            continue
        floor = (KGRID, svi_w(KGRID, prev["params"])) if prev is not None else None
        s = calibrate_slice(g.k.values, g.iv.values, tau, g.vega.values, w_floor=floor,
                            x0=prev["params"] if prev is not None else None)
        if s["rmse_vol"] > 0.03 or not (s["r2"] > 0.5):
            # piso de calendario incompativel com os dados: recalibra livre e mantem o melhor
            s2 = calibrate_slice(g.k.values, g.iv.values, tau, g.vega.values)
            if s2["rmse_vol"] < s["rmse_vol"]:
                s = s2
        s["expiry"] = g["expiry"].iloc[0] if "expiry" in g else None
        if not (s["r2"] > 0.5) and s["rmse_vol"] > 0.03:
            continue                                  # fatia descartada (dados insuficientes/ruidosos)
        slices.append(s)
        if s["rmse_vol"] < 0.03:                      # so fatias bem ajustadas viram piso de calendario
            prev = s
    return SVISurface(slices) if slices else None
