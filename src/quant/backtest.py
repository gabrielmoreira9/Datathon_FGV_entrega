"""Frente E: simulador das 4 carteiras (NAV em USD, 1 unidade de BTC/ETH de nocional).

Marcacao diaria das opcoes:
  - na data de rolagem (1o dia util do mes) usa-se a superficie SVI calibrada nos negocios do dia;
  - entre rolagens: sigma_t(k, tau) = sigma_SVI_rolagem(k, tau) * DVOL_t / DVOL_rolagem
    (sticky-moneyness com nivel ancorado no indice DVOL diario da Deribit);
  - forward F_t = S_t * exp(r_imp * tau), r_imp extraido da paridade put-call na rolagem.
Custos: taxa de opcao 0.03% do subjacente (teto 12.5% do premio) + meio spread bid-ask empirico,
taxa de entrega 0.015% no vencimento, taxa taker de 0.05% no perpetuo e funding horario real.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from . import black76 as b
from .svi import SVISurface

FEE_OPT, FEE_DELIV, FEE_PERP = 0.0003, 0.00015, 0.0005


class Market:
    def __init__(self, daily, rolls):
        """daily: DataFrame (S, dvol, funding_d) indexado por data; rolls: dict data -> registro pkl."""
        self.d = daily
        self.rolls = rolls
        self.roll_dates = sorted(rolls)

    def ctx(self, date):
        """Superficie, DVOL de ancoragem e r_imp vigentes em `date`."""
        rd = max(x for x in self.roll_dates if x <= date)
        rec = self.rolls[rd]
        term = rec["term"]
        r = float(term[(term.tau > 10 / 365) & (term.tau < 0.3)].r_imp.median())
        return rec, SVISurface(rec["slices"]), float(self.d.dvol.get(rd, np.nan)), r

    def vol(self, surf, dvol_anchor, date, F, K, tau):
        scale = self.d.dvol.loc[date] / dvol_anchor if np.isfinite(dvol_anchor) else 1.0
        return surf.iv(np.log(np.asarray(K) / F), max(tau, 1 / 365)) * scale


def _value(legs, F, tau, sig_fn):
    """Valor USD e Gregas USD agregadas de uma lista de pernas {K, is_call, q}."""
    tot = dict(price=0.0, delta=0.0, gamma=0.0, vega=0.0, theta=0.0, vanna=0.0, volga=0.0)
    if tau <= 0:
        for L in legs:
            tot["price"] += L["q"] * (max(F - L["K"], 0) if L["is_call"] else max(L["K"] - F, 0))
        return tot
    for L in legs:
        g = b.greeks_usd(F, L["K"], tau, sig_fn(L["K"]), L["is_call"])
        for k in tot:
            tot[k] += L["q"] * float(g[k])
    return tot


def run_strategy(mkt, kind, start, end, half_spread_rel, rf=0.04, band=0.10, cost_mult=1.0):
    """kind in {'spot','put','collar','dgh'}. Retorna DataFrame diario com NAV e decomposicao."""
    dates = mkt.d.loc[start:end].index
    rolls = [x for x in mkt.roll_dates if start <= x <= end]
    S0 = mkt.d.S.loc[dates[0]]
    cash = S0 if kind == "dgh" else 0.0
    spot_q = 0.0 if kind == "dgh" else 1.0
    perp_q, legs, expiry, units = 0.0, [], None, 1.0
    rows, prev = [], None
    costs_cum = 0.0

    def opt_cost(K, is_call, F, tau, sig, q):
        px = float(b.price_usd(F, K, tau, sig, is_call))
        fee = min(FEE_OPT * F, 0.125 * px)
        return cost_mult * abs(q) * (fee + half_spread_rel * px)

    for i, t in enumerate(dates):
        S = mkt.d.S.loc[t]
        if prev is not None:
            dt = (t - prev).days / 365
            if kind == "dgh":
                cash *= np.exp(rf * dt)
            # perpetuo: P&L de preco e funding (longs pagam funding positivo)
            if perp_q:
                cash += perp_q * (S - mkt.d.S.loc[prev]) - perp_q * mkt.d.S.loc[prev] * mkt.d.funding_d.loc[t]
        rec, surf, dv0, r = mkt.ctx(t)

        # --- vencimento/rolagem ---
        if kind != "spot" and (t in rolls or not legs) and t < dates[-1]:
            if legs:                       # liquida pelo intrinseco (payoff inverso convertido a USD)
                for L in legs:
                    pay = max(S - L["K"], 0) if L["is_call"] else max(L["K"] - S, 0)
                    cash += L["q"] * pay
                    c = cost_mult * min(FEE_DELIV * S, 0.125 * pay) * abs(L["q"]) if pay > 0 else 0.0
                    cash -= c; costs_cum += c
            # carteira autofinanciada: redimensiona o nocional para NAV/S a cada rolagem
            nav_now = spot_q * S + cash + perp_q * 0.0
            units = nav_now / S
            if kind != "dgh":
                trade_spot = units - spot_q
                c = cost_mult * abs(trade_spot) * S * FEE_PERP if prev is not None else 0.0
                cash -= trade_spot * S + c; costs_cum += c
                spot_q = units
            nxt = [x for x in rolls if x > t]
            expiry = nxt[0] if nxt else t + pd.Timedelta(days=30)
            tau = (expiry - t).days / 365
            F = S * np.exp(r * tau)
            sig_fn = lambda K, F=F, tau=tau: float(mkt.vol(surf, dv0, t, F, K, tau))
            if kind == "put":
                Kp = b.strike_from_delta(F, tau, sig_fn, -0.20, False)
                legs = [dict(K=Kp, is_call=False, q=units)]
            elif kind == "collar":
                Kp = b.strike_from_delta(F, tau, sig_fn, -0.25, False)
                P = float(b.price_usd(F, Kp, tau, sig_fn(Kp), False))
                Kc = brentq(lambda K: float(b.price_usd(F, K, tau, sig_fn(K), True)) - P, F * 1.0001, F * 4)
                legs = [dict(K=Kp, is_call=False, q=units), dict(K=Kc, is_call=True, q=-units)]
            elif kind == "dgh":            # short straddle ATM + asas 10-delta (iron butterfly)
                Kw_p = b.strike_from_delta(F, tau, sig_fn, -0.10, False)
                Kw_c = b.strike_from_delta(F, tau, sig_fn, 0.10, True)
                legs = [dict(K=F, is_call=True, q=-units), dict(K=F, is_call=False, q=-units),
                        dict(K=Kw_p, is_call=False, q=units), dict(K=Kw_c, is_call=True, q=units)]
            for L in legs:
                px = float(b.price_usd(F, L["K"], tau, sig_fn(L["K"]), L["is_call"]))
                cash -= L["q"] * px
                c = opt_cost(L["K"], L["is_call"], F, tau, sig_fn(L["K"]), L["q"])
                cash -= c; costs_cum += c

        tau = max((expiry - t).days / 365, 0) if expiry is not None else 0
        F = S * np.exp(r * tau)
        sig_fn = lambda K, F=F, tau=tau: float(mkt.vol(surf, dv0, t, F, K, tau))
        g = _value(legs, F, tau, sig_fn) if legs else dict(price=0, delta=0, gamma=0, vega=0, theta=0, vanna=0, volga=0)

        # --- hedge dinamico de delta via perpetuo (carteira 4) ---
        if kind == "dgh":
            port_delta = g["delta"] + perp_q
            if abs(port_delta) > band * units:
                trade = -port_delta
                c = cost_mult * abs(trade) * S * FEE_PERP
                cash -= c; costs_cum += c
                perp_q += trade

        nav = spot_q * S + g["price"] + cash
        rows.append(dict(date=t, S=S, nav=nav, opt_value=g["price"], cash=cash, perp_q=perp_q,
                         delta=spot_q + g["delta"] + perp_q, gamma=g["gamma"], vega=g["vega"],
                         theta=g["theta"], vanna=g["vanna"], volga=g["volga"], costs=costs_cum,
                         strikes=tuple(round(L["K"]) for L in legs), dvol=mkt.d.dvol.loc[t]))
        prev = t
    out = pd.DataFrame(rows).set_index("date")
    out["ret"] = np.log(out.nav).diff()
    return out


def metrics(ret, bench, rf=0.04, ppy=365):
    """Bateria de indicadores de risco-retorno sobre log-retornos diarios."""
    r = ret.dropna(); bm = bench.reindex(r.index).fillna(0)
    rf_d = rf / ppy
    nav = np.exp(r.cumsum())
    yrs = len(r) / ppy
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(ppy)
    down = r[r < 0].std() * np.sqrt(ppy)
    mdd = (nav / nav.cummax() - 1).min()
    ex, exm = r - rf_d, bm - rf_d
    beta = np.cov(ex, exm)[0, 1] / np.var(exm, ddof=1)
    alpha = (ex.mean() - beta * exm.mean()) * ppy
    var99 = -np.percentile(r, 1)
    cvar99 = -r[r <= -var99].mean()
    return {"Retorno a.a.": cagr, "Volatilidade": vol, "Sharpe": (r.mean() * ppy - rf) / vol,
            "Sortino": (r.mean() * ppy - rf) / down, "Calmar": cagr / abs(mdd), "Alpha (Jensen)": alpha,
            "Beta": beta, "MDD": mdd, "VaR 99% (1d)": var99, "CVaR 99% (1d)": cvar99}
