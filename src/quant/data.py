"""Frente A: engenharia de dados (Deribit, BTC e ETH).

- parse de instrumentos (BTC-27DEC24-60000-C)
- forward implicito por maturidade via Paridade Put-Call inversa: C_moeda - P_moeda = 1 - K/F
- taxa implicita r_imp(tau) = ln(F/S)/tau
- agregacao por instrumento (cadeia diaria) para calibracao SVI
"""
import glob, os
import numpy as np
import pandas as pd
from .black76 import greeks_usd

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
YEAR = 365 * 86400.0


def parse_instruments(names):
    p = pd.Series(names).str.split("-", expand=True)
    exp = pd.to_datetime(p[1], format="%d%b%y") + pd.Timedelta(hours=8)     # vencimento 08:00 UTC
    return pd.DataFrame({"expiry": exp.values, "K": p[2].astype(float).values,
                         "is_call": (p[3] == "C").values})


# ------------------------------- series diarias -------------------------------
def load_daily(coin):
    perp = pd.read_csv(f"{RAW}/{coin}_perp_diario.csv.gz")
    perp["date"] = pd.to_datetime(perp.ticks, unit="ms").dt.normalize()
    perp = perp.groupby("date").last()[["close", "volume"]].rename(columns={"close": "S"})
    dv = pd.read_csv(f"{RAW}/{coin}_dvol.csv.gz")
    dv["date"] = pd.to_datetime(dv.ts, unit="ms").dt.normalize()
    dv = dv.groupby("date").last()[["close"]].rename(columns={"close": "dvol"})
    fu = pd.read_csv(f"{RAW}/{coin}_funding.csv.gz")
    fu["date"] = pd.to_datetime(fu.timestamp, unit="ms").dt.normalize()
    # registros horarios: interest_1h = funding efetivo da hora -> soma diaria
    fu = fu.groupby("date").interest_1h.sum().rename("funding_d").to_frame()
    df = perp.join(dv, how="left").join(fu, how="left")
    df["dvol"] = df.dvol.ffill(); df["funding_d"] = df.funding_d.fillna(0)
    df["ret"] = np.log(df.S).diff()
    return df.dropna(subset=["S"])


def trade_days(coin):
    fs = sorted(glob.glob(f"{RAW}/{coin}_trades_*.csv.gz"))
    return [pd.Timestamp(os.path.basename(f)[-15:-7]) for f in fs]


# ------------------------------ cadeia de opcoes ------------------------------
def implied_forwards(t):
    """Base F/indice por vencimento via paridade put-call inversa em janelas de 1h."""
    t = t.copy()
    t["hour"] = t.timestamp // 3_600_000
    g = t.groupby(["hour", "expiry", "K", "is_call"]).agg(mp=("mark_price", "median"),
                                                          idx=("index_price", "median")).reset_index()
    c = g[g.is_call].drop(columns="is_call"); p = g[~g.is_call].drop(columns="is_call")
    m = c.merge(p, on=["hour", "expiry", "K"], suffixes=("_c", "_p"))
    m["idx"] = (m.idx_c + m.idx_p) / 2
    m = m[np.abs(np.log(m.K / m.idx)) < 0.15]
    m["F"] = m.K / (1 - (m.mp_c - m.mp_p))
    m["basis"] = m.F / m.idx
    m = m[(m.basis > 0.9) & (m.basis < 1.2)]
    return m.groupby("expiry").basis.median()


def load_chain(coin, day, min_tau_days=2, window_hours=4):
    """Uma linha por instrumento negociado no dia: k, iv (Deribit), F, tau, vega, volume.
    window_hours: usa apenas a janela de W horas com mais negocios (fotografia quase sincrona
    da superficie; essencial em dias de crise em que a vol intradiaria se move dezenas de pontos)."""
    t = pd.read_csv(f"{RAW}/{coin}_trades_{day:%Y%m%d}.csv.gz",
                    usecols=["timestamp", "price", "mark_price", "iv", "instrument_name",
                             "index_price", "amount"])
    if window_hours:
        h = (t.timestamp // 3_600_000)
        cnt = h.value_counts().sort_index()
        roll = cnt.reindex(range(cnt.index.min(), cnt.index.max() + 1), fill_value=0) \
                  .rolling(window_hours).sum()
        h_end = roll.idxmax()
        t = t[(h > h_end - window_hours) & (h <= h_end)].reset_index(drop=True)
    t = pd.concat([t, parse_instruments(t.instrument_name)], axis=1)
    ts = pd.to_datetime(t.timestamp, unit="ms")
    t["tau"] = (t.expiry - ts).dt.total_seconds() / YEAR
    t = t[(t.tau > min_tau_days / 365) & (t.iv > 5) & (t.iv < 300)]
    basis = implied_forwards(t)
    # base para vencimentos sem par C/P: interpola r_imp em tau
    tau_e = t.groupby("expiry").tau.median()
    r_known = np.log(basis) / tau_e.reindex(basis.index)
    r_all = pd.Series(np.interp(tau_e.values, tau_e.reindex(r_known.index).values, r_known.values)
                      if len(r_known) else 0.0, index=tau_e.index)
    t["basis"] = t.expiry.map(basis).fillna(np.exp(t.expiry.map(r_all) * t.tau))
    t["F"] = t.index_price * t.basis
    t["k"] = np.log(t.K / t.F)
    agg = t.groupby("instrument_name").agg(
        expiry=("expiry", "first"), K=("K", "first"), is_call=("is_call", "first"),
        tau=("tau", "median"), F=("F", "median"), S=("index_price", "median"), k=("k", "median"),
        iv=("iv", "median"), price=("price", "median"), mark=("mark_price", "median"),
        amount=("amount", "sum"), n=("price", "size")).reset_index()
    agg["iv"] /= 100
    agg["is_call"] = agg.is_call.astype(bool)
    # tau comum por vencimento (necessario para fatias SVI)
    agg["tau"] = agg.groupby("expiry").tau.transform("median")
    agg["vega"] = greeks_usd(agg.F, agg.K, agg.tau, agg.iv, agg.is_call)["vega"] / agg.F
    # usa OTM (mais liquidas): puts para k<0 e calls para k>=0
    agg["otm"] = np.where(agg.k < 0, ~agg.is_call, agg.is_call)
    term = pd.DataFrame({"tau": tau_e, "basis": t.groupby("expiry").basis.first()})
    term["r_imp"] = np.log(term.basis) / term.tau
    term["from_parity"] = term.index.isin(basis.index)
    return agg, t, term


def load_snapshot(coin):
    s = pd.read_csv(f"{RAW}/{coin}_snapshot_option.csv.gz")
    s = pd.concat([s, parse_instruments(s.instrument_name)], axis=1)
    now = pd.to_datetime(s.creation_timestamp, unit="ms")
    s["tau"] = (s.expiry - now).dt.total_seconds() / YEAR
    s["F"] = s.underlying_price
    s["k"] = np.log(s.K / s.F)
    s["iv"] = s.mark_iv / 100
    s["vega"] = greeks_usd(s.F, s.K, s.tau, s.iv, s.is_call)["vega"] / s.F
    s["otm"] = np.where(s.k < 0, ~s.is_call, s.is_call)
    s["spread_rel"] = (s.ask_price - s.bid_price) / s.mark_price
    f = pd.read_csv(f"{RAW}/{coin}_snapshot_future.csv.gz")
    f = f[~f.instrument_name.str.contains("PERPETUAL")].copy()
    f["expiry"] = pd.to_datetime(f.instrument_name.str.split("-").str[1], format="%d%b%y") + pd.Timedelta(hours=8)
    f["tau"] = (f.expiry - pd.to_datetime(f.creation_timestamp, unit="ms")).dt.total_seconds() / YEAR
    f["S"] = f.estimated_delivery_price
    f["r_imp"] = np.log(f.mark_price / f.S) / f.tau
    return s, f.sort_values("tau")
