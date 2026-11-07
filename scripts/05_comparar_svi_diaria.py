"""Compara a marcacao mensal por DVOL com superficies SVI diarias.

Mantem as rolagens mensais e os custos atuais. Tambem repete o teste de skew
do README em datas mensais e em todas as datas, usando os mesmos resultados
futuros de 30 dias. Uso: python scripts/05_comparar_svi_diaria.py
"""
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quant import backtest as bt, data


ROOT = os.path.join(os.path.dirname(__file__), "..")
SURFACE_DIR = os.path.join(ROOT, "data", "processed", "surfaces")
TABLE_DIR = os.path.join(ROOT, "output", "tables")
os.makedirs(TABLE_DIR, exist_ok=True)
COINS = ["BTC", "ETH"]
KINDS = ["spot", "put", "collar", "dgh"]
STRATEGY_NAMES = {
    "spot": "C1 Spot",
    "put": "C2 Protective Put",
    "collar": "C3 Zero-Cost Collar",
    "dgh": "C4 Short Iron Butterfly + Hedge de Delta",
}


def load_surfaces(coin):
    records = {}
    for filename in sorted(glob.glob(os.path.join(SURFACE_DIR, f"{coin}_*.pkl"))):
        with open(filename, "rb") as fh:
            rec = pickle.load(fh)
        records[pd.Timestamp(rec["date"])] = rec
    return records


def half_spread(coin):
    options, _ = data.load_snapshot(coin)
    liquid = options[(options.tau > 15 / 365) & (options.tau < 45 / 365)
                     & (options.k.abs() < 0.3) & (options.bid_price > 0)]
    return float(((liquid.ask_price - liquid.bid_price) / 2 / liquid.mark_price).median())


def forward_skew_rows(coin, daily, surfaces):
    rows = []
    spot = daily.S
    for day, rec in sorted(surfaces.items()):
        end = day + pd.Timedelta(days=30)
        path = spot.loc[day:end]
        if len(path) < 25:
            continue
        rows.append({
            "coin": coin,
            "date": day,
            "rr25": rec["rr25"] * 100,
            "atm30": rec["atm30"] * 100,
            "ret30": np.log(path.iloc[-1] / path.iloc[0]),
            "mdd30": (path / path.cummax() - 1).min(),
        })
    return pd.DataFrame(rows)


def block_bootstrap_ci(frame, target, coin=None, block_days=30, reps=1000, seed=731):
    """Moving-block bootstrap by calendar date; keeps BTC/ETH paired when pooled."""
    subset = frame if coin is None else frame[frame.coin == coin]
    xwide = subset.pivot(index="date", columns="coin", values="rr25").sort_index()
    ywide = subset.pivot(index="date", columns="coin", values=target).reindex(xwide.index)
    common = xwide.columns.intersection(ywide.columns)
    x = xwide[common].to_numpy(dtype=float)
    y = ywide[common].to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x, y = x[valid], y[valid]
    n = len(x)
    if n < block_days * 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(n / block_days))
    estimates = []
    max_start = n - block_days
    for _ in range(reps):
        starts = rng.integers(0, max_start + 1, size=block_count)
        idx = np.concatenate([np.arange(s, s + block_days) for s in starts])[:n]
        rho = stats.spearmanr(x[idx].ravel(), y[idx].ravel()).statistic
        if np.isfinite(rho):
            estimates.append(rho)
    if not estimates:
        return np.nan, np.nan
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def skew_summary(panel):
    rows = []
    for sample, subset in [("rolagens_mensais", panel[panel.date.dt.day == 1]),
                           ("todas_as_datas", panel)]:
        for target in ["ret30", "mdd30"]:
            group_specs = [(coin, subset[subset.coin == coin]) for coin in COINS]
            group_specs.append(("BTC+ETH", subset))
            for label, group in group_specs:
                if len(group) < 3:
                    continue
                rho, p_iid = stats.spearmanr(group.rr25, group[target])
                ci_low, ci_high = (np.nan, np.nan)
                if sample == "todas_as_datas":
                    ci_low, ci_high = block_bootstrap_ci(
                        subset, target, coin=None if label == "BTC+ETH" else label,
                        seed=731 + len(rows),
                    )
                rows.append({
                    "amostra": sample,
                    "moeda": label,
                    "alvo": target,
                    "n_datas": int(group.date.nunique()),
                    "n_observacoes": int(len(group)),
                    "spearman_rho": float(rho),
                    "p_iid_apenas_referencia": float(p_iid) if sample == "rolagens_mensais" else np.nan,
                    "ci95_block30_inferior": float(ci_low),
                    "ci95_block30_superior": float(ci_high),
                })
    return pd.DataFrame(rows)


def main():
    daily_data = {coin: data.load_daily(coin) for coin in COINS}
    surfaces = {coin: load_surfaces(coin) for coin in COINS}
    rows = []

    for coin in COINS:
        monthly_rolls = {day: rec for day, rec in surfaces[coin].items() if day.day == 1}
        spread = half_spread(coin)
        start, end = min(monthly_rolls), daily_data[coin].index[-1]

        methods = {
            "mensal_dvol": bt.Market(daily_data[coin], monthly_rolls),
            "svi_diaria": bt.Market(daily_data[coin], monthly_rolls,
                                    daily_surfaces=surfaces[coin]),
        }
        for method, market in methods.items():
            results = {}
            for kind in KINDS:
                result = bt.run_strategy(market, kind, start, end, spread)
                results[kind] = result
                result.drop(columns="strikes").to_csv(
                    os.path.join(TABLE_DIR, f"backtest_{coin}_{kind}_{method}.csv")
                )
            for kind, result in results.items():
                metrics = bt.metrics(result.ret, results["spot"].ret)
                rows.append({
                    "moeda": coin,
                    "marcacao": method,
                    "carteira": STRATEGY_NAMES[kind],
                    **metrics,
                    "Custos totais (% NAV ini.)": float(result.costs.iloc[-1] / result.nav.iloc[0]),
                })
            print(f"{coin}: concluida marcacao {method}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, "comparacao_svi_diaria.csv"), index=False)

    skew = pd.concat([forward_skew_rows(c, daily_data[c], surfaces[c]) for c in COINS],
                     ignore_index=True)
    skew.to_csv(os.path.join(TABLE_DIR, "skew_preditivo_diario.csv"), index=False)
    skew_summary(skew).to_csv(os.path.join(TABLE_DIR, "skew_preditivo_comparacao.csv"), index=False)
    print("Salvos: comparacao_svi_diaria.csv, skew_preditivo_diario.csv e "
          "skew_preditivo_comparacao.csv", flush=True)


if __name__ == "__main__":
    main()
