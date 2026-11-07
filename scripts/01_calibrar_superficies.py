"""Calibra a superficie SVI de cada dia com negocios (BTC e ETH) e salva em data/processed/.
Idempotente: pula dias ja calibrados. Uso: python scripts/01_calibrar_superficies.py [BTC|ETH] [max_dias]"""
import os, sys, pickle, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quant import data, svi, black76 as b

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "surfaces")
os.makedirs(OUT, exist_ok=True)


def summary(surf, F_ref, tau_target=30 / 365):
    """ATM IV e 25-Delta Risk Reversal / Butterfly em maturidade constante de 30 dias."""
    iv_fn = lambda K: float(surf.iv(np.log(K / F_ref), tau_target))
    Kc = b.strike_from_delta(F_ref, tau_target, iv_fn, 0.25, True)
    Kp = b.strike_from_delta(F_ref, tau_target, iv_fn, -0.25, False)
    atm, c25, p25 = iv_fn(F_ref), iv_fn(Kc), iv_fn(Kp)
    return dict(atm30=atm, rr25=c25 - p25, bf25=0.5 * (c25 + p25) - atm)


def run(coin, max_days=None):
    days = data.trade_days(coin)
    done = 0
    for d in days:
        fn = f"{OUT}/{coin}_{d:%Y%m%d}.pkl"
        if os.path.exists(fn):
            continue
        t0 = time.time()
        agg, trades, term = data.load_chain(coin, d)
        otm = agg[agg.otm]
        surf = svi.calibrate_surface(otm)
        # auditoria da IV: extracao propria (Newton/Brent) vs IV reportada pela Deribit
        iv_own, meth = b.implied_vol(agg.price * agg.F, agg.F, agg.K, agg.tau, agg.is_call)
        agg["iv_own"], agg["iv_method"] = iv_own, meth
        F_ref = float(np.exp(np.log(agg.F).median())) if len(agg) else np.nan
        S_ref = float(trades.index_price.median())
        rec = dict(coin=coin, date=d, S=S_ref, F_ref=F_ref, chain=agg, term=term,
                   slices=surf.slices, ts_window=(trades.timestamp.min(), trades.timestamp.max()))
        rec.update(summary(surf, S_ref * float(np.exp(term.r_imp.median() * 30 / 365))))
        with open(fn + ".tmp", "wb") as fh:
            pickle.dump(rec, fh)
        os.replace(fn + ".tmp", fn)                      # escrita atomica
        done += 1
        print(f"{coin} {d:%Y-%m-%d}: {len(surf.slices)} fatias, ATM30={rec['atm30']:.3f} "
              f"RR25={rec['rr25']:+.3f} ({time.time()-t0:.0f}s)", flush=True)
        if max_days and done >= max_days:
            break


if __name__ == "__main__":
    coins = [sys.argv[1]] if len(sys.argv) > 1 else ["BTC", "ETH"]
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for c in coins:
        run(c, mx)
