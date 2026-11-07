"""
Coletor de dados Deribit (API publica, sem chave) para o 7o Datathon FGV EESP.
Rode no SEU computador:   pip install requests pandas
                          python coletar_dados_deribit.py
Saida: pasta ./dados_deribit/  -> compacte e envie para o Claude.
Tempo estimado: 10-30 min (depende da janela historica).
"""
import os, time, requests, pandas as pd
from datetime import datetime, timezone, timedelta

API  = "https://www.deribit.com/api/v2/public/"
HIST = "https://history.deribit.com/api/v2/public/"
OUT  = "dados_deribit"; os.makedirs(OUT, exist_ok=True)
MOEDAS = ["BTC", "ETH"]
INICIO = datetime(2024, 1, 1, tzinfo=timezone.utc)   # janela do backtest
FIM    = datetime.now(timezone.utc)
ms = lambda d: int(d.timestamp() * 1000)

def get(base, metodo, **p):
    for t in range(5):
        try:
            r = requests.get(base + metodo, params=p, timeout=30)
            if r.status_code == 429: time.sleep(2 + t * 2); continue
            r.raise_for_status(); return r.json()["result"]
        except Exception as e:
            print("  retry", metodo, e); time.sleep(2 + t * 2)
    raise RuntimeError(metodo)

def salvar(df, nome):
    df.to_csv(f"{OUT}/{nome}.csv.gz", index=False, compression="gzip")
    print(f"  -> {nome}: {len(df):,} linhas")

def candles(instr, res="1D"):
    """OHLCV diario (futuros, perps e indice)."""
    rows, ini = [], INICIO
    while ini < FIM:
        fim = min(ini + timedelta(days=900), FIM)
        r = get(API, "get_tradingview_chart_data", instrument_name=instr,
                start_timestamp=ms(ini), end_timestamp=ms(fim), resolution=res)
        if r.get("ticks"):
            rows.append(pd.DataFrame({k: r[k] for k in ["ticks","open","high","low","close","volume"]}))
        ini = fim
    df = pd.concat(rows) if rows else pd.DataFrame()
    if len(df): df["data"] = pd.to_datetime(df.ticks, unit="ms"); df["instrumento"] = instr
    return df

for c in MOEDAS:
    print(f"\n=== {c} ===")
    # 1) Snapshot atual da cadeia completa (opcoes + futuros): mark_iv, bid/ask, OI, underlying
    for kind in ["option", "future"]:
        salvar(pd.DataFrame(get(API, "get_book_summary_by_currency", currency=c, kind=kind)),
               f"{c}_snapshot_{kind}")
    salvar(pd.DataFrame(get(API, "get_instruments", currency=c, kind="option", expired="false")),
           f"{c}_instrumentos_opcoes")

    # 2) Perpetuo diario + historico de funding
    salvar(candles(f"{c}-PERPETUAL"), f"{c}_perp_diario")
    fr, ini = [], INICIO
    while ini < FIM:
        fim = min(ini + timedelta(days=30), FIM)
        fr += get(API, "get_funding_rate_history", instrument_name=f"{c}-PERPETUAL",
                  start_timestamp=ms(ini), end_timestamp=ms(fim))
        ini = fim
    salvar(pd.DataFrame(fr), f"{c}_funding")

    # 3) Indice DVOL (vol implicita 30d) diario
    dv, ini = [], INICIO
    while ini < FIM:
        fim = min(ini + timedelta(days=900), FIM)
        r = get(API, "get_volatility_index_data", currency=c,
                start_timestamp=ms(ini), end_timestamp=ms(fim), resolution="1D")
        dv += r["data"]; ini = fim
    salvar(pd.DataFrame(dv, columns=["ts","open","high","low","close"]), f"{c}_dvol")

    # 4) HISTORICO de negocios de opcoes (traz iv, index_price, mark_price por trade)
    #    -> base para reconstruir superficies SVI diarias e o backtest.
    trades, ini = [], INICIO
    while ini < FIM:
        fim = min(ini + timedelta(days=1), FIM)
        cur = ms(ini)
        while True:
            r = get(HIST, "get_last_trades_by_currency_and_time", currency=c, kind="option",
                    start_timestamp=cur, end_timestamp=ms(fim), count=1000, sorting="asc")
            t = r["trades"]; trades += t
            if not r.get("has_more") or not t: break
            cur = t[-1]["timestamp"] + 1
        ini = fim
        if ini.day == 1: print("  opcoes ate", ini.date(), f"({len(trades):,} trades)")
    salvar(pd.DataFrame(trades), f"{c}_opcoes_trades")

    # 5) Futuros datados ativos (diario) para estrutura a termo r_imp(tau)
    for f in pd.read_csv(f"{OUT}/{c}_snapshot_future.csv.gz").instrument_name:
        if "PERPETUAL" not in f: salvar(candles(f), f"{c}_fut_{f}")

print("\nPronto! Compacte a pasta 'dados_deribit' e envie.")
