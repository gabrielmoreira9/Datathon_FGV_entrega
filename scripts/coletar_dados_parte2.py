"""
Coleta complementar (parte 2) - mais leve e tolerante a erros.
  pip install requests pandas
  python coletar_dados_parte2.py
Gera a pasta ./dados_parte2/  -> compacte e envie (igual a anterior).
Tempo estimado: 5-15 min. Se algum dia falhar, ele pula e continua.
"""
import os, time, requests, pandas as pd
from datetime import datetime, timezone, timedelta

API  = "https://www.deribit.com/api/v2/public/"
HIST = "https://history.deribit.com/api/v2/public/"
OUT  = "dados_parte2"; os.makedirs(OUT, exist_ok=True)
INICIO = datetime(2024, 1, 1, tzinfo=timezone.utc)
FIM    = datetime.now(timezone.utc)
ms = lambda d: int(d.timestamp() * 1000)

def get(base, metodo, **p):
    for t in range(4):
        try:
            r = requests.get(base + metodo, params=p, timeout=30)
            if r.status_code == 429: time.sleep(2 + 2 * t); continue
            j = r.json()
            if "result" in j: return j["result"]
            print("   erro API:", j.get("error")); time.sleep(1 + t)
        except Exception as e:
            print("   retry", metodo, e); time.sleep(2 + 2 * t)
    return None

def salvar(df, nome):
    df.to_csv(f"{OUT}/{nome}.csv.gz", index=False, compression="gzip")
    print(f"  -> {nome}: {len(df):,} linhas")

# ---------- 1) ETH: mesmos dados basicos que ja temos do BTC ----------
print("=== ETH basico ===")
for kind in ["option", "future"]:
    r = get(API, "get_book_summary_by_currency", currency="ETH", kind=kind)
    if r: salvar(pd.DataFrame(r), f"ETH_snapshot_{kind}")
r = get(API, "get_tradingview_chart_data", instrument_name="ETH-PERPETUAL",
        start_timestamp=ms(INICIO), end_timestamp=ms(FIM), resolution="1D")
if r: salvar(pd.DataFrame({k: r[k] for k in ["ticks","open","high","low","close","volume"]}), "ETH_perp_diario")
dv, ini = [], INICIO
while ini < FIM:
    fim = min(ini + timedelta(days=700), FIM)
    r = get(API, "get_volatility_index_data", currency="ETH", start_timestamp=ms(ini),
            end_timestamp=ms(fim), resolution="1D")
    if r: dv += r["data"]
    ini = fim
salvar(pd.DataFrame(dv, columns=["ts","open","high","low","close"]), "ETH_dvol")
fr, ini = [], INICIO
while ini < FIM:
    fim = min(ini + timedelta(days=30), FIM)
    r = get(API, "get_funding_rate_history", instrument_name="ETH-PERPETUAL",
            start_timestamp=ms(ini), end_timestamp=ms(fim))
    if r: fr += r
    ini = fim
salvar(pd.DataFrame(fr), "ETH_funding")

# ---------- 2) Negocios de opcoes em DIAS SELECIONADOS ----------
# 1o dia de cada mes (datas de rolagem do backtest) + dias de crise/calmaria
dias = pd.date_range("2024-01-02", FIM.date(), freq="MS").tolist()
dias += [pd.Timestamp(d) for d in ["2024-08-05", "2025-04-07", "2025-10-10",
                                   "2026-02-05", "2025-09-18", "2026-09-17", "2026-11-04"]]
dias = sorted(set(dias))
for moeda in ["BTC", "ETH"]:
    print(f"=== {moeda} opcoes: {len(dias)} dias ===")
    for d in dias:
        nome = f"{moeda}_trades_{d:%Y%m%d}"
        if os.path.exists(f"{OUT}/{nome}.csv.gz"): continue
        a = datetime(d.year, d.month, d.day, tzinfo=timezone.utc); b = a + timedelta(days=1)
        trades, cur = [], ms(a)
        for _ in range(40):                         # ate 40 mil trades por dia
            r = (get(HIST, "get_last_trades_by_currency_and_time", currency=moeda, kind="option",
                     start_timestamp=cur, end_timestamp=ms(b), count=1000, sorting="asc")
                 or get(API, "get_last_trades_by_currency_and_time", currency=moeda, kind="option",
                        start_timestamp=cur, end_timestamp=ms(b), count=1000, sorting="asc"))
            if not r or not r.get("trades"): break
            trades += r["trades"]
            if not r.get("has_more"): break
            cur = r["trades"][-1]["timestamp"] + 1
        if trades: salvar(pd.DataFrame(trades), nome)
        else: print("  (sem dados)", nome)

print("\nPronto! Compacte a pasta 'dados_parte2' e envie.")
