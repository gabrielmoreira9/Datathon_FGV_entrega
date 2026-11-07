"""Baixa negocios historicos de opcoes da Deribit em arquivos diarios.

Uso:
    python scripts/04_coletar_trades_diarios.py
    python scripts/04_coletar_trades_diarios.py --start 2024-02-01 --end 2026-11-07
    python scripts/04_coletar_trades_diarios.py --coins BTC

O script e retomavel: arquivos diarios ja existentes em data/raw/ sao pulados.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests


HISTORY_API = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PAGE_SIZE = 1000  # limite maximo documentado pela Deribit


def timestamp_ms(dt):
    return int(dt.timestamp() * 1000)


def fetch_window(session, coin, start_ms, end_ms):
    """Busca intervalo inclusivo; divide recursivamente páginas cheias.

    A divisão temporal evita avançar o cursor apenas pelo timestamp do último
    negócio, o que poderia perder negócios diferentes ocorridos no mesmo ms.
    """
    params = {
        "currency": coin,
        "kind": "option",
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "count": PAGE_SIZE,
        "sorting": "asc",
    }
    for attempt in range(6):
        try:
            response = session.get(HISTORY_API, params=params, timeout=45)
            if response.status_code == 429:
                time.sleep(min(2 ** attempt, 30))
                continue
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(payload["error"])
            result = payload["result"]
            break
        except (requests.RequestException, ValueError, KeyError) as exc:
            if attempt == 5:
                raise RuntimeError(f"Falha na API Deribit ({coin}, {start_ms}-{end_ms}): {exc}") from exc
            time.sleep(min(2 ** attempt, 30))
    else:
        raise RuntimeError(f"Limite de tentativas excedido ({coin}, {start_ms}-{end_ms})")

    trades = result.get("trades", [])
    if not result.get("has_more"):
        return trades

    if start_ms >= end_ms:
        raise RuntimeError(
            f"Mais de {PAGE_SIZE} negocios no mesmo milissegundo ({coin}, {start_ms}); "
            "a API nao permite pagina-los por timestamp."
        )

    midpoint = (start_ms + end_ms) // 2
    if midpoint < start_ms or midpoint >= end_ms:
        raise RuntimeError(f"Nao foi possivel subdividir intervalo: {start_ms}-{end_ms}")
    left = fetch_window(session, coin, start_ms, midpoint)
    right = fetch_window(session, coin, midpoint + 1, end_ms)
    return left + right


def download_day(session, coin, day):
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return fetch_window(session, coin, timestamp_ms(start), timestamp_ms(end) - 1)


def collect_one(coin, day):
    """Baixa e grava um dia, usando sessao HTTP isolada por worker."""
    session = requests.Session()
    session.headers.update({"User-Agent": "fgv-datathon-options-data-collector/1.0"})
    trades = download_day(session, coin, day)
    return coin, day, save_day(coin, day, trades)


def save_day(coin, day, trades):
    if not trades:
        return 0
    df = pd.DataFrame(trades)
    if "trade_id" in df:
        df = df.drop_duplicates("trade_id")
    df = df.sort_values([c for c in ("timestamp", "trade_id") if c in df]).reset_index(drop=True)
    filename = os.path.join(RAW, f"{coin}_trades_{day:%Y%m%d}.csv.gz")
    temp = filename + ".tmp"
    df.to_csv(temp, index=False, compression="gzip")
    os.replace(temp, filename)
    return len(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-02-01", help="primeiro dia UTC (inclusive)")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat(),
                        help="ultimo dia UTC (inclusive)")
    parser.add_argument("--coins", nargs="+", choices=["BTC", "ETH"], default=["BTC", "ETH"])
    parser.add_argument("--workers", type=int, default=4,
                        help="dias baixados em paralelo (padrao: 4; reduza se houver rate limit)")
    args = parser.parse_args()
    first = date.fromisoformat(args.start)
    last = date.fromisoformat(args.end)
    if last < first:
        parser.error("--end deve ser igual ou posterior a --start")

    os.makedirs(RAW, exist_ok=True)
    if args.workers < 1:
        parser.error("--workers deve ser pelo menos 1")
    tasks = []
    for coin in args.coins:
        day = first
        while day <= last:
            path = os.path.join(RAW, f"{coin}_trades_{day:%Y%m%d}.csv.gz")
            if not os.path.exists(path):
                tasks.append((coin, day))
            day += timedelta(days=1)
    total = len(tasks)
    print(f"Dias pendentes: {total:,}; workers: {args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect_one, coin, day): (coin, day) for coin, day in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            coin, day, count = future.result()
            if count:
                print(f"{coin} {day}: {count:,} negocios ({completed}/{total})", flush=True)
            elif day.day == 1:
                print(f"{coin} {day}: sem negocios ({completed}/{total})", flush=True)
    print("Coleta concluida. Dias sem negocios nao geram arquivo.")


if __name__ == "__main__":
    main()
