"""Roda backtests, metricas e gera todas as figuras/tabelas em output/.
Uso: python scripts/02_analise.py"""
import os, sys, glob, pickle, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quant import data, svi, black76 as b, backtest as bt

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG, TAB = f"{ROOT}/output/figs", f"{ROOT}/output/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
NAVY, TEAL, ORANGE, GRAY, GOLD = "#0B2545", "#1B98E0", "#E4572E", "#8D99AE", "#F4A259"
COL = {"spot": GRAY, "put": TEAL, "collar": NAVY, "dgh": ORANGE}
NAME = {"spot": "C1 Spot", "put": "C2 Protective Put", "collar": "C3 Zero-Cost Collar", "dgh": "C4 Delta-Gamma Hedge"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "savefig.dpi": 200,
                     "axes.grid": True, "grid.alpha": 0.25})
SUM = {}
import matplotlib.dates as mdates


def datefmt(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight"); plt.close(fig)


def load_surfaces(coin):
    recs = {}
    for f in sorted(glob.glob(f"{ROOT}/data/processed/surfaces/{coin}_*.pkl")):
        r = pickle.load(open(f, "rb")); recs[r["date"]] = r
    return recs


DAILY = {c: data.load_daily(c) for c in ["BTC", "ETH"]}
SURF = {c: load_surfaces(c) for c in ["BTC", "ETH"]}
SNAP = {c: data.load_snapshot(c) for c in ["BTC", "ETH"]}

# ======================= 1. Fatos estilizados / caudas / VRP ======================
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for c, colr in [("BTC", NAVY), ("ETH", ORANGE)]:
    r = DAILY[c].ret.dropna()
    kur = stats.kurtosis(r); sk = stats.skew(r)
    jb = stats.jarque_bera(r)
    SUM[f"{c}_kurtosis"], SUM[f"{c}_skew"], SUM[f"{c}_jb_p"] = float(kur), float(sk), float(jb.pvalue)
    SUM[f"{c}_vol_ann"] = float(r.std() * np.sqrt(365))
    z = (r - r.mean()) / r.std()
    h, e = np.histogram(z, bins=70, density=True)
    ax[0].semilogy((e[1:] + e[:-1]) / 2, h, "o", ms=3, color=colr, label=f"{c} (curtose exc. {kur:.1f})")
    nav = DAILY[c].S / DAILY[c].S.cummax() - 1
    ax[1].plot(nav.index, nav * 100, color=colr, lw=1.3, label=f"{c} (MDD {nav.min()*100:.0f}%)")
    SUM[f"{c}_mdd_spot"] = float(nav.min())
    # VRP: DVOL (30d implicita) vs vol realizada nos 30 dias seguintes
    rv = r.rolling(30).std().shift(-30) * np.sqrt(365) * 100
    vrp = (DAILY[c].dvol - rv).dropna()
    SUM[f"{c}_vrp_mean"], SUM[f"{c}_vrp_pos"] = float(vrp.mean()), float((vrp > 0).mean())
    ax[2].plot(vrp.index, vrp.rolling(30).mean(), color=colr, lw=1.6, label=f"{c}: média {vrp.mean():.1f} pts")
x = np.linspace(-8, 8, 300)
ax[0].semilogy(x, stats.norm.pdf(x), "k--", lw=1, label="Normal")
ax[0].set_ylim(1e-4, 1); ax[0].set_xlim(-8, 8)
ax[0].set_title("Densidade dos retornos diários padronizados (log)"); ax[0].legend(fontsize=8)
ax[1].set_title("Drawdown do ativo à vista (%)"); ax[1].legend(fontsize=8)
ax[2].axhline(0, color="k", lw=0.8)
ax[2].set_title("Prêmio de risco de variância: DVOL − vol. realizada 30d"); ax[2].legend(fontsize=8)
ax[2].set_ylabel("pontos de vol (média móvel 30d)"); datefmt(ax[1]); datefmt(ax[2])
save(fig, "s02_caudas_drawdown_vrp")

# ======================= 2. Payoff linear vs inverso + Delta dual ======================
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ST = np.linspace(30000, 140000, 400); K = 76000
ax[0].plot(ST / 1e3, np.maximum(K - ST, 0) / 1e3, color=NAVY, lw=2, label="Put — payoff USD (linear)")
ax0b = ax[0].twinx()
ax0b.plot(ST / 1e3, np.maximum(K / ST - 1, 0), color=ORANGE, lw=2, label="Put — payoff em BTC (hiperbólico)")
ax[0].set_xlabel("S_T (mil USD)"); ax[0].set_ylabel("USD (mil)", color=NAVY); ax0b.set_ylabel("BTC", color=ORANGE)
ax[0].set_title("Put inversa K = 76k: USD vs BTC"); ax0b.spines["right"].set_visible(True); ax0b.grid(False)
ax[1].plot(ST / 1e3, np.maximum(ST - K, 0) / 1e3, color=NAVY, lw=2)
ax1b = ax[1].twinx(); ax1b.plot(ST / 1e3, np.maximum(1 - K / ST, 0), color=ORANGE, lw=2)
ax1b.axhline(1, color=ORANGE, ls=":", lw=1); ax1b.text(35, 0.93, "teto: 1 BTC", color=ORANGE, fontsize=8)
ax[1].set_title("Call inversa K = 76k: USD vs BTC"); ax[1].set_xlabel("S_T (mil USD)")
ax1b.spines["right"].set_visible(True); ax1b.grid(False)
F, tau = 76000, 30 / 365; Ks = np.linspace(50000, 110000, 200)
for cp, ls in [(False, "-"), (True, "--")]:
    du = b.greeks_usd(F, Ks, tau, 0.42, cp)["delta"]; dc = b.delta_coin(F, Ks, tau, 0.42, cp)
    ax[2].plot(Ks / 1e3, du, color=NAVY, ls=ls, label=f"{'Call' if cp else 'Put'} Δ USD (Black-76)")
    ax[2].plot(Ks / 1e3, dc, color=ORANGE, ls=ls, label=f"{'Call' if cp else 'Put'} Δ em moeda (ajust. prêmio)")
ax[2].set_title("Delta dual: USD vs moeda (30d, σ = 42%)"); ax[2].set_xlabel("Strike (mil USD)"); ax[2].legend(fontsize=7)
save(fig, "s03_payoff_inverso_delta_dual")

# ======================= 3. Pipeline + estrutura a termo ======================
fn = {}
for c in ["BTC", "ETH"]:
    raw = sum(len(pd.read_csv(f, usecols=["timestamp"])) for f in glob.glob(f"{ROOT}/data/raw/{c}_trades_*.csv.gz"))
    win = sum(r["chain"].n.sum() for r in SURF[c].values())
    inst = sum(len(r["chain"]) for r in SURF[c].values())
    otm = sum(int(r["chain"].otm.sum()) for r in SURF[c].values())
    used = sum(sum(s["n"] for s in r["slices"]) for r in SURF[c].values())
    nsl = sum(len(r["slices"]) for r in SURF[c].values())
    fn[c] = [raw, win, inst, otm, used]; SUM[f"{c}_funnel"] = fn[c]; SUM[f"{c}_nslices"] = nsl
fig, ax = plt.subplots(1, 2, figsize=(15, 4.3))
labels = ["Negócios brutos\n(41 dias)", "Janela 4h de\nmaior liquidez", "Instrumentos\nagregados", "OTM\n(puts k<0, calls k≥0)", "Pontos usados\nno SVI (τ ≥ 2d, |k|<1,2)"]
xx = np.arange(5)
ax[0].bar(xx - 0.2, fn["BTC"], 0.4, color=NAVY, label="BTC"); ax[0].bar(xx + 0.2, fn["ETH"], 0.4, color=ORANGE, label="ETH")
ax[0].set_yscale("log"); ax[0].set_xticks(xx); ax[0].set_xticklabels(labels, fontsize=8)
for i in range(5):
    ax[0].text(i - 0.2, fn["BTC"][i] * 1.15, f"{fn['BTC'][i]:,}".replace(",", "."), ha="center", fontsize=7)
    ax[0].text(i + 0.2, fn["ETH"][i] * 1.15, f"{fn['ETH'][i]:,}".replace(",", "."), ha="center", fontsize=7)
ax[0].set_title("Funil de engenharia de dados (escala log)"); ax[0].legend()
for c, colr in [("BTC", NAVY), ("ETH", ORANGE)]:
    fut = SNAP[c][1]
    ax[1].plot(fut.tau * 365, fut.r_imp * 100, "o-", color=colr, label=f"{c} — futuros datados hoje (05/11/26)")
    h = []
    for d, r in SURF[c].items():
        tm = r["term"]; tm = tm[tm.from_parity & (tm.tau > 10 / 365) & (tm.tau < 0.3)]
        if len(tm): h.append((d, tm.r_imp.median()))
    h = pd.DataFrame(h, columns=["d", "r"]).set_index("d")
    SUM[f"{c}_rimp_hist_mean"], SUM[f"{c}_rimp_hist_min"], SUM[f"{c}_rimp_hist_max"] = float(h.r.mean()), float(h.r.min()), float(h.r.max())
ax[1].set_xscale("log"); ax[1].set_xlabel("dias até o vencimento (log)"); ax[1].set_ylabel("r_imp(τ) % a.a.")
ax[1].set_title("Estrutura a termo da taxa implícita r_imp(τ) = ln(F/S)/τ"); ax[1].legend(fontsize=8)
save(fig, "s04_funil_estrutura_termo")

# série histórica de r_imp 10d-110d (paridade put-call)
fig, ax = plt.subplots(figsize=(7, 3.2))
for c, colr in [("BTC", NAVY), ("ETH", ORANGE)]:
    h = [(d, r["term"][r["term"].from_parity & (r["term"].tau > 10 / 365) & (r["term"].tau < 0.3)].r_imp.median()) for d, r in SURF[c].items()]
    h = pd.DataFrame(h, columns=["d", "r"]).dropna().set_index("d")
    ax.plot(h.index, h.r * 100, "o-", ms=3, color=colr, label=c); datefmt(ax)
ax.set_title("Taxa implícita (10–110 dias) via paridade put-call inversa"); ax.set_ylabel("% a.a."); ax.legend()
save(fig, "s04_rimp_historico")

# ======================= 4. Ajuste SVI + auditoria IV ======================
def fit_plot(ax, rec, title, ntaus=4):
    ch = rec["chain"]; o = ch[ch.otm]
    sl = [s for s in rec["slices"] if s["tau"] > 5 / 365]
    pick = [sl[int(i)] for i in np.linspace(0, len(sl) - 1, ntaus)]
    cols = cm.viridis(np.linspace(0, 0.9, len(pick)))
    for s, cc in zip(pick, cols):
        pts = o[np.isclose(o.tau, s["tau"])]
        kk = np.linspace(pts.k.min() - 0.05, pts.k.max() + 0.05, 200)
        ax.scatter(pts.k, pts.iv * 100, s=12, color=cc, alpha=0.7)
        ax.plot(kk, np.sqrt(svi.svi_w(kk, s["params"]) / s["tau"]) * 100, color=cc, lw=2,
                label=f"τ = {s['tau']*365:.0f}d  RMSE {s['rmse_vol']*100:.2f} pts  R² {s['r2']:.3f}")
    ax.set_xlabel("log-moneyness k = ln(K/F)"); ax.set_ylabel("vol implícita (%)"); ax.set_title(title); ax.legend(fontsize=7)


fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
fit_plot(ax[0], SURF["BTC"][pd.Timestamp("2026-11-04")], "BTC 04/11/2026 — mercado (pontos) vs Raw SVI")
fit_plot(ax[1], SURF["ETH"][pd.Timestamp("2026-11-04")], "ETH 04/11/2026 — mercado (pontos) vs Raw SVI")
allc = pd.concat([r["chain"].assign(coin=c) for c in ["BTC", "ETH"] for r in SURF[c].values()])
err = ((allc.iv_own - allc.iv) * 100).dropna()
SUM["iv_audit_n"] = int(len(err)); SUM["iv_audit_med_abs"] = float(err.abs().median())
SUM["iv_audit_p95_abs"] = float(err.abs().quantile(0.95)); SUM["iv_newton_share"] = float((allc.iv_method == 1).mean() / (allc.iv_method > 0).mean())
SUM["iv_conv_rate"] = float((allc.iv_method > 0).mean())
ax[2].hist(err.clip(-3, 3), bins=80, color=NAVY)
ax[2].set_title("Auditoria: IV própria (Newton/Brent) − IV Deribit")
ax[2].set_xlabel("pontos de vol"); ax[2].text(0.02, 0.95, f"n = {len(err):,}\nmediana |erro| = {err.abs().median():.2f} pts\np95 |erro| = {err.abs().quantile(.95):.2f} pts".replace(",", "."),
                                             transform=ax[2].transAxes, va="top", fontsize=9)
save(fig, "s05_svi_fit_auditoria")

# qualidade de calibracao (todas as fatias) + checagem de butterfly no intervalo cotado
q = []
for c in ["BTC", "ETH"]:
    for d, r in SURF[c].items():
        o = r["chain"][r["chain"].otm]
        for s in r["slices"]:
            pts = o[np.isclose(o.tau, s["tau"])]
            kk = np.linspace(pts.k.min(), pts.k.max(), 200) if len(pts) else svi.KGRID
            w = svi.svi_w(kk, s["params"])
            q.append(dict(coin=c, date=d, tau=s["tau"], rmse=s["rmse_vol"] * 100, r2=s["r2"],
                          g_ok=bool(svi.g_density(kk, s["params"]).min() >= -1e-4),
                          lee_ok=bool(s["params"][1] * (1 + abs(s["params"][2])) <= 2 + 1e-6),
                          crisis=d in [pd.Timestamp(x) for x in ["2024-08-05", "2025-04-07", "2025-10-10", "2026-02-05"]]))
q = pd.DataFrame(q); q.to_csv(f"{TAB}/qualidade_svi.csv", index=False)
SUM["svi_slices"] = int(len(q)); SUM["svi_rmse_med"] = float(q.rmse.median()); SUM["svi_r2_med"] = float(q.r2.median())
SUM["svi_rmse_med_calm"] = float(q[~q.crisis].rmse.median()); SUM["svi_rmse_med_crisis"] = float(q[q.crisis].rmse.median())
SUM["svi_g_ok"] = float(q.g_ok.mean()); SUM["svi_lee_ok"] = float(q.lee_ok.mean())
# calendario: w crescente em tau no intervalo k em [-0.3, 0.3]
cal = []
for c in ["BTC", "ETH"]:
    for d, r in SURF[c].items():
        if len(r["slices"]) < 2:
            continue  # sem duas maturidades nao ha teste de arbitragem de calendario
        kk = np.linspace(-0.3, 0.3, 61); W = np.array([svi.svi_w(kk, s["params"]) for s in r["slices"]])
        cal.append(np.mean(np.diff(W, axis=0) >= -1e-5))
SUM["svi_calendar_n"] = int(len(cal))
SUM["svi_calendar_ok"] = float(np.mean(cal)) if cal else np.nan

# ======================= 5. Superficies 3D calma vs crise + skew ======================
def surf3d(ax, rec, title, zmax):
    S_ = svi.SVISurface(rec["slices"])
    kk = np.linspace(-0.4, 0.25, 50); tt = np.linspace(14 / 365, min(0.5, S_.taus[-1]), 40)
    Kg, Tg = np.meshgrid(kk, tt)
    Z = np.array([S_.iv(kk, t_) for t_ in tt]) * 100
    ax.plot_surface(Kg, Tg * 365, Z, cmap="viridis", vmin=30, vmax=zmax, edgecolor="none", alpha=0.95)
    ax.set_xlabel("ln(K/F)"); ax.set_ylabel("dias"); ax.set_zlim(20, zmax)
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.view_init(25, -55)


fig = plt.figure(figsize=(16, 4.8))
a1 = fig.add_subplot(1, 3, 1, projection="3d"); surf3d(a1, SURF["BTC"][pd.Timestamp("2025-09-18")], "BTC — regime normal (18/09/2025)", 130)
a2 = fig.add_subplot(1, 3, 2, projection="3d"); surf3d(a2, SURF["BTC"][pd.Timestamp("2024-08-05")], "BTC — crise (05/08/2024, carry trade do iene)", 130)
a3 = fig.add_subplot(1, 3, 3)
for c, colr in [("BTC", NAVY), ("ETH", ORANGE)]:
    rr = pd.Series({d: r["rr25"] * 100 for d, r in SURF[c].items()}).sort_index()
    a3.plot(rr.index, rr, "o-", ms=3, color=colr, label=f"RR 25Δ {c}")
a3.axhline(0, color="k", lw=0.8); a3.set_ylabel("RR 25Δ 30d (pts de vol)")
a3b = a3.twinx(); a3b.plot(DAILY["BTC"].S.index, DAILY["BTC"].S / 1e3, color=GRAY, lw=1, alpha=0.7)
a3b.set_ylabel("BTC (mil USD)", color=GRAY); a3b.grid(False); a3b.spines["right"].set_visible(True)
a3.set_title("Skew 25-Delta (call − put) vs preço do BTC"); a3.legend(fontsize=8, loc="lower left"); datefmt(a3)
fig.subplots_adjust(wspace=0.35); save(fig, "s06_superficies_skew")

# poder preditivo do skew: RR na rolagem vs retorno / max drawdown dos 30 dias seguintes
pred = []
for c in ["BTC", "ETH"]:
    S_ = DAILY[c].S
    for d, r in SURF[c].items():
        if d.day != 1: continue
        fwd = S_.loc[d:d + pd.Timedelta(days=30)]
        if len(fwd) < 25: continue
        pred.append(dict(coin=c, date=d, rr=r["rr25"] * 100, atm=r["atm30"] * 100,
                         ret30=np.log(fwd.iloc[-1] / fwd.iloc[0]), mdd30=(fwd / fwd.cummax() - 1).min()))
pred = pd.DataFrame(pred); pred.to_csv(f"{TAB}/skew_preditivo.csv", index=False)
for y in ["ret30", "mdd30"]:
    rho, p = stats.spearmanr(pred.rr, pred[y]); SUM[f"skew_{y}_rho"], SUM[f"skew_{y}_p"] = float(rho), float(p)
SUM["skew_n"] = int(len(pred))
rr_all = pd.concat([pd.Series({d: r["rr25"] for d, r in SURF[c].items()}) for c in ["BTC", "ETH"]])
SUM["rr_neg_share"] = float((rr_all < 0).mean())
for c in ["BTC", "ETH"]:
    for d in ["2024-08-05", "2025-09-18", "2026-11-04"]:
        r = SURF[c].get(pd.Timestamp(d))
        if r: SUM[f"{c}_{d}_atm"], SUM[f"{c}_{d}_rr"] = float(r["atm30"]), float(r["rr25"])

# densidade neutra ao risco (Breeden-Litzenberger) calma vs crise
fig, ax = plt.subplots(figsize=(7, 3.6))
for d, colr, lab in [("2025-09-18", NAVY, "normal 18/09/25"), ("2024-08-05", ORANGE, "crise 05/08/24")]:
    S_ = svi.SVISurface(SURF["BTC"][pd.Timestamp(d)]["slices"]); t_ = 30 / 365
    kk = np.linspace(-0.6, 0.5, 600); Kk = np.exp(kk)
    C = b.price_usd(1.0, Kk, t_, S_.iv(kk, t_), True)
    dens = np.gradient(np.gradient(C, Kk), Kk)
    ax.plot(kk, dens, color=colr, lw=2, label=f"BTC {lab}")
ax.set_xlabel("ln(K/F)"); ax.set_title("Densidade neutra ao risco 30d (Breeden-Litzenberger sobre SVI)"); ax.legend()
save(fig, "s06_densidade_bl")

# ======================= 6. Backtests ======================
RES, MET, GROSS = {}, {}, {}
CACHE = f"{ROOT}/data/processed/cache_backtest.pkl"
if os.path.exists(CACHE):
    RES, GROSS = pickle.load(open(CACHE, "rb"))
for c in ["BTC", "ETH"]:
    s, _ = SNAP[c]
    m = s[(s.tau > 15 / 365) & (s.tau < 45 / 365) & (s.k.abs() < 0.3) & (s.bid_price > 0)]
    hs = float(((m.ask_price - m.bid_price) / 2 / m.mark_price).median()); SUM[f"{c}_half_spread"] = hs
    rolls = {d: r for d, r in SURF[c].items() if d.day == 1}
    mkt = bt.Market(DAILY[c], rolls)
    st, en = min(rolls), DAILY[c].index[-1]
    if c not in RES:
        RES[c] = {k: bt.run_strategy(mkt, k, st, en, hs) for k in NAME}
        GROSS[c] = {k: bt.run_strategy(mkt, k, st, en, hs, cost_mult=0.0) for k in NAME}
        pickle.dump((RES, GROSS), open(CACHE, "wb"))
    MET[c] = pd.DataFrame({NAME[k]: bt.metrics(v.ret, RES[c]["spot"].ret) for k, v in RES[c].items()}).T
    MET[c]["Custos totais (% NAV ini.)"] = [RES[c][k].costs.iloc[-1] / RES[c][k].nav.iloc[0] for k in NAME]
    MET[c]["Retorno a.a. sem custos"] = [bt.metrics(GROSS[c][k].ret, RES[c]["spot"].ret)["Retorno a.a."] for k in NAME]
    MET[c].to_csv(f"{TAB}/metricas_{c}.csv")
    for k, v in RES[c].items(): v.drop(columns="strikes").to_csv(f"{TAB}/backtest_{c}_{k}.csv")
    SUM[f"{c}_period"] = [str(st.date()), str(en.date())]
    print(c); print(MET[c].round(3).to_string())

fig, ax = plt.subplots(2, 2, figsize=(15, 6.4), gridspec_kw={"height_ratios": [2, 1]}, sharex="col")
for j, c in enumerate(["BTC", "ETH"]):
    for k, v in RES[c].items():
        nav = v.nav / v.nav.iloc[0] * 100
        ax[0, j].plot(nav.index, nav, color=COL[k], lw=1.8 if k != "spot" else 1.3, label=NAME[k])
        dd = (v.nav / v.nav.cummax() - 1) * 100
        ax[1, j].plot(dd.index, dd, color=COL[k], lw=1.2)
    ax[0, j].set_title(f"{c}: curva de capital (base 100, líquida de custos)"); ax[0, j].legend(fontsize=8); datefmt(ax[1, j])
    ax[1, j].set_title("Underwater drawdown (%)")
save(fig, "s09_equity_underwater")

# ======================= 7. Gregas + Taylor + stress ======================
# superficie de hoje calibrada no snapshot (mark_iv)
SNC = f"{ROOT}/data/processed/cache_snapshot_svi.pkl"
if os.path.exists(SNC):
    SN = pickle.load(open(SNC, "rb"))
else:
    SN = {}
    for c in ["BTC"]:
        s, _ = SNAP[c]; o = s[s.otm & (s.tau > 2 / 365) & s.iv.notna()]
        SN[c] = svi.calibrate_surface(o)
    pickle.dump(SN, open(SNC, "wb"))
fig, ax = plt.subplots(1, 4, figsize=(16, 3.9))
S_ = SN["BTC"]; F0 = float(SNAP["BTC"][0].query("tau > 15/365 and tau < 45/365").underlying_price.median())
kk = np.linspace(-0.4, 0.3, 120); Kk = F0 * np.exp(kk)
for t_d, cc in zip([7, 30, 90, 180], cm.viridis(np.linspace(0, 0.85, 4))):
    t_ = t_d / 365; g = b.greeks_usd(F0, Kk, t_, S_.iv(kk, t_), False)
    ax[0].plot(kk, g["delta"], color=cc, label=f"{t_d}d"); ax[1].plot(kk, g["gamma"] * F0 * 0.01, color=cc)
    ax[2].plot(kk, g["vega"] / 100, color=cc); ax[3].plot(kk, g["vanna"] * 0.01, color=cc, ls="-")
ax[0].set_title("Delta (put, USD)"); ax[1].set_title("Gamma: ΔDelta p/ 1% de S"); ax[2].set_title("Vega: USD por 1 pt de vol")
ax[3].set_title("Vanna: ΔDelta p/ 1 pt de vol"); ax[0].legend(fontsize=8)
for a in ax: a.set_xlabel("ln(K/F)")
save(fig, "s07_gregas_btc")

# Taylor P&L da carteira 4 (BTC)
v = RES["BTC"]["dgh"].copy(); dS = v.S.diff(); dsig = v.dvol.diff() / 100; dt = 1 / 365
lag = v.shift(1)
att = pd.DataFrame({"Delta": lag.delta * dS, "Gamma": 0.5 * lag.gamma * dS**2, "Vega": lag.vega * dsig,
                    "Theta": lag.theta * dt, "Vanna+Volga": lag.vanna * dS * dsig + 0.5 * lag.volga * dsig**2})
roll_days = set(d for d in SURF["BTC"] if d.day == 1)
fund = -lag.perp_q * lag.S * DAILY["BTC"].funding_d.reindex(v.index)
carry = lag.cash * (np.exp(0.04 * dt) - 1)
att["Funding+juros"] = fund + carry
att["Custos"] = -v.costs.diff()
dnav = v.nav.diff()
mask = ~v.index.isin(list(roll_days)) & ~v.index.isin([d + pd.Timedelta(days=1) for d in roll_days])
att = att[mask].dropna(); dnav = dnav[mask].reindex(att.index)
att["Resíduo"] = dnav - att.sum(axis=1)
SUM["taylor_r2"] = float(1 - att["Resíduo"].var() / dnav.var())
SUM["taylor_share"] = {k: float(att[k].sum() / v.nav.iloc[0]) for k in att.columns}
mon = att.resample("QE").sum() / v.nav.iloc[0] * 100
mon.index = [f"{d.year}T{(d.month - 1) // 3 + 1}" for d in mon.index]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
bottom_p = np.zeros(len(mon)); bottom_n = np.zeros(len(mon))
pal = [GRAY, NAVY, TEAL, GOLD, "#7B2CBF", "#2A9D8F", ORANGE, "#BBBBBB"]
for colname, cc in zip(mon.columns, pal):
    vals = mon[colname].values; pos = np.where(vals > 0, vals, 0); neg = np.where(vals < 0, vals, 0)
    ax[0].bar(mon.index, pos, 0.7, bottom=bottom_p, color=cc, label=colname); ax[0].bar(mon.index, neg, 0.7, bottom=bottom_n, color=cc)
    bottom_p += pos; bottom_n += neg
ax[0].plot(mon.index, mon.sum(axis=1), "k_", ms=18, mew=2, label="P&L total")
ax[0].set_title(f"C4 BTC — atribuição de P&L por Série de Taylor (trimestral, % NAV); R² diário = {SUM['taylor_r2']:.2f}")
ax[0].legend(fontsize=7, ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False); ax[0].tick_params(axis="x", labelsize=8)

# stress: choque instantaneo de preco x vol (BTC, livro de hoje, 30d)
F0 = float(SNAP["BTC"][0].query("tau > 15/365 and tau < 45/365").underlying_price.median()); t_ = 30 / 365
ivf = lambda K: float(S_.iv(np.log(K / F0), t_))
Kp20 = b.strike_from_delta(F0, t_, ivf, -0.20, False); Kp25 = b.strike_from_delta(F0, t_, ivf, -0.25, False)
from scipy.optimize import brentq
P25 = float(b.price_usd(F0, Kp25, t_, ivf(Kp25), False))
Kc = brentq(lambda K: float(b.price_usd(F0, K, t_, ivf(K), True)) - P25, F0, 3 * F0)
books = {"collar": [(1, None, None), (1, Kp25, False), (-1, Kc, True)]}
shS = np.linspace(-0.4, 0.4, 17); shV = np.linspace(-0.4, 1.0, 15)


def book_val(book, F, volmult, tt):
    tot = 0.0
    for q_, K_, cp in book:
        if K_ is None: tot += q_ * F; continue
        tot += q_ * float(b.price_usd(F, K_, tt, ivf(K_) * volmult, cp))
    return tot


base = book_val(books["collar"], F0, 1.0, t_)
H = np.array([[(book_val(books["collar"], F0 * (1 + a), 1 + vv, t_ - 1 / 365) - base) / base * 100 for a in shS] for vv in shV])
im = ax[1].imshow(H, origin="lower", aspect="auto", cmap="RdYlGn", vmin=-40, vmax=40,
                  extent=[shS[0] * 100, shS[-1] * 100, shV[0] * 100, shV[-1] * 100])
plt.colorbar(im, ax=ax[1], label="P&L % do NAV")
ax[1].set_xlabel("choque no preço do BTC (%)"); ax[1].set_ylabel("choque relativo na vol (%)")
ax[1].set_title("Stress C3 Collar (livro de hoje): preço × vol")
cs = ax[1].contour(shS * 100, shV * 100, H, levels=[-20, -10, 0, 10, 20], colors="k", linewidths=0.6)
ax[1].clabel(cs, fmt="%d%%", fontsize=7)
save(fig, "s07_taylor_stress")
SUM["stress_collar_-40"] = float(H[np.argmin(abs(shV - 0.4)), 0]); SUM["stress_spot_-40"] = -40.0
SUM["today_collar"] = dict(F=F0, Kp=Kp25, Kc=Kc, put_prem_pct=P25 / F0)

# ======================= 8. Perfis de payoff das carteiras ======================
fig, ax = plt.subplots(figsize=(7, 3.6))
STg = np.linspace(0.5, 1.5, 300) * F0
Kp20v = Kp20; P20 = float(b.price_usd(F0, Kp20, t_, ivf(Kp20), False))
ax.plot(STg / F0 * 100 - 100, (STg - F0) / F0 * 100, color=GRAY, label="C1 Spot")
ax.plot(STg / F0 * 100 - 100, (STg - F0 + np.maximum(Kp20 - STg, 0) - P20) / F0 * 100, color=TEAL, label=f"C2 Put 20Δ (K={Kp20/F0*100:.0f}%)")
ax.plot(STg / F0 * 100 - 100, (STg - F0 + np.maximum(Kp25 - STg, 0) - np.maximum(STg - Kc, 0)) / F0 * 100, color=NAVY, lw=2,
        label=f"C3 Collar (K_put={Kp25/F0*100:.0f}%, K_call={Kc/F0*100:.0f}%)")
Kwp = b.strike_from_delta(F0, t_, ivf, -0.10, False); Kwc = b.strike_from_delta(F0, t_, ivf, 0.10, True)
cred = sum(q_ * float(b.price_usd(F0, K_, t_, ivf(K_), cp)) for q_, K_, cp in [(1, F0, True), (1, F0, False), (-1, Kwp, False), (-1, Kwc, True)])
pay4 = -np.maximum(STg - F0, 0) - np.maximum(F0 - STg, 0) + np.maximum(Kwp - STg, 0) + np.maximum(STg - Kwc, 0) + cred
ax.plot(STg / F0 * 100 - 100, pay4 / F0 * 100, color=ORANGE, label="C4 Iron fly (antes do hedge de Δ)")
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("variação do BTC no vencimento (%)"); ax.set_ylabel("P&L (% do nocional)")
ax.set_title("Perfis de payoff no vencimento (30d, superfície de hoje)"); ax.legend(fontsize=7)
save(fig, "s08_payoffs")

# "Miracles" e "Disasters": meses (rolagem a rolagem) com alta > +20% ou queda < -20%
for c in ["BTC", "ETH"]:
    rd = sorted(d for d in SURF[c] if d.day == 1)
    px = DAILY[c].S.reindex(rd).dropna()
    mret = px.pct_change().dropna()
    SUM[f"{c}_miracles"], SUM[f"{c}_disasters"], SUM[f"{c}_nmonths"] = int((mret > 0.20).sum()), int((mret < -0.20).sum()), int(len(mret))
    SUM[f"{c}_best_month"], SUM[f"{c}_worst_month"] = float(mret.max()), float(mret.min())
json.dump(SUM, open(f"{TAB}/resumo.json", "w"), indent=1, default=str)
print(json.dumps(SUM, indent=1, default=str))
