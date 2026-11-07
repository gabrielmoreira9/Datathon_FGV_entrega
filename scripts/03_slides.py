"""Monta o PDF de 10 slides (16:9) a partir das figuras e tabelas geradas em 02_analise.py."""
import os, json, textwrap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG, TAB = f"{ROOT}/output/figs", f"{ROOT}/output/tables"
OUT = f"{ROOT}/output/Datathon_FGV_Derivativos_Cripto.pdf"
S = json.load(open(f"{TAB}/resumo.json"))
M = {c: pd.read_csv(f"{TAB}/metricas_{c}.csv", index_col=0) for c in ["BTC", "ETH"]}

TEAM = os.environ.get("EQUIPE", "Equipe: [Integrante 1] · [Integrante 2] · [Integrante 3]")
REPO = os.environ.get("REPO", "github.com/[equipe]/cripto-derivativos-fgv")

NAVY, TEAL, ORANGE, GRAY, LIGHT, INK = "#0B2545", "#1B98E0", "#E4572E", "#8D99AE", "#EEF3F8", "#1D2433"
plt.rcParams.update({"font.family": ["Lato", "DejaVu Sans"], "mathtext.fontset": "dejavusans"})
W, H = 13.333, 7.5


def pct(x, d=1): return f"{x*100:.{d}f}%".replace(".", ",")
def num(x, d=2): return f"{x:.{d}f}".replace(".", ",")


def new_slide(n, title, kicker):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.add_artist(Rectangle((0, 0.885), 1, 0.115, transform=fig.transFigure, color=NAVY))
    fig.text(0.03, 0.945, f"{n:02d}", color=TEAL, fontsize=22, fontweight="bold", va="center")
    fig.text(0.075, 0.952, title, color="white", fontsize=21, fontweight="bold", va="center")
    fig.text(0.075, 0.905, kicker, color="#BFD7EA", fontsize=11.5, va="center", style="italic")
    fig.add_artist(Rectangle((0, 0), 1, 0.035, transform=fig.transFigure, color=LIGHT))
    fig.text(0.03, 0.017, "7º Desafio de Dados de Moedas Digitais — FGV EESP · Modelagem de Risco e Precificação de Derivativos de Criptoativos",
             fontsize=8, color=GRAY, va="center")
    fig.text(0.97, 0.017, f"{n}/10", fontsize=8, color=GRAY, va="center", ha="right")
    return fig


def img(fig, name, rect):
    ax = fig.add_axes(rect); ax.imshow(plt.imread(f"{FIG}/{name}.png")); ax.axis("off")
    return ax


def text_block(fig, x, y, items, width=60, size=11, gap=0.012, color=INK, bullet="▪"):
    """Itens: str (bullet) ou ('h', titulo). Retorna y final."""
    lh = size / 72 / H * 1.32
    for it in items:
        if isinstance(it, tuple) and it[0] == "h":
            fig.text(x, y, it[1], fontsize=size + 1.5, fontweight="bold", color=NAVY, va="top"); y -= lh * 1.35; continue
        lines = textwrap.wrap(it, width)
        for i, L in enumerate(lines):
            if i == 0 and bullet:
                fig.text(x, y, bullet, fontsize=size - 2, color=TEAL, va="top")
            fig.text(x + (0.014 if bullet else 0), y, L, fontsize=size, color=color, va="top")
            y -= lh
        y -= gap
    return y


def callout(fig, rect, title, body, color=NAVY, size=11, width=40):
    x, y, w, h = rect
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, fc=LIGHT, ec=color, lw=1.4))
    fig.add_artist(Rectangle((x, y), 0.006, h, transform=fig.transFigure, color=color))
    fig.text(x + 0.015, y + h - 0.018, title, fontsize=size + 1, fontweight="bold", color=color, va="top")
    yy = y + h - 0.018 - (size + 1) / 72 / H * 1.6
    for L in textwrap.wrap(body, width):
        fig.text(x + 0.015, yy, L, fontsize=size, color=INK, va="top"); yy -= size / 72 / H * 1.3


def table(fig, rect, df, size=9.5, hl_rows=(), col_w=None, header_color=NAVY):
    ax = fig.add_axes(rect); ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=list(df.columns), rowLabels=list(df.index), loc="upper center",
                 cellLoc="center", rowLoc="left", colWidths=col_w)
    t.auto_set_font_size(False); t.set_fontsize(size); t.scale(1, 1.45)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#D5DEE8"); cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(header_color); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
        elif r - 1 in hl_rows:
            cell.set_facecolor("#E3F1FB"); cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F7F9FC")
    return t


b, e = M["BTC"], M["ETH"]
C1, C2, C3, C4 = b.index

with PdfPages(OUT) as pdf:
    # ------------------------------------------------------------------ 01 CAPA
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor(NAVY)
    fig.text(0.06, 0.88, "7º DESAFIO DE DADOS DE MOEDAS DIGITAIS · FGV EESP & FGV DIGITAL FINANCE", color=TEAL, fontsize=12, fontweight="bold")
    fig.text(0.06, 0.74, "Modelagem de Risco e Precificação\nde Derivativos de Criptoativos", color="white", fontsize=36, fontweight="bold", va="center", linespacing=1.15)
    fig.text(0.06, 0.615, "Proteger contra os “Disasters” sem perder os “Miracles”: quanto custa o hedge com opções em BTC e ETH?",
             color="white", fontsize=15.5, fontweight="bold")
    fig.text(0.06, 0.565, "Black-76 inverso · Superfície SVI livre de arbitragem · Gregas de 1ª e 2ª ordem · Backtest de 4 carteiras (Deribit 2024–2026)",
             color="#BFD7EA", fontsize=12.5, style="italic")
    NT = f"{(S['BTC_funnel'][0]+S['ETH_funnel'][0])/1e3:.0f}"
    pill = [("Dados reais Deribit", f"{NT} mil negócios de opções, 82 superfícies SVI diárias, 2,8 anos de séries diárias (BTC e ETH)"),
            ("Motor quantitativo", f"{S['svi_slices']} fatias SVI · RMSE mediano {num(S['svi_rmse_med'])} pt de vol · R² {num(S['svi_r2_med'],3)}"),
            ("Resultado-chave (BTC)", f"Collar: vol {pct(b.loc[C1,'Volatilidade'],0)}→{pct(b.loc[C3,'Volatilidade'],0)}, MDD {pct(b.loc[C1,'MDD'],0)}→{pct(b.loc[C3,'MDD'],0)}, CVaR99 −{pct(1-b.loc[C3,'CVaR 99% (1d)']/b.loc[C1,'CVaR 99% (1d)'],0)} com Sharpe {num(b.loc[C3,'Sharpe'])} vs {num(b.loc[C1,'Sharpe'])}")]
    for i, (t, d) in enumerate(pill):
        x = 0.06 + i * 0.305
        fig.add_artist(FancyBboxPatch((x, 0.27), 0.285, 0.24, boxstyle="round,pad=0.005,rounding_size=0.015", transform=fig.transFigure,
                                      fc="#13315C", ec=TEAL if i < 2 else ORANGE, lw=1.6))
        fig.text(x + 0.015, 0.48, t, color=TEAL if i < 2 else ORANGE, fontsize=13, fontweight="bold", va="top")
        yy = 0.425
        for L in textwrap.wrap(d, 40):
            fig.text(x + 0.015, yy, L, color="white", fontsize=12, va="top"); yy -= 0.04
    fig.text(0.06, 0.15, TEAM, color="white", fontsize=13)
    fig.text(0.06, 0.10, f"Código aberto e reprodutível: {REPO}", color="#BFD7EA", fontsize=11)
    fig.text(0.94, 0.10, "1/10", color=GRAY, fontsize=9, ha="right")
    pdf.savefig(fig); plt.close(fig)

    # ------------------------------------------------------------------ 02 PROBLEMA
    fig = new_slide(2, "O problema: risco de cauda e a falha da hipótese gaussiana",
                    "Institucionais devem alocar cripto? Só se conseguirem controlar os “Disasters” — e o seguro tem preço")
    img(fig, "s02_caudas_drawdown_vrp", [0.02, 0.36, 0.96, 0.51])
    y = text_block(fig, 0.04, 0.33, [
        f"Caudas pesadas: curtose em excesso de {num(S['BTC_kurtosis'],1)} (BTC) e {num(S['ETH_kurtosis'],1)} (ETH) nos log-retornos diários; assimetria negativa ({num(S['BTC_skew'])} / {num(S['ETH_skew'])}). Jarque-Bera rejeita normalidade (p < 10⁻¹⁰⁰).",
        f"Drawdowns que paralisam alocadores: {pct(S['BTC_mdd_spot'],0)} no BTC e {pct(S['ETH_mdd_spot'],0)} no ETH no período; vol anualizada de {pct(S['BTC_vol_ann'],0)} e {pct(S['ETH_vol_ann'],0)}.",
    ], width=78, size=11.5)
    callout(fig, [0.62, 0.13, 0.35, 0.2], "Prêmio de risco de variância",
            f"A vol implícita (DVOL) superou a realizada em média {num(S['BTC_vrp_mean'],1)} pts no BTC ({pct(S['BTC_vrp_pos'],0)} dos dias) e {num(S['ETH_vrp_mean'],1)} pts no ETH: proteção tem custo estrutural, e vendê-la é remunerado — fora das crises.", width=50, size=10.5)
    text_block(fig, 0.04, 0.17, ["Mercado 24/7 com cascatas de liquidação alavancada: os saltos invalidam Black-Scholes com vol constante e exigem superfície de volatilidade completa (sorriso + termo)."], width=78, size=11.5)
    pdf.savefig(fig); plt.close(fig)

    # ------------------------------------------------------------------ 03 DERIVATIVOS
    fig = new_slide(3, "Catálogo de derivativos e a mecânica inversa (coin-margined)",
                    "Na Deribit o strike é em USD, mas prêmio e liquidação são em BTC/ETH — o payoff vira hiperbólico")
    img(fig, "s03_payoff_inverso_delta_dual", [0.02, 0.44, 0.96, 0.43])
    df = pd.DataFrame({
        "Mecânica": ["Vencimento fixo, em USD", "Perpétuo; funding 8h", "Prêmio/payoff em moeda", "Prêmio/payoff em moeda"],
        "Payoff / preço": ["F(t,T)", "≈ índice (via funding)", "max(1 − K/S_T, 0)", "max(K/S_T − 1, 0)"],
        "Uso no projeto": ["F do Black-76; r_imp(τ) = ln(F/S)/τ", "Hedge dinâmico de Δ na C4", "Venda 25Δ (C3); asa 10Δ (C4)", "Seguro 20Δ/25Δ (C2, C3); asa (C4)"]},
        index=["Futuros datados", "Perpetual swap", "Call europeia inversa", "Put europeia inversa"])
    table(fig, [0.15, 0.06, 0.50, 0.34], df, size=9.2, col_w=[0.29, 0.27, 0.44])
    callout(fig, [0.67, 0.08, 0.30, 0.31], "Por que importa",
            "Convenção Deribit verificada no snapshot: Black-76 sobre F, r = 0, preço em moeda = V_USD/F (erro mediano 0,00001 BTC). Em moeda o Delta é ajustado ao prêmio, Δ_moeda = Δ_USD − V/F: a put inversa chega a −1,45 (vs −1,0 em USD) e a call tem teto de 1 BTC. Hedge calculado só em USD erra o tamanho da proteção para quem contabiliza em BTC — o motor reporta os dois.", width=44, size=10)
    pdf.savefig(fig); plt.close(fig)

    # ------------------------------------------------------------------ 04 PIPELINE
    fig = new_slide(4, "Pipeline de engenharia de dados e higienização",
                    "Fonte única e auditável: API pública da Deribit (≈ 85% da liquidez global de opções cripto)")
    steps = [("Coleta", "API Deribit: negócios de opções em 41 datas × 2 moedas; cadeia completa, futuros, perpétuo, funding horário, DVOL"),
             ("Janela síncrona", "Seleciona as 4h de maior liquidez do dia (essencial em crises: a vol intradiária varia dezenas de pontos)"),
             ("Forward implícito", "Paridade put-call inversa C − P = 1 − K/F por hora e vencimento → F, base e r_imp(τ)"),
             ("Filtros", "τ ≥ 2 dias, IV ∈ (5%, 300%), OTM (put k<0 / call k≥0), |k| < 1,2; agregação por instrumento"),
             ("Saída", "Cadeias diárias + superfícies SVI (pickle), séries diárias S, DVOL, funding (CSV)")]
    for i, (t, d) in enumerate(steps):
        x = 0.03 + i * 0.193
        fig.add_artist(FancyBboxPatch((x, 0.66), 0.178, 0.19, boxstyle="round,pad=0.004,rounding_size=0.01", transform=fig.transFigure,
                                      fc=LIGHT, ec=NAVY, lw=1.2))
        fig.text(x + 0.01, 0.835, f"{i+1}. {t}", fontsize=12, fontweight="bold", color=NAVY, va="top")
        yy = 0.795
        for L in textwrap.wrap(d, 30):
            fig.text(x + 0.01, yy, L, fontsize=9.2, color=INK, va="top"); yy -= 0.027
        if i < 4: fig.text(x + 0.181, 0.755, "➜", fontsize=13, color=TEAL, va="center")
    img(fig, "s04_funil_estrutura_termo", [0.02, 0.09, 0.96, 0.55])
    fig.text(0.04, 0.06, f"Taxa implícita 10–110d via paridade: média {pct(S['BTC_rimp_hist_mean'])} a.a. (BTC) e {pct(S['ETH_rimp_hist_mean'])} (ETH), "
             f"variando de {pct(S['BTC_rimp_hist_min'])} a {pct(S['BTC_rimp_hist_max'])} no BTC — usar uma r fixa em Black-Scholes distorce o forward.", fontsize=10.5, color=INK)
    pdf.savefig(fig); plt.close(fig)

    # ------------------------------------------------------------------ 05 MOTOR
    fig = new_slide(5, "Motor Black-76 e parametrização Raw SVI",
                    "Precificação sobre o forward, extração própria da vol implícita e sorriso livre de arbitragem estática")
    eqs = [r"$C=e^{-r\tau}[F\,N(d_1)-K\,N(d_2)]\quad P=e^{-r\tau}[K\,N(-d_2)-F\,N(-d_1)]\quad d_{1,2}=\frac{\ln(F/K)\pm\frac{1}{2}\sigma^2\tau}{\sigma\sqrt{\tau}}$",
           r"$\sigma_{k+1}=\sigma_k-\frac{V_{B76}(\sigma_k)-V_{mkt}}{\nu(\sigma_k)}$  (Newton)  $\rightarrow$  Brent em $[0{,}05;\,5]$ se não convergir",
           r"$w(k)=a+b\,[\rho\,(k-m)+\sqrt{(k-m)^2+s^2}\,],\quad k=\ln(K/F),\quad w=\sigma^2\tau$",
           r"$b(1+|\rho|)\leq 2\ (\mathrm{Lee}),\quad g(k)\geq 0\ (\mathrm{butterfly}),\quad \partial_\tau w\geq 0\ (\mathrm{calend\acute{a}rio})$"]
    fig.add_artist(FancyBboxPatch((0.03, 0.62), 0.94, 0.235, boxstyle="round,pad=0.004,rounding_size=0.01", transform=fig.transFigure, fc=LIGHT, ec=LIGHT))
    for i, eq in enumerate(eqs):
        fig.text(0.045, 0.83 - i * 0.055, eq, fontsize=13.5, color=NAVY, va="top")
    img(fig, "s05_svi_fit_auditoria", [0.02, 0.17, 0.96, 0.44])
    fig.text(0.04, 0.135, "Calibração: scipy.optimize.least_squares ponderado por Vega, sequencial da maturidade curta para a longa, com penalidades de não-arbitragem.", fontsize=10.5, color=INK)
    kp = [(f"{S['svi_slices']}", "fatias SVI calibradas"), (f"{num(S['svi_rmse_med_calm'])} / {num(S['svi_rmse_med_crisis'])}", "RMSE mediano (pts) calma / crise"),
          (f"{num(S['svi_r2_med'],3)}", "R² mediano"), (f"{pct(S['svi_g_ok'],1)}", "fatias com g(k) ≥ 0 na faixa cotada"),
          (f"{pct(S['svi_calendar_ok'],1)}", "pares sem arbitragem de calendário"), (f"{num(S['iv_audit_med_abs'])} pt", "|IV própria − IV Deribit| mediano")]
    for i, (v_, l_) in enumerate(kp):
        x = 0.04 + i * 0.157
        fig.text(x, 0.095, v_, fontsize=17, fontweight="bold", color=TEAL, va="top")
        fig.text(x, 0.055, l_, fontsize=8.8, color=INK, va="top")
    pdf.savefig(fig); plt.close(fig)

    # ------------------------------------------------------------------ 06 SUPERFICIE
    fig = new_slide(6, "Superfície de volatilidade 3D e dinâmica do skew",
                    "Mesma moeda, dois regimes: o pânico eleva o nível e entorta a asa das puts")
    img(fig, "s06_superficies_skew", [0.01, 0.33, 0.98, 0.55])
    img(fig, "s06_densidade_bl", [0.60, 0.05, 0.38, 0.28])
    text_block(fig, 0.04, 0.31, [
        f"Nível: a vol ATM 30d do BTC saltou de {pct(S['BTC_2025-09-18_atm'],0)} (18/09/25) para {pct(S['BTC_2024-08-05_atm'],0)} no crash de 05/08/24; o RR 25Δ foi de {num(S['BTC_2025-09-18_rr']*100,1)} para {num(S['BTC_2024-08-05_rr']*100,1)} pts.",
        f"Inclinação: o skew 25Δ ficou negativo (puts mais caras) em {pct(S['rr_neg_share'],0)} das 82 datas — a 'demanda por seguro' domina desde 2025, invertendo o padrão de 'call skew' de 2024.",
        f"Poder preditivo: correlação de Spearman entre o RR na rolagem e o retorno dos 30 dias seguintes ρ = {num(S['skew_ret30_rho'])} (p = {num(S['skew_ret30_p'])}, n = {S['skew_n']}); com o drawdown, ρ = {num(S['skew_mdd30_rho'])} (p = {num(S['skew_mdd30_p'])}). O skew descreve o medo, mas não antecipa quedas de forma significativa.",
    ], width=95, size=10.5)
    pdf.savefig(fig); plt.close(fig)

    # ------------------------------------------------------------------ 07 GREGAS
    fig = new_slide(7, "Mapeamento analítico das Gregas e decomposição de Taylor",
                    r"ΔΠ ≈ Δ·ΔS + ½Γ·ΔS² + ν·Δσ + Θ·Δt + Vanna·ΔS·Δσ + ½Volga·Δσ²  —  fórmulas fechadas de Black-76, validadas por diferenças finitas")
    img(fig, "s07_gregas_btc", [0.02, 0.60, 0.96, 0.27])
    img(fig, "s07_taylor_stress", [0.02, 0.11, 0.96, 0.48])
    sh = S["taylor_share"]
    fig.text(0.04, 0.075, f"C4 (BTC): a expansão de 2ª ordem explica R² = {num(S['taylor_r2'])} do P&L diário. Acumulado (% NAV inicial): Theta {pct(sh['Theta'],0)}, Gamma {pct(sh['Gamma'],0)}, "
             f"Vanna+Volga {pct(sh['Vanna+Volga'],0)}, Vega {pct(sh['Vega'],0)}, custos {pct(sh['Custos'],0)}: carry de Theta pago com Gamma.", fontsize=10.2, color=INK)
    fig.text(0.04, 0.048, f"Stress (direita): com queda instantânea de 40% no BTC e +40% na vol, o Collar de hoje perde {pct(-S['stress_collar_-40']/100,0)} do NAV contra 40% do spot.", fontsize=10.2, color=INK)
    pdf.savefig(fig); plt.close(fig)

    # ------------------------------------------------------------------ 08 CARTEIRAS
    fig = new_slide(8, "Engenharia das 4 carteiras e regras do backtest",
                    "Rolagem mensal no 1º dia do mês, carteira autofinanciada (nocional = NAV/S), marcação diária na superfície")
    rules = [("C1 Spot", GRAY, "100% comprado no ativo (benchmark)."),
             ("C2 Protective Put", TEAL, "Spot + put 20Δ com vencimento na próxima rolagem (~30d)."),
             ("C3 Zero-Cost Collar", NAVY, "Spot + put 25Δ − call com K₂ tal que C(K₂) = P(K₁): prêmio líquido zero."),
             ("C4 Delta-Gamma Hedge", ORANGE, "Caixa em USD + venda de straddle ATM + compra de asas 10Δ (gamma limitado); Δ zerado via perpétuo quando |Δ| > 0,10·nocional.")]
    y = 0.84
    for t, c, d in rules:
        fig.add_artist(Rectangle((0.035, y - 0.075), 0.006, 0.07, transform=fig.transFigure, color=c))
        fig.text(0.05, y, t, fontsize=13, fontweight="bold", color=c, va="top")
        yy = y - 0.035
        for L in textwrap.wrap(d, 62):
            fig.text(0.05, yy, L, fontsize=10.5, color=INK, va="top"); yy -= 0.028
        y -= 0.115
    img(fig, "s08_payoffs", [0.50, 0.38, 0.48, 0.48])
    callout(fig, [0.035, 0.12, 0.44, 0.24], "Marcação a mercado e custos (realistas)",
            f"Na rolagem: superfície SVI calibrada nos negócios do dia. Entre rolagens: σ(k,τ,t) = σ_SVI(k,τ)·DVOL_t/DVOL_rolagem (sticky-moneyness ancorado no DVOL diário). F = S·e^(r_imp·τ). Custos: taxa de opção 0,03% do subjacente (teto 12,5% do prêmio) + meio spread bid-ask medido no livro ({pct(S['BTC_half_spread'])} do prêmio no BTC), entrega 0,015%, perpétuo taker 0,05% e funding horário real. rf = 4% a.a.", width=70, size=10.3)
    callout(fig, [0.50, 0.17, 0.47, 0.19], "Bateria de 10 indicadores",
            "Retorno anualizado, volatilidade, Sharpe, Sortino, Calmar, Alpha de Jensen e Beta contra o spot, Max Drawdown, VaR 99% e CVaR 99% (Expected Shortfall) históricos de 1 dia. Período: 01/02/2024 a 05/11/2026 (34 rolagens mensais, 1.009 dias), BTC e ETH.", width=74, size=10.5, color=TEAL)
    pdf.savefig(fig); plt.close(fig)

    # ------------------------------------------------------------------ 09 RESULTADOS
    fig = new_slide(9, "Resultados empíricos: Sharpe, Alpha e mitigação de cauda",
                    "Líquido de custos · 01/02/2024 – 05/11/2026 · proteger dos “Disasters” funciona; abrir mão dos “Miracles” custa caro")
    img(fig, "s09_equity_underwater", [0.01, 0.40, 0.60, 0.48])
    cols = ["Retorno a.a.", "Volatilidade", "Sharpe", "Sortino", "Calmar", "Alpha (Jensen)", "MDD", "CVaR 99% (1d)"]
    short = ["Ret.", "Vol", "Sharpe", "Sortino", "Calmar", "α", "MDD", "CVaR99"]
    for j, (c, mm) in enumerate([("BTC", b), ("ETH", e)]):
        d = mm[cols].copy(); d.columns = short
        for k in ["Ret.", "Vol", "α", "MDD", "CVaR99"]: d[k] = d[k].map(lambda x: pct(x, 1))
        for k in ["Sharpe", "Sortino", "Calmar"]: d[k] = d[k].map(lambda x: num(x, 2))
        d.index = ["C1 Spot", "C2 Put 20Δ", "C3 Collar", "C4 ΔΓ-Hedge"]
        fig.text(0.625, 0.855 - j * 0.24, c, fontsize=13, fontweight="bold", color=NAVY)
        table(fig, [0.675, 0.64 - j * 0.24, 0.31, 0.2], d, size=8.6)
    items = [("h", "Leitura"),
             f"BTC — o Collar entrega Sharpe de {num(b.loc[C3,'Sharpe'])} (spot {num(b.loc[C1,'Sharpe'])}) com metade da vol, MDD de {pct(b.loc[C3,'MDD'],0)} (spot {pct(b.loc[C1,'MDD'],0)}) e CVaR 99% {pct(1-b.loc[C3,'CVaR 99% (1d)']/b.loc[C1,'CVaR 99% (1d)'],0)} menor: a melhor relação risco-retorno defensiva.",
             f"ETH — a Put 20Δ gera alpha de {pct(e.loc[C2,'Alpha (Jensen)'],1)} a.a. e reduz o CVaR em {pct(1-e.loc[C2,'CVaR 99% (1d)']/e.loc[C1,'CVaR 99% (1d)'],0)}; o Collar destrói valor ({pct(e.loc[C3,'Retorno a.a.'],0)} a.a.) porque vende os “Miracles”: o ETH teve {S['ETH_miracles']} meses acima de +20% em {S['ETH_nmonths']} (BTC: {S['BTC_miracles']}).",
             f"C4 — neutro em direção (β ≈ {num(b.loc[C4,'Beta'])}), captura o prêmio de variância: {pct(b.loc[C4,'Retorno a.a. sem custos'],1)} a.a. bruto no BTC, mas custos de {pct(b.loc[C4,'Custos totais (% NAV ini.)'],0)} do NAV reduzem para {pct(b.loc[C4,'Retorno a.a.'],1)}; no ETH (VRP menor) não paga os custos."]
    text_block(fig, 0.03, 0.37, items, width=140, size=11.2, gap=0.01)
    pdf.savefig(fig); plt.close(fig)

    # ------------------------------------------------------------------ 10 CONCLUSAO
    fig = new_slide(10, "Conclusões, aplicabilidade institucional e código aberto",
                    "Infraestrutura quantitativa completa, reprodutível e honesta sobre o custo da proteção")
    pillars = [("Inovação metodológica", TEAL,
                ["Black-76 sobre o forward implícito por paridade put-call inversa, sem r arbitrária",
                 "Raw SVI sequencial com penalização de butterfly, Lee e calendário + janela síncrona de liquidez",
                 "Gregas duais (USD × moeda) e atribuição de P&L de 2ª ordem (Vanna/Volga)"]),
               ("Robustez empírica", NAVY,
                [f"{(S['BTC_funnel'][0]+S['ETH_funnel'][0])/1e3:.0f} mil negócios de opções, {S['svi_slices']} fatias SVI, BTC e ETH, regimes de calma e 4 crises",
                 "Custos reais: taxas, spread do livro, funding horário",
                 f"Achado central: não existe hedge universal. Cortar 'Disasters' funciona nas duas moedas; vender os 'Miracles' (collar) só compensa onde eles são raros — BTC ({S['BTC_miracles']} meses > +20%) sim, ETH ({S['ETH_miracles']}) não"]),
               ("Infraestrutura open-source", ORANGE,
                ["Biblioteca Python vetorizada (numpy/scipy/pandas): black76, svi, data, backtest",
                 "3 scripts reprodutíveis: coleta → calibração → análise/slides",
                 f"Repositório: {REPO}"])]
    for i, (t, c, its) in enumerate(pillars):
        x = 0.03 + i * 0.322
        fig.add_artist(FancyBboxPatch((x, 0.40), 0.30, 0.43, boxstyle="round,pad=0.004,rounding_size=0.012", transform=fig.transFigure, fc=LIGHT, ec=c, lw=1.8))
        fig.add_artist(Rectangle((x, 0.78), 0.30, 0.05, transform=fig.transFigure, color=c))
        fig.text(x + 0.015, 0.805, t, fontsize=14, fontweight="bold", color="white", va="center")
        text_block(fig, x + 0.012, 0.755, its, width=40, size=12, gap=0.018)
    text_block(fig, 0.03, 0.31, [("h", "Aplicações no Brasil"),
        "Fundos multimercado e de cripto (CVM 175): política de hedge com orçamento de CVaR e regra de rolagem auditável.",
        "ETFs e COEs estruturados na B3: precificação do 'capital protegido' com superfície SVI em vez de vol fixa.",
        "Protocolos DeFi de opções: marcação livre de arbitragem e monitoramento de Gregas em tempo real."], width=95, size=11.2, gap=0.008)
    callout(fig, [0.66, 0.08, 0.31, 0.27], "Limitações e próximos passos",
            "Marcação intra-mês via escala DVOL (não toda a superfície diária); amostra de 34 rolagens mensais; spread estimado no livro atual. Próximo: superfícies diárias completas, modelos de salto (Bates/SVJ) e otimização de hedge sob restrição de CVaR.", width=46, size=10.4, color=ORANGE)
    fig.text(0.03, 0.045, "Créditos: implementação de código e análise com apoio de assistente de IA (Claude, Anthropic); concepção, revisão e conclusões da equipe.",
             fontsize=8.5, color=GRAY, style="italic")
    pdf.savefig(fig); plt.close(fig)
print("PDF:", OUT)
