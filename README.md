# Modelagem de Risco e Precificação de Derivativos de Criptoativos

**7º Desafio de Dados (Datathon) de Moedas Digitais — FGV EESP & FGV Digital Finance**

Motor quantitativo em Python para opções inversas (coin-margined) de BTC e ETH na Deribit:
Black-76 sobre o forward implícito, extração própria de vol implícita (Newton-Raphson + Brent),
superfície Raw SVI com penalização de arbitragem estática, vetor de Gregas de 1ª e 2ª ordem,
atribuição de P&L por Série de Taylor e backtest de 4 carteiras com custos reais.

**Pergunta:** proteger contra os "Disasters" sem perder os "Miracles" — quanto custa o hedge com opções em BTC e ETH?
**Achado central:** não existe hedge universal. Cortar quedas extremas funciona nas duas moedas; vender a cauda direita
(collar) só compensa onde altas extremas são raras — BTC teve 3 meses > +20% em 33 (collar vence), ETH teve 7 (put vence).

## Resultados (01/02/2024 – 05/11/2026, custos estimados)

Os custos usam taxas oficiais, funding observado e um meio-spread mediano medido no snapshot de 05/11/2026 aplicado ao histórico. Portanto, os resultados líquidos são estimativas, não uma reconstrução de execução histórica. A tabela de sensibilidade em `output/tables/sensibilidade_custos.csv` mostra os cenários de 50%, 100% (base) e 150% dos custos modelados.

| BTC | Ret. a.a. | Vol | Sharpe | MDD | CVaR 99% (1d) |
|---|---|---|---|---|---|
| C1 Spot | 22,8% | 45,7% | 0,36 | −53,0% | 9,2% |
| C2 Protective Put 20Δ | 15,2% | 37,7% | 0,27 | −44,1% | 6,6% |
| C3 Zero-Cost Collar | 12,2% | 22,3% | 0,33 | −29,6% | 4,6% |
| C4 Short Iron Butterfly + Hedge de Delta | 5,4% | 8,2% | 0,16 | −6,9% | 2,4% |

| ETH | Ret. a.a. | Vol | Sharpe | MDD | CVaR 99% (1d) |
|---|---|---|---|---|---|
| C1 Spot | −1,0% | 66,4% | −0,08 | −66,9% | 13,9% |
| C2 Protective Put 20Δ | 4,4% | 52,5% | 0,01 | −58,2% | 8,6% |
| C3 Zero-Cost Collar | −12,1% | 29,6% | −0,57 | −51,0% | 6,3% |
| C4 Short Iron Butterfly + Hedge de Delta | −7,9% | 11,9% | −1,02 | −23,2% | 4,1% |

Tabelas completas (Sortino, Calmar, Alpha de Jensen, Beta, VaR, custos) em `output/tables/metricas_*.csv`.

## Estrutura

```
src/quant/
  black76.py   Modelos 1, 2 e 4: Black-76 (USD e inverso), IV híbrida, Gregas (Δ, Γ, ν, Θ, ρ, Vanna, Volga), delta em moeda
  svi.py       Modelo 3: Raw SVI, g(k) de Gatheral-Jacquier, condição de Lee, calendário, superfície interpolada
  data.py      Pipeline: parse de instrumentos, forward por paridade put-call inversa, r_imp(τ), cadeias diárias
  backtest.py  Simulador das 4 carteiras e bateria de métricas
scripts/
  coletar_dados_deribit.py / coletar_dados_parte2.py   coleta via API pública Deribit (sem chave)
  01_calibrar_superficies.py   calibra superfícies SVI diárias (BTC e ETH) -> data/processed/surfaces
  02_analise.py                backtests, métricas, figuras e output/tables/resumo.json
  03_slides.py                 gera output/Datathon_FGV_Derivativos_Cripto.pdf (10 slides)
data/raw/          dados brutos da Deribit (CSV.gz)
output/figs, output/tables, output/*.pdf
```

## Reproduzir

```bash
pip install numpy scipy pandas matplotlib requests
python scripts/coletar_dados_deribit.py     # coleta dados-base via API pública
python scripts/coletar_dados_parte2.py      # (mova as saídas para data/raw/)
python scripts/04_coletar_trades_diarios.py # amplia trades de opções para todos os dias
python scripts/01_calibrar_superficies.py   # ~20 min, idempotente (pode ser interrompido e retomado)
python scripts/02_analise.py                # ~2 min
python scripts/05_comparar_svi_diaria.py    # compara marcação por DVOL e SVI diária; skew diário
EQUIPE="Equipe: Nome 1 · Nome 2 · Nome 3" REPO="github.com/usuario/repo" python scripts/03_slides.py
```

Para recalcular do zero, apague `data/processed/` (as superfícies e os backtests ficam em cache).

## Dados

API pública da Deribit (`history.deribit.com` e `www.deribit.com/api/v2/public`), BTC e ETH. Os arquivos brutos e caches processados não são versionados; execute os coletores para reproduzir a análise:

- negócios de opções em todos os dias de 01/02/2024 a 07/11/2026: 1.011 arquivos diários por moeda e 20,5 milhões de negócios no total;
- snapshot da cadeia completa em 05/11/2026 (mark_iv, bid/ask, OI) e futuros datados;
- perpétuo diário, funding horário e índice DVOL desde jan/2024.

Para ampliar a amostra de opções, `scripts/04_coletar_trades_diarios.py` baixa os
negócios de BTC e ETH por dia, de 01/02/2024 até a data atual UTC. Os arquivos
seguem o padrão `data/raw/{BTC,ETH}_trades_YYYYMMDD.csv.gz`, usado pelo pipeline.
A coleta pode ser interrompida e retomada: dias com arquivo já existente são
ignorados. Também aceita `--start`, `--end` e `--coins BTC` para limitar o período
ou a moeda. Dias sem negócios não criam arquivo. A calibração SVI e o backtest
devem ser executados separadamente depois da coleta.

## Decisões metodológicas

- **Convenção inversa.** O preço em moeda é V_USD/F, com Black-76 e r = 0. Isso foi verificado no snapshot (erro mediano de 1e-5 BTC).
- **Forward.** Sai da paridade put-call inversa C − P = 1 − K/F, calculada por hora e vencimento. Daí vem r_imp(τ) = ln(F/S)/τ.
- **Janela síncrona.** O SVI é calibrado na janela de 4h com mais negócios do dia, porque em crises a vol intradiária varia dezenas de pontos.
- **Calibração SVI.** Mínimos quadrados ponderados por Vega, da maturidade curta para a longa, com penalidades para:
  - Lee: b(1+|ρ|) ≤ 2;
  - butterfly: g(k) ≥ 0;
  - calendário: w(k,τ) não decrescente.
  
  Fatias mal ajustadas não são usadas como piso de calendário.
- **Backtest principal.** Rolagem mensal com carteira autofinanciada (nocional = NAV/S). Entre rolagens, a marcação usa σ_SVI·DVOL_t/DVOL_rolagem. O script `05_comparar_svi_diaria.py` repete a marcação com superfícies SVI diárias mantendo as mesmas rolagens.
- **C4.** Short iron butterfly (straddle ATM vendido e asas 10Δ compradas), com hedge dinâmico de delta via perpétuo. É uma estratégia short gamma; o hedge de delta não elimina gamma.
- **Custos.**
  - taxa de opção de 0,03% do subjacente, com teto de 12,5% do prêmio;
- meio spread bid-ask medido no snapshot atual e aplicado como proxy ao histórico; a tabela de sensibilidade varia os custos totais modelados;
  - taxa de entrega de 0,015%;
  - perpétuo taker de 0,05%;
  - funding horário real;
  - rf = 4% a.a.

## Créditos

Implementação de código e análise com apoio de assistente de IA (Claude, Anthropic); concepção, revisão e conclusões da equipe.

## Limitações

- O backtest principal (`02_analise.py`) mantém a marcação intra-mês escalada pelo DVOL. `05_comparar_svi_diaria.py` compara esse baseline com a marcação pela superfície SVI diária, mantendo as rolagens mensais.
- O backtest principal continua com 34 rolagens mensais; a nova comparação não altera essa frequência.
- O spread vem do livro atual, pois não há snapshots históricos de bid/ask.
- O teste original do skew 25Δ usa as datas de rolagem (Spearman ρ ≈ −0,10; p ≈ 0,44). A análise diária adicional mantém o horizonte de 30 dias e usa bootstrap em blocos de 30 dias para refletir a sobreposição das janelas.
