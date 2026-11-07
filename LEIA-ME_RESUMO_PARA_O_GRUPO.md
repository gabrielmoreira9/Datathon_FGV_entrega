# Datathon FGV: resumo para o grupo

## O que fizemos

Testamos se vale a pena **proteger uma carteira de Bitcoin e Ethereum usando opções**, com dados reais da corretora Deribit de fev/2024 a nov/2026.

Comparamos 4 carteiras:

1. **Só a moeda (Spot):** comprar e segurar, sem proteção.
2. **Put:** comprar todo mês um "seguro" contra queda.
3. **Collar:** comprar o seguro contra queda e pagar por ele abrindo mão de parte da alta (custo zero).
4. **Hedge neutro (Delta-Gamma):** vender volatilidade e neutralizar a direção do preço todos os dias.

Todos os resultados já incluem custos reais de operação: taxas, spread e funding.

## Conclusão em uma frase

**Proteger contra quedas fortes ("Disasters") funciona, mas abrir mão das altas fortes ("Miracles") pode sair caro. Não existe um hedge que sirva para tudo.**

## Os números principais

| | BTC | ETH |
|---|---|---|
| Pior queda sem proteção | −53% | −67% |
| Melhor estratégia de proteção | **Collar** | **Put** |
| Pior queda com essa proteção | −30% | −58% |
| Retorno ao ano: sem proteção / com proteção | 22,8% / 12,2% | −1,0% / +4,4% |

- **BTC:** o Collar cortou a volatilidade pela metade (46% → 22%) e quase não piorou o retorno ajustado ao risco (Sharpe 0,33 contra 0,36).
- **ETH:** o Collar perdeu dinheiro (−12% ao ano). O ETH teve **7 meses com alta acima de 20%** (o BTC teve 3), e o Collar abre mão justamente dessas altas.
- **Hedge neutro:** ganhou 11,8% ao ano no BTC antes dos custos, mas só 5,4% depois. Os custos comem mais da metade.

## Por que o trabalho é forte

- Os temas batem com a lista da própria organização: carteiras, retornos de cauda ("Miracles e Disasters"), hedge e derivativos.
- Usamos dados reais da Deribit, a fonte que o regulamento sugere.
- O código é aberto e reproduzível, o que o regulamento recomenda.
- A conclusão é honesta e tem nuance, em vez de dizer só "o hedge venceu".

## O que falta fazer (ANTES de enviar)

1. **Colocar os nomes da equipe** no slide 1. Hoje está escrito "[Integrante 1]...".
2. **Criar o repositório no GitHub**, subir a pasta e colocar o link nos slides 1 e 10.
3. Revisar o PDF juntos e ensaiar a defesa oral, marcada para 16/11.
4. **Enviar o PDF pelo Forms até domingo, 08/11, às 23:59:** https://forms.gle/gpF8ZbTd5QM1rGmW9

Para atualizar nomes e link no PDF:

- se alguém do grupo usa Python, rode:
  `EQUIPE="Equipe: Nome 1 · Nome 2 · Nome 3" REPO="github.com/usuario/repo" python scripts/03_slides.py`
- se ninguém usa, peça para a Ana regerar.

## Importante: transparência

O slide 10 tem uma nota de crédito dizendo que **o código e a análise tiveram apoio de assistente de IA (Claude)**. O regulamento permite mentoria desde que seja creditada. Se a banca perguntar, respondam com tranquilidade. Por isso, **todos precisam entender o raciocínio** antes da defesa.

## Perguntas que a banca provavelmente vai fazer

- **"O período não é curto?"** Sim, são menos de 3 anos. Os resultados são indicativos, não definitivos (está nas limitações, slide 10).
- **"Como vocês precificaram as opções entre um mês e outro?"** Usamos a superfície real do dia da rolagem, ajustada diariamente pelo índice de volatilidade DVOL. É uma simplificação, e ela está declarada no slide 10.
- **"O que há de novo?"** Aplicamos tudo a opções "inversas" (pagas em BTC/ETH), com dados e custos reais. Mostramos que o hedge ideal depende de quantos "Miracles" o ativo tem.

## O que tem no zip

| Arquivo | Para que serve |
|---|---|
| `Datathon_FGV_Derivativos_Cripto.pdf` | **O arquivo que vamos enviar** (10 slides) |
| `LEIA-ME_RESUMO_PARA_O_GRUPO.md` | Este resumo |
| `README.md` | Documentação técnica para o GitHub |
| `src/` e `scripts/` | Código Python |
| `data/raw/` | Dados baixados da Deribit |
| `output/figs` e `output/tables` | Gráficos e tabelas de resultados |
