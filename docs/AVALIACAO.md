# Avaliação

Sem medir, *"melhorou"* é opinião. Ajustar o `TOP_K`, o peso da busca híbrida
ou o modelo vira adivinhação, porque nada diz se o resultado ficou melhor ou
pior.

São **15 perguntas anotadas** — 10 que devem ser respondidas e 5 que devem ser
recusadas — rodando contra a cadeia real:

```bash
python -m avaliacao.rodar
```

| Modelo | `TOP_K` | Acertos | Recusou certo | Fonte certa | Armadilhas | Mediana |
|---|---|---|---|---|---|---|
| `llama3.2:3b` (local) | 10 | 13/15 | 100% | 100% | 50% | 61,8s |
| `llama3.2:3b` (local) | 4 | 12/15 | 100% | 100% | 50% | 25,7s |

> Medido num i5-1135G7 sem GPU, com `temperature=0`. Duas execuções seguidas
> produzem resultado idêntico.

## O que ela mede

| | Pergunta que responde |
|---|---|
| **Recusa** | O sistema admite quando não sabe? |
| **Armadilha** | Evita os valores vencidos que estão no texto literal da lei? |
| **Fonte** | A busca trouxe o documento que de fato contém a resposta? |
| **Conteúdo** | Os números e termos esperados aparecem? |

A terceira é a mais útil: ela **separa falha de busca de falha de modelo**. Se a
fonte certa chegou e a resposta saiu errada, mexer no `TOP_K` ou no glossário é
perda de tempo. Se a fonte não chegou, melhorar o prompt é inútil.

## O que ela encontrou

Construí a avaliação esperando comparar modelos. Ela encontrou problemas no
**sistema** antes de comparar modelo nenhum.

### 1. O diferencial do projeto não funcionava

As notas de vigência são 5 trechos numa base de 2.099 — **0,24%**. Perguntado
*"posso abrir uma EIRELI?"*, o buscador ranqueia por relevância ao termo:

```
40 trechos  in_drei_81.pdf              tratam a EIRELI como opção válida
 7 trechos  manual_sociedade_limitada.pdf
 5 trechos  manual_empresario_individual.pdf
 2 trechos  notas_de_vigencia.txt       dizem que ela foi EXTINTA
```

Dois contra cinquenta e três. As notas perdiam **todas** as vagas, e o agente
respondia com a redação vencida — citando fonte oficial, com total confiança.

A causa é estrutural: **a informação corretiva é abafada pela informação que ela
existe para corrigir**, justamente por ser menor. A correção foi dar autoridade
estrutural ao documento de curadoria, com um buscador dedicado que sempre
contribui e entra primeiro no contexto.

**Acerto de fonte: 78% → 100%.**

### 2. A recusa era detectada por frase exata

O modelo parafraseia. Estas eram contadas como respostas válidas, e a interface
mostrava a lista de fontes junto de um "não sei":

> *"Não encontrei **informações** nos documentos disponíveis sobre abertura de
> empresa em Portugal."*

### 3. A medição era instável

Com a configuração idêntica, uma pergunta passava numa rodada e falhava na
seguinte. `temperature=0` tornou as execuções reproduzíveis — e elevou a recusa
correta de 80% para 100%: o modelo estava "criativamente" respondendo o que
deveria recusar.

### 4. O próprio gabarito tinha um falso positivo

A pergunta sobre MEI/ME/EPP **passava** exigindo apenas que `360.000` e
`4.800.000` aparecessem. Uma resposta que atribuía os R$ 360.000 ao MEI (o certo
são R$ 81.000), inventava limites de empregados e chamava ME e EPP de "tipo
jurídico" — tudo errado, e contrariando a regra 4 do prompt — passava intacta.

**A avaliação mediu o que eu pensei em medir, não o que importava.** O gabarito
foi endurecido com os termos proibidos.

## Limitações conhecidas

- **`R$ 45,65`** — o texto da LC 123 traz esse valor absoluto para a
  contribuição do MEI. É 5% do salário-mínimo **de 2016**, congelado na redação;
  a regra viva é o percentual. Falta uma nota de vigência para esse dispositivo.
- **15 perguntas é pouco.** Cobre as falhas conhecidas, não o domínio.
- **O corte após a recusa é por parágrafo.** Se o modelo recusar e continuar na
  mesma linha, nada é cortado.
- **Só um modelo medido.** A comparação entre provedores é trabalho em aberto.
