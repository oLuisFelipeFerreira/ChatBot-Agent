<div align="center">

# ⚖️ Simples Assim

**Assistente de IA para dúvidas de direito empresarial brasileiro.**
Responde em linguagem natural com base em legislação e documentos oficiais —
sempre indicando a fonte, o arquivo e a página.

[![Python](https://img.shields.io/badge/Python-3.11+-1F6F4A?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1F6F4A?style=flat-square)](https://python.langchain.com/)
[![Claude](https://img.shields.io/badge/Claude-Anthropic-1F6F4A?style=flat-square)](https://www.anthropic.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-1F6F4A?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OCI](https://img.shields.io/badge/Oracle_Cloud-deploy-1F6F4A?style=flat-square&logo=oracle&logoColor=white)](https://www.oracle.com/cloud/)

</div>

---

## O problema

Abrir e manter uma empresa no Brasil significa navegar por leis complementares,
instruções normativas e manuais de registro. A informação existe e é pública —
mas está espalhada em documentos longos, técnicos e difíceis de consultar.

Quem precisa da resposta (o empreendedor) raramente tem o vocabulário jurídico
para encontrá-la. Quem pergunta "qual a diferença entre MEI, ME e EPP?" não sabe
que a lei escreve "consideram-se microempresas ou empresas de pequeno porte...".

**Simples Assim** faz essa ponte: você pergunta em português comum, ele responde
com base no texto oficial e mostra exatamente de onde tirou cada informação.

---

## Demonstração

> _(adicionar capturas de tela aqui)_

<!--
![Tela inicial](docs/tela-inicial.png)
![Resposta com fontes](docs/resposta.png)
-->

**Exemplo real:**

> **Consulta:** Quanto o MEI paga por mês?
>
> O MEI paga um valor fixo mensal, independentemente de faturar ou não naquele
> mês, por meio do DAS. O valor é a soma de: contribuição previdenciária de 5%
> do salário-mínimo; R$ 1,00 de ICMS (se for contribuinte); R$ 5,00 de ISS (se
> for contribuinte). O MEI transportador autônomo de cargas recolhe 12% em vez
> de 5%. O pagamento é devido mesmo com receita zero ou empresa inativa.
>
> **Fontes:** `[Tributário] mei_perguntas_respostas.pdf — páginas 10, 12` ·
> `[Legislação] lc123_simples_nacional.html`

**Teste de ancoragem** — o agente recusa o que está fora da base, em vez de inventar:

> **Consulta:** Qual a capital da França?
>
> Não encontrei essa informação nos documentos disponíveis.

---

## Arquitetura

```
                    ┌──────────────────────────────┐
   Navegador ──────►│  web/   HTML · CSS · JS      │
                    │  interface, sem framework    │
                    └──────────────┬───────────────┘
                                   │  POST /api/perguntar
                    ┌──────────────▼───────────────┐
                    │  api.py   FastAPI            │
                    │  valida, orquestra, serve    │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  src/   o agente (RAG)       │
                    └──────────────────────────────┘
```

### O pipeline de RAG

```
Documentos oficiais (PDF · HTML · TXT)
        │
        ▼
  Extração e limpeza ......... remove cabeçalhos, rodapés e texto revogado
        │
        ▼
  Chunking jurídico .......... divide respeitando "Art." e "§"
        │
        ▼
  Embeddings multilíngues .... paraphrase-multilingual-MiniLM-L12-v2
        │
        ▼
  Índice FAISS ............... busca por similaridade semântica
        │
        ├──────────────┬────────────────┐
        ▼              ▼                ▼
   Expansão de     BM25            Busca
   siglas       (palavra-chave)   semântica
        │              │                │
        └──────────────┴────────────────┘
                       │  EnsembleRetriever (40 / 60)
                       ▼
              Multi-query (opcional)
                       │
                       ▼
              Claude Haiku 4.5 → resposta + fontes
```

### Decisões técnicas

| Escolha | Por quê |
|---|---|
| **Busca híbrida** BM25 + semântica | Texto jurídico exige as duas coisas: significado (*faturamento* ≈ *receita bruta*) e termo exato (*art. 966*, *EPP*, valores) |
| **Embeddings multilíngues** | O modelo padrão em inglês degradava a recuperação em português jurídico |
| **Chunking por artigo** | Separadores `Art.` e `§` mantêm cada dispositivo íntegro, em vez de cortar por contagem de caracteres |
| **Remoção de texto revogado** | Páginas de texto compilado do Planalto exibem a redação antiga *tachada*; sem tratar, o agente citaria limites que não valem mais |
| **Glossário de siglas** | Usuários escrevem "EPP"; a lei escreve "empresa de pequeno porte". A expansão é local e não custa API |
| **Citação via metadados** | A fonte vem da estrutura do índice, não do texto gerado — o modelo não tem como errar a página |
| **FAISS** em vez de banco vetorial gerenciado | Base de ~2.000 chunks roda local, sem servidor e sem custo |
| **FastAPI + front próprio** | Separação real entre cérebro e interface; o mesmo agente serve a API, a web e o terminal |

---

## Confiabilidade

Um assistente jurídico que inventa é pior do que nenhum. As defesas em camadas:

- **Prompt ancorado** — responde apenas com o contexto recuperado e admite quando
  não sabe, com uma frase fixa e verificável.
- **Proibição de analogia** — não pode estender a uma categoria o que o documento
  afirma sobre outra. Corrige o erro clássico de assumir que ME e EPP seguem as
  mesmas regras do MEI.
- **Separação conceitual explícita** — tipo jurídico, porte e regime tributário
  são eixos independentes, e o prompt exige que não sejam confundidos.
- **Notas de vigência** — documento curado na própria base registra onde o texto
  literal da lei está superado por norma posterior (ex.: a extinção da EIRELI).
- **Fontes sempre visíveis** — e ocultas quando não há resposta, para não sugerir
  embasamento inexistente.
- **Resiliência a falhas** — retentativas em três camadas para sobrecarga da API.

---

## Base de conhecimento

Apenas fontes oficiais (`.gov.br`). Textos de lei não são protegidos por direito
autoral (Lei 9.610/1998, art. 8º, IV).

| Categoria | Documentos |
|---|---|
| **Legislação** | Código Civil (Livro II — Do Direito de Empresa) · Lei Complementar 123/2006 |
| **Registro empresarial** | Manuais do DREI (Empresário Individual e Sociedade Limitada) · IN DREI 81/2020 |
| **Tributário** | Perguntas e Respostas MEI e Simei · Manual da DASN-SIMEI (Receita Federal) |
| **Notas de vigência** | Curadoria própria sobre dispositivos superados |

> **Escopo deliberado:** cobre as regras vigentes. Documentos da Reforma
> Tributária (IBS/CBS, em transição de 2026 a 2033) ficaram de fora para não
> misturar regimes — empresas do Simples não têm alteração em 2026.

---

## Como executar

```bash
git clone https://github.com/oLuisFelipeFerreira/alura-agent.git
cd alura-agent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # preencha ANTHROPIC_API_KEY

python baixar_documentos.py        # baixa os documentos oficiais
python -m src.ingest               # constrói o índice

uvicorn api:app --port 8000        # http://localhost:8000
```

### Outras interfaces

```bash
python -m src.main                 # terminal
streamlit run app.py               # interface Streamlit
```

### Ferramentas de diagnóstico

Construídas para depurar a recuperação sem gastar chamadas de API:

```bash
python -m src.inspecionar_indice "sua pergunta"   # compara busca semântica vs híbrida
python -m src.ver_chunks "360.000"                # busca literal dentro do índice
```

A distinção que elas permitem é essencial: se o trecho **está** no índice mas não
é recuperado, o problema é a busca; se **não está**, o problema é a leitura do
documento. Sem isso, a depuração vira adivinhação.

---

## Deploy

Roda em uma instância **OCI Compute** (Ampere A1, Ubuntu 24.04), dentro de uma
VCN com sub-rede pública e Internet Gateway.

```bash
# na instância
git pull && source .venv/bin/activate
pip install -r requirements.txt

nohup uvicorn api:app --host 0.0.0.0 --port 8000 > servidor.log 2>&1 &
```

Portas liberadas na Security List da OCI **e** no firewall interno (`iptables`),
com a regra inserida antes do `REJECT` da cadeia INPUT.

---

## Estrutura

```
.
├── api.py                      # API FastAPI + serve o front
├── app.py                      # interface Streamlit (alternativa)
├── baixar_documentos.py        # baixa a base oficial automaticamente
├── web/                        # front-end
│   ├── index.html
│   ├── estilo.css
│   └── app.js
├── src/
│   ├── ingest.py               # extração, limpeza, chunking e indexação
│   ├── busca.py                # busca híbrida, glossário, multi-query
│   ├── agent.py                # prompt ancorado e cadeia de resposta
│   ├── main.py                 # interface de terminal
│   ├── inspecionar_indice.py   # diagnóstico de recuperação
│   └── ver_chunks.py           # busca literal no índice
├── data/                       # documentos, organizados por categoria
├── requirements.txt
└── .env.example
```

---

## Tecnologias

`Python` · `LangChain` · `Claude (Anthropic)` · `FAISS` · `sentence-transformers`
· `rank_bm25` · `BeautifulSoup` · `pypdf` · `FastAPI` · `Uvicorn` · `Streamlit`
· `Oracle Cloud Infrastructure`

---

## Próximos passos

- [ ] Pipeline de atualização automática quando a legislação mudar
- [ ] Registro de perguntas sem resposta para identificar lacunas na base
- [ ] Avaliação por resposta, alimentando o ciclo de melhoria
- [ ] HTTPS com domínio próprio
- [ ] Serviço `systemd` para reinício automático
- [ ] Ampliação para direito trabalhista e contratos empresariais

---

## Autor

**Luís Felipe Ferreira**

[LinkedIn](https://www.linkedin.com/in/luís-felipe-barbosa-ferreira-33a0a9210) ·
[GitHub](https://github.com/oLuisFelipeFerreira)

Projeto desenvolvido a partir do desafio final do curso de Agentes de IA da
**Alura**, expandido para uso real.

---

> ⚠️ **Conteúdo informativo gerado por IA a partir de documentos oficiais.**
> Não constitui consultoria jurídica ou contábil. Para casos concretos,
> procure um advogado ou contador.
