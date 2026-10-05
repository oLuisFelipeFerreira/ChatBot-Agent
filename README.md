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
        Modelo de linguagem (trocável) → resposta + fontes
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

### Pré-requisitos

| | |
|---|---|
| **Python 3.11 – 3.13** | `python --version`. No Windows, marque *"Add Python to PATH"* no instalador |
| **Git** | para clonar o repositório |
| **~2 GB livres** | o modelo de embeddings (≈ 500 MB) é baixado na primeira execução |
| **Internet na 1ª vez** | baixa os documentos oficiais e o modelo de embeddings |
| **Uma chave de IA** | Anthropic, Google, Groq, OpenAI — ou **nenhuma**, usando Ollama local (veja [Trocar o modelo de IA](#trocar-o-modelo-de-ia)) |

Não é preciso instalar o `uvicorn` à parte: ele vem no `requirements.txt`.

### Linux / macOS

```bash
git clone https://github.com/oLuisFelipeFerreira/alura-agent.git
cd alura-agent

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env               # edite e preencha a chave do provedor
python testar_api.py               # confere se a chave funciona (opcional, mas recomendado)

python -m src.ingest               # constrói o índice (demora na 1ª vez)

python -m uvicorn api:app --port 8000
```

### Windows (PowerShell)

```powershell
git clone https://github.com/oLuisFelipeFerreira/alura-agent.git
cd alura-agent

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

copy .env.example .env
notepad .env                       # preencha a chave do provedor
python testar_api.py

python -m src.ingest

python -m uvicorn api:app --port 8000
```

> Se o PowerShell bloquear o `Activate.ps1`, rode uma vez:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

Depois abra **http://localhost:8000**.

**O passo que não dá para pular é o `ingest`.** Os documentos de `data/` vêm
junto com o repositório, mas `faiss_index/` está no `.gitignore` porque é
gerado — em uma máquina nova ele não existe. A API até constrói o índice
sozinha ao subir, mas aí você espera minutos olhando um terminal parado, sem
saber se travou. Rode antes e veja o progresso.

`python baixar_documentos.py` é **opcional**: serve para rebaixar a legislação
quando ela mudar. Ele pula arquivos que já existem, então em um clone novo não
faz nada. Para atualizar de verdade, apague o arquivo antigo em `data/`, rode o
script e reindexe.

### Outras interfaces

```bash
python -m src.main                 # terminal
streamlit run app.py               # interface Streamlit
```

### Se algo der errado

| Sintoma | Causa e solução |
|---|---|
| `'uvicorn' não é reconhecido` | O venv não está ativo, ou o script não está no PATH. Use `python -m uvicorn api:app --port 8000` |
| `ModuleNotFoundError: langchain` (ou bs4, fastapi...) | Venv não ativado antes do `pip install`. Ative e rode `pip install -r requirements.txt` de novo |
| `ANTHROPIC_API_KEY nao encontrada` | Falta o `.env` ou a chave está vazia. Ou troque o provedor no `.env` |
| `Chave de API inválida ou sem crédito` | A chave do `.env` expirou, foi digitada errada ou a conta está sem saldo. Rode `python testar_api.py` para confirmar |
| `Falta o pacote do provedor 'x'` | Instale o pacote indicado na mensagem (ex.: `pip install langchain-groq`) |
| `Indice 'faiss_index' nao encontrado` | Rode `python -m src.ingest` |
| Índice vazio / sem fontes | `data/` está vazia — confira se o clone veio completo ou rode `python baixar_documentos.py` |
| Primeira execução muito lenta | Normal: está baixando o modelo de embeddings (≈ 500 MB). Só acontece uma vez |
| Erro 529 / *overloaded* | Sobrecarga temporária do provedor. Ele já tenta de novo sozinho; para economizar chamadas, coloque `USAR_MULTIQUERY = False` em `src/agent.py` |
| Porta 8000 ocupada | `python -m uvicorn api:app --port 8001` |
| Reindexar do zero | Apague a pasta `faiss_index/` e rode `python -m src.ingest` |

---

## Trocar o modelo de IA

O projeto não depende da Anthropic. A geração da resposta é escolhida por
variável de ambiente; **a busca nos documentos roda sempre local e de graça**
(embeddings HuggingFace + FAISS + BM25), então trocar de modelo não exige
reindexar nada.

Edite o `.env`:

| Provedor | `.env` | Pacote | Custo |
|---|---|---|---|
| **Ollama** (local) | `LLM_PROVIDER=ollama`<br>`LLM_MODEL=llama3.1` | `pip install langchain-ollama` | **Grátis, sem chave** |
| **Google Gemini** | `LLM_PROVIDER=google_genai`<br>`LLM_MODEL=gemini-3.8-flash`<br>`GOOGLE_API_KEY=...` | `pip install langchain-google-genai` | Camada gratuita |
| **Groq** | `LLM_PROVIDER=groq`<br>`LLM_MODEL=llama-3.3-70b-versatile`<br>`GROQ_API_KEY=...` | `pip install langchain-groq` | Camada gratuita |
| **Anthropic** (padrão) | `LLM_PROVIDER=anthropic`<br>`LLM_MODEL=claude-haiku-4-5-20251001`<br>`ANTHROPIC_API_KEY=...` | já instalado | Pago |
| **OpenAI** | `LLM_PROVIDER=openai`<br>`LLM_MODEL=gpt-4o-mini`<br>`OPENAI_API_KEY=...` | `pip install langchain-openai` | Pago |

**Sem chave nenhuma, do zero:** instale o [Ollama](https://ollama.com), rode
`ollama pull llama3.1`, `pip install langchain-ollama`, ponha as duas linhas do
Ollama no `.env` e suba a API normalmente.

```bash
python testar_api.py    # confirma qual provedor está ativo e se ele responde
```

> **Sobre a qualidade:** o prompt é rígido de propósito (não inventar, não
> generalizar por analogia, citar artigo). Modelos menores — em especial os
> locais de 7–8B — seguem essas regras com menos consistência. Se a resposta
> ficar solta, vale comparar com um modelo maior antes de suspeitar da base.

Quer um provedor que não está na lista? Qualquer um suportado pelo
`init_chat_model` do LangChain funciona: instale o pacote e ponha o nome dele
em `LLM_PROVIDER`.

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
├── testar_api.py               # checa se o provedor de IA responde
├── test_provedor.py            # teste offline da troca de provedor
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

`Python` · `LangChain` · `Claude · Gemini · Groq · Ollama (trocável)` · `FAISS` · `sentence-transformers`
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
