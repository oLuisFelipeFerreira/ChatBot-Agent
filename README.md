# Alura Agent

Agente de IA que responde perguntas em linguagem natural sobre documentos
internos (PDF ou CSV), usando RAG (Retrieval-Augmented Generation) com
LangChain e Claude (Anthropic). Projeto final do curso Alura, implantado
na Oracle Cloud Infrastructure (OCI).

## Objetivo

Reduzir o tempo que colaboradores gastam procurando informação em manuais,
relatórios e planilhas, permitindo que qualquer pessoa pergunte diretamente
ao agente e receba uma resposta em linguagem natural.

## Arquitetura

```
Documento (PDF/CSV)
        |
        v
  Split em chunks  --->  Embeddings (HuggingFace, local)
        |                          |
        |                          v
        |                  Vector store (FAISS)
        |                          |
        v                          v
     Pergunta  ------------->  Retriever (top-k chunks relevantes)
                                    |
                                    v
                          Claude (Anthropic) via LangChain
                                    |
                                    v
                               Resposta final
```

## Como executar localmente

1. Clone o repositório e entre na pasta:
   ```bash
   git clone <url-do-repositorio>
   cd alura-agent
   ```

2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure sua chave de API:
   ```bash
   cp .env.example .env
   # edite o .env e cole sua ANTHROPIC_API_KEY
   ```

4. Coloque seu documento em `data/` (ex: `data/documento.pdf`).

5. Rode o agente:
   ```bash
   python -m src.main data/documento.pdf
   ```

## Exemplos de perguntas e respostas

> _(preencher depois dos testes locais, com pergunta + resposta real do agente)_

- **Pergunta:** ...
  **Resposta:** ...

## Demonstração na nuvem (OCI)

> _(adicionar aqui o link público ou print da aplicação rodando na instância OCI)_

## Estrutura do projeto

```
alura-agent/
├── data/           # documento(s) de referência
├── src/
│   ├── ingest.py   # leitura, chunking, embeddings e vector store
│   ├── agent.py    # chain de recuperação + resposta com Claude
│   └── main.py     # CLI de perguntas e respostas
├── requirements.txt
└── .env.example
```

## Tecnologias

Python, LangChain, HuggingFace Embeddings (local, gratuito), FAISS, Claude
(Anthropic) e Oracle Cloud Infrastructure (OCI Compute) para a implantação.
