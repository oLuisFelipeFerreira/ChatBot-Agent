"""
Teste rápido da API key do Claude (Anthropic).

O que este script faz:
1. Le a variavel ANTHROPIC_API_KEY do arquivo .env
2. Faz UMA pergunta simples ao Claude
3. Imprime a resposta

Se aparecer uma resposta, sua chave e seu ambiente estao funcionando.
Se der erro, a mensagem indica o que ajustar (chave errada, biblioteca
faltando, sem internet, etc.).

Uso:
    python testar_api.py
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Carrega as variaveis do arquivo .env para dentro do programa
load_dotenv()

# Confere se a chave existe antes de tentar usar
if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERRO: nao encontrei a ANTHROPIC_API_KEY.")
    print("Copie o arquivo .env.example para .env e cole sua chave nele.")
    raise SystemExit(1)

# Cria o cliente do modelo. claude-sonnet-5 e o modelo atual da linha Sonnet.
llm = ChatAnthropic(model="claude-sonnet-5")

print("Enviando uma pergunta de teste ao Claude...\n")

# Faz uma unica chamada simples
resposta = llm.invoke("Responda em uma frase curta: voce esta funcionando?")

print("Resposta do Claude:")
print(resposta.content)
print("\nTudo certo! Sua API key e seu ambiente estao funcionando.")
