"""
Teste rapido do modelo de linguagem configurado.

O que este script faz:
1. Le LLM_PROVIDER / LLM_MODEL e a chave correspondente do arquivo .env
2. Faz UMA pergunta simples ao modelo
3. Imprime a resposta

Se aparecer uma resposta, sua chave e seu ambiente estao funcionando.
Se der erro, a mensagem indica o que ajustar (chave errada, pacote do
provedor faltando, sem internet, etc.).

Rode isto ANTES de subir a API: e a forma mais barata de descobrir que o
.env esta errado, sem esperar o indice carregar.

Uso:
    python testar_api.py
"""

from src.agent import criar_llm, PROVEDOR, MODELO

print(f"Provedor: {PROVEDOR} | Modelo: {MODELO}")
print("Enviando uma pergunta de teste...\n")

llm = criar_llm()
resposta = llm.invoke("Responda em uma frase curta: voce esta funcionando?")

print("Resposta do modelo:")
print(resposta.content)
print("\nTudo certo! Sua chave e seu ambiente estao funcionando.")
