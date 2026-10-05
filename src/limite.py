"""
Limite de requisicoes por IP e por dia.

Por que isso existe: a API fica exposta na internet e cada pergunta custa
uma (ou duas) chamadas ao modelo. Sem limite, qualquer pessoa que ache o
endereco consome a sua cota -- ou o seu dinheiro.

Sao dois limites, com motivos diferentes:
  - POR IP, por minuto: impede que UMA pessoa (ou um script) monopolize.
  - GLOBAL, por dia: protege a cota diaria da camada gratuita do provedor.
    Melhor o site dizer "volte amanha" do que o provedor cortar no meio.
"""

import time


class Limite:
    """Janela deslizante simples, em memoria.

    ponytail: contagem em memoria de UM processo. Se um dia rodar com
    varios workers (uvicorn --workers) ou mais de uma maquina, cada um tera
    a sua contagem e o limite real sera multiplicado. Nesse momento, troque
    por Redis. Enquanto for um processo so, isto basta e nao adiciona
    dependencia nenhuma.
    """

    def __init__(self, por_ip_minuto: int = 6, global_dia: int = 300):
        self.por_ip_minuto = por_ip_minuto
        self.global_dia = global_dia
        self._ips: dict[str, list[float]] = {}
        self._dia: list[float] = []

    def checar(self, ip: str, agora: float | None = None) -> str | None:
        """Registra o acesso e devolve o motivo da recusa, ou None se pode.

        'agora' e injetavel para o teste nao precisar dormir de verdade.
        """
        agora = time.time() if agora is None else agora

        # Cota diaria primeiro: se o dia acabou, nem adianta olhar o IP.
        self._dia = [t for t in self._dia if agora - t < 86400]
        if len(self._dia) >= self.global_dia:
            return (
                "A cota diária de consultas deste site acabou. "
                "Tente novamente amanhã."
            )

        recentes = [t for t in self._ips.get(ip, []) if agora - t < 60]
        if len(recentes) >= self.por_ip_minuto:
            return "Muitas perguntas seguidas. Espere um minuto e tente de novo."

        recentes.append(agora)
        self._ips[ip] = recentes
        self._dia.append(agora)

        # Descarta IPs que ficaram so com registros velhos, senao o
        # dicionario cresce para sempre com quem passou uma vez e sumiu.
        if len(self._ips) > 1000:
            self._ips = {
                k: v for k, v in self._ips.items() if any(agora - t < 60 for t in v)
            }
        return None
