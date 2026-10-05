/* =====================================================================
   Assistente Empresarial - logica do front
   Conversa com a API (api.py) e desenha cada resposta como um parecer.
   ===================================================================== */

const formulario = document.getElementById("formulario");
const campo      = document.getElementById("pergunta");
const botao      = document.getElementById("enviar");
const respostas  = document.getElementById("respostas");
const sugestoes  = document.getElementById("sugestoes");
const abertura   = document.getElementById("abertura");
const modelo     = document.getElementById("modelo-parecer");

let ocupado = false;

/* --- Data da base ---------------------------------------------------- */

fetch("/api/info")
  .then((r) => r.json())
  .then((d) => {
    document.getElementById("vigencia").textContent = d.base_atualizada_em;
  })
  .catch(() => {
    document.querySelector(".vigencia")?.remove();
  });

/* --- Campo que cresce conforme o texto ------------------------------- */

function ajustarAltura() {
  campo.style.height = "auto";
  campo.style.height = campo.scrollHeight + "px";
}
campo.addEventListener("input", ajustarAltura);

// Enter envia; Shift+Enter quebra linha
campo.addEventListener("keydown", (evento) => {
  if (evento.key === "Enter" && !evento.shiftKey) {
    evento.preventDefault();
    formulario.requestSubmit();
  }
});

/* --- Sugestoes ------------------------------------------------------- */

sugestoes.addEventListener("click", (evento) => {
  const botaoSugestao = evento.target.closest(".sugestao");
  if (!botaoSugestao || ocupado) return;
  campo.value = botaoSugestao.textContent.trim();
  ajustarAltura();
  formulario.requestSubmit();
});

/* --- Envio ----------------------------------------------------------- */

formulario.addEventListener("submit", (evento) => {
  evento.preventDefault();
  const pergunta = campo.value.trim();
  if (pergunta.length < 3 || ocupado) return;
  consultar(pergunta);
});

async function consultar(pergunta) {
  ocupado = true;
  botao.disabled = true;

  // A abertura e as sugestoes saem de cena na primeira consulta:
  // a partir daqui a tela pertence as respostas.
  abertura.hidden = true;
  sugestoes.hidden = true;

  const parecer = criarParecer(pergunta);
  const corpo = parecer.querySelector(".corpo");
  corpo.innerHTML =
    '<p class="pensando"><span class="pulso"></span>Consultando os documentos…</p>';
  parecer.scrollIntoView({ behavior: "smooth", block: "start" });

  campo.value = "";
  ajustarAltura();

  try {
    const dados = await consultarComRetentativa(pergunta, corpo);
    corpo.innerHTML = formatarTexto(dados.resposta);
    mostrarFontes(parecer, dados.fontes);
  } catch (erro) {
    corpo.innerHTML =
      '<p class="falha">' + escapar(erro.message) + "</p>";
  } finally {
    ocupado = false;
    botao.disabled = false;
    campo.focus();
  }
}

/**
 * Envia a pergunta e, se o servico estiver congestionado (503), tenta de
 * novo sozinho -- esperando um pouco mais a cada tentativa. Sobrecarga da
 * API e passageira; nao faz sentido desistir na primeira negativa.
 */
async function consultarComRetentativa(pergunta, corpo, tentativas = 3) {
  for (let tentativa = 1; tentativa <= tentativas; tentativa++) {
    const resposta = await fetch("/api/perguntar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pergunta }),
    });

    if (resposta.ok) return resposta.json();

    const erro = await resposta.json().catch(() => ({}));

    const congestionado = resposta.status === 503;
    if (congestionado && tentativa < tentativas) {
      corpo.innerHTML =
        '<p class="pensando"><span class="pulso"></span>' +
        "Serviço congestionado. Tentando novamente…</p>";
      await new Promise((r) => setTimeout(r, tentativa * 2500));
      continue;
    }

    // 429 = limite de uso. A mensagem do servidor ja diz o que fazer
    // (esperar um minuto, voltar amanha); acrescentar "tente novamente"
    // so contradiria ela.
    if (resposta.status === 429) {
      throw new Error(erro.detail || "Limite de consultas atingido.");
    }

    throw new Error(
      congestionado
        ? "O serviço está congestionado agora. Espere alguns instantes e pergunte de novo."
        : (erro.detail || "Não foi possível consultar agora.") +
          " Tente novamente."
    );
  }
}

/* --- Montagem do parecer --------------------------------------------- */

function criarParecer(pergunta) {
  const no = modelo.content.cloneNode(true);
  const artigo = no.querySelector(".parecer");
  artigo.querySelector(".texto-pergunta").textContent = pergunta;
  respostas.prepend(artigo);
  return artigo;
}

function mostrarFontes(parecer, fontes) {
  if (!fontes || fontes.length === 0) return;
  const bloco = parecer.querySelector(".fontes");
  const lista = bloco.querySelector("ul");

  fontes.forEach((fonte) => {
    const item = document.createElement("li");

    const categoria = document.createElement("span");
    categoria.className = "categoria";
    categoria.textContent = fonte.categoria;

    const arquivo = document.createElement("span");
    arquivo.textContent = fonte.arquivo;

    item.append(categoria, arquivo);

    if (fonte.paginas && fonte.paginas.length) {
      const paginas = document.createElement("span");
      paginas.className = "paginas";
      const rotulo = fonte.paginas.length > 1 ? "páginas" : "página";
      paginas.textContent = rotulo + " " + fonte.paginas.join(", ");
      item.append(paginas);
    }
    lista.append(item);
  });

  bloco.hidden = false;
}

/* --- Formatacao do texto --------------------------------------------- */

function escapar(texto) {
  const div = document.createElement("div");
  div.textContent = texto;
  return div.innerHTML;
}

/**
 * Converte a resposta (markdown simples) em HTML.
 * Escapamos o texto ANTES de aplicar qualquer marcacao, para que conteudo
 * vindo do modelo nunca seja interpretado como HTML.
 */
function formatarTexto(texto) {
  const linhas = escapar(texto).split("\n");
  const saida = [];
  let emLista = false;

  const fecharLista = () => {
    if (emLista) { saida.push("</ul>"); emLista = false; }
  };

  linhas.forEach((linhaBruta) => {
    const linha = linhaBruta.trim();

    if (!linha) { fecharLista(); return; }

    // Titulos: ## Texto  ou  **Texto** sozinho na linha
    const titulo = linha.match(/^#{1,4}\s+(.*)$/);
    const negritoSolo = linha.match(/^\*\*(.+)\*\*:?$/);
    if (titulo || negritoSolo) {
      fecharLista();
      saida.push("<h3>" + realce(titulo ? titulo[1] : negritoSolo[1]) + "</h3>");
      return;
    }

    // Itens de lista: -, *, • ou "1."
    const item = linha.match(/^(?:[-*•]|\d+\.)\s+(.*)$/);
    if (item) {
      if (!emLista) { saida.push("<ul>"); emLista = true; }
      saida.push("<li>" + realce(item[1]) + "</li>");
      return;
    }

    fecharLista();
    saida.push("<p>" + realce(linha) + "</p>");
  });

  fecharLista();
  return saida.join("");
}

/** Aplica **negrito** dentro de uma linha ja escapada. */
function realce(trecho) {
  return trecho.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

ajustarAltura();
campo.focus();
