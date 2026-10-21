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

/**
 * O historico vive AQUI, no navegador, e viaja em cada pergunta.
 *
 * O servidor fica sem estado: nada para limpar, nada que se perca num
 * restart, e funciona igual com varios processos. O preco e confiar no
 * cliente, por isso a API limita o tamanho do que aceita.
 */
const historico = [];
const MAX_HISTORICO = 4;

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
    const resposta = await transmitir(pergunta, corpo, parecer);
    if (resposta) {
      historico.push({ pergunta, resposta });
      if (historico.length > MAX_HISTORICO) historico.shift();
    }
  } catch (erro) {
    corpo.innerHTML = '<p class="falha">' + escapar(erro.message) + "</p>";
  } finally {
    ocupado = false;
    botao.disabled = false;
    campo.focus();
  }
}

/**
 * Le a resposta em pedacos, conforme o modelo escreve.
 *
 * Medido: a resposta leva de 25 a 60 segundos, e a maior parte disso
 * acontece ANTES da primeira palavra (o modelo lendo o contexto). O
 * streaming nao deixa mais rapido -- deixa visivel que algo esta
 * acontecendo, em vez de um "carregando" parado por um minuto.
 *
 * O formato e NDJSON: um objeto JSON por linha.
 */
async function transmitir(pergunta, corpo, parecer) {
  const resposta = await fetch("/api/perguntar/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pergunta, historico }),
  });

  // 429 (limite de uso) e 503 (agente carregando) chegam ANTES do fluxo
  // comecar, entao ainda sao erros HTTP normais.
  if (!resposta.ok) {
    const erro = await resposta.json().catch(() => ({}));
    throw new Error(erro.detail || "Não foi possível consultar agora.");
  }

  const leitor = resposta.body.getReader();
  const decodificador = new TextDecoder();
  let sobra = "";
  let texto = "";
  let primeiro = true;

  for (;;) {
    const { done, value } = await leitor.read();
    if (done) break;

    // Um pedaco da rede pode cortar uma linha no meio. Guardamos o resto
    // e so processamos linhas completas -- sem isto o JSON.parse falha
    // de forma intermitente e dificil de reproduzir.
    sobra += decodificador.decode(value, { stream: true });
    const linhas = sobra.split("\n");
    sobra = linhas.pop();

    for (const linha of linhas) {
      if (!linha.trim()) continue;
      const evento = JSON.parse(linha);

      if (evento.tipo === "texto") {
        if (primeiro) {
          corpo.innerHTML = "";
          primeiro = false;
        }
        texto += evento.dado;
        corpo.innerHTML = formatarTexto(texto);
      } else if (evento.tipo === "substituir") {
        // O agente recusou e continuou falando. O servidor manda o texto
        // limpo e trocamos o que ja foi exibido.
        texto = evento.dado;
        corpo.innerHTML = formatarTexto(texto);
      } else if (evento.tipo === "fontes") {
        mostrarFontes(parecer, evento.dado);
      } else if (evento.tipo === "erro") {
        // Depois do primeiro byte o status HTTP ja foi enviado e nao pode
        // mais virar 500, entao o erro viaja dentro do fluxo.
        throw new Error(evento.dado);
      }
    }
  }
  return texto;
}


/* --- Montagem do parecer --------------------------------------------- */

function criarParecer(pergunta) {
  const no = modelo.content.cloneNode(true);
  const artigo = no.querySelector(".parecer");
  artigo.querySelector(".texto-pergunta").textContent = pergunta;
  respostas.prepend(artigo);
  return artigo;
}

/**
 * Desenha cada fonte como um LINK para o documento original, na pagina
 * citada, com o trecho exato que embasou a resposta logo abaixo.
 *
 * A promessa do projeto e "voce pode conferir". Enquanto a fonte era
 * texto morto -- "pagina 39" sem link e sem trecho -- a promessa nao se
 * cumpria: era preciso acreditar.
 */
function mostrarFontes(parecer, fontes) {
  if (!fontes || fontes.length === 0) return;
  const bloco = parecer.querySelector(".fontes");
  const lista = bloco.querySelector("ul");

  fontes.forEach((fonte) => {
    const item = document.createElement("li");

    const categoria = document.createElement("span");
    categoria.className = "categoria";
    categoria.textContent = fonte.categoria;
    item.append(categoria);

    // #page=N e entendido pelo leitor de PDF do navegador, entao o
    // documento abre direto no ponto citado -- nao no comeco de 100
    // paginas.
    if (fonte.caminho) {
      const link = document.createElement("a");
      link.className = "arquivo";
      const pagina = fonte.paginas && fonte.paginas.length ? fonte.paginas[0] : null;
      link.href = "/documentos/" + fonte.caminho + (pagina ? "#page=" + pagina : "");
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = fonte.arquivo;
      item.append(link);
    } else {
      const nome = document.createElement("span");
      nome.textContent = fonte.arquivo;
      item.append(nome);
    }

    if (fonte.paginas && fonte.paginas.length) {
      const paginas = document.createElement("span");
      paginas.className = "paginas";
      const rotulo = fonte.paginas.length > 1 ? "páginas" : "página";
      paginas.textContent = rotulo + " " + fonte.paginas.join(", ");
      item.append(paginas);
    }

    if (fonte.trechos && fonte.trechos.length) {
      item.append(caixaDeTrechos(fonte.trechos));
    }
    lista.append(item);
  });

  bloco.hidden = false;
}

/**
 * O texto exato que o modelo recebeu, fechado por padrao.
 *
 * Fica recolhido porque a maioria quer so a resposta; quem duvida, abre.
 * O texto nao e cortado: truncar a prova derrotaria o proposito de
 * mostra-la.
 */
function caixaDeTrechos(trechos) {
  const detalhes = document.createElement("details");
  detalhes.className = "trechos";

  const resumo = document.createElement("summary");
  resumo.textContent =
    trechos.length > 1
      ? "ver os " + trechos.length + " trechos citados"
      : "ver o trecho citado";
  detalhes.append(resumo);

  trechos.forEach((trecho) => {
    const bloco = document.createElement("blockquote");
    if (trecho.pagina) {
      const marca = document.createElement("span");
      marca.className = "marca-pagina";
      marca.textContent = "pág. " + trecho.pagina;
      bloco.append(marca);
    }
    bloco.append(document.createTextNode(trecho.texto));
    detalhes.append(bloco);
  });
  return detalhes;
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
