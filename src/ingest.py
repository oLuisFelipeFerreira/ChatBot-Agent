"""
Le os documentos de referencia, limpa o texto, divide em pedacos (chunks),
gera embeddings locais e monta um vector store FAISS.

A pasta data/ e organizada em SUBPASTAS, e o nome de cada subpasta vira a
CATEGORIA do documento (guardada nos metadados e exibida nas fontes):

    data/
      legislacao/            -> categoria "Legislacao"
      registro-empresarial/  -> categoria "Registro Empresarial"
      tributario/            -> categoria "Tributario"

Formatos aceitos: .pdf, .csv, .html, .htm, .txt
(as leis do Planalto sao paginas HTML; e so salvar a pagina como .html)

Uso:
    python -m src.ingest            # indexa a pasta data/ inteira
    python -m src.ingest data/legislacao
"""

import json
import re
import sys
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"
# Os chunks sao salvos em disco porque o BM25 (busca por palavra-chave)
# precisa deles em memoria -- ele nao persiste sozinho como o FAISS.
CHUNKS_FILE = "faiss_index/chunks.json"
# Modelo de embeddings MULTILINGUE.
# O all-MiniLM-L6-v2 (usado antes) e treinado em ingles e falhava com
# texto juridico em portugues. Este entende portugues e melhora muito a
# recuperacao. Mesmas 384 dimensoes, entao o custo e parecido.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PASTA_DADOS = "data"

# Arquivos que ficam na pasta de dados mas NAO sao base de conhecimento.
# O README da pasta explica como adicionar documentos; indexa-lo coloca o
# texto "Coloque aqui o arquivo PDF..." competindo com a lei no indice.
IGNORAR = {"README.md"}

EXTENSOES = {".pdf", ".csv", ".html", ".htm", ".txt", ".md"}

# Textos juridicos tem artigos longos: chunks maiores preservam melhor o sentido
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _ler_texto_bruto(path: str) -> str:
    """Le um arquivo de texto tentando varias codificacoes.

    Documentos oficiais antigos (como as paginas do Planalto) costumam
    usar ISO-8859-1 em vez de UTF-8. Tentamos na ordem mais provavel e,
    em ultimo caso, ignoramos os bytes invalidos em vez de quebrar.
    """
    dados = Path(path).read_bytes()
    for codificacao in ("utf-8", "iso-8859-1", "cp1252"):
        try:
            return dados.decode(codificacao)
        except UnicodeDecodeError:
            continue
    return dados.decode("utf-8", errors="ignore")


def _carregar_html(path: str):
    """Extrai o texto de um arquivo HTML.

    Dois cuidados importantes, descobertos com as paginas do Planalto:

    1) CODIFICACAO: nao deixamos o BeautifulSoup adivinhar. Ele errava e
       escolhia cp1250, transformando "apuração" em "apuraçăo". Decodificamos
       nos mesmos, com ordem controlada, e passamos a string pronta.

    2) QUEBRA DE PALAVRAS: as paginas do Planalto vem do Word e tem <span>
       no MEIO das palavras. Usar get_text(separator="\\n") inseria quebras
       ali, virando "O v alor d e vido". A solucao e extrair BLOCO A BLOCO
       (cada <p>, <li>, <td>) sem separador interno: assim o texto inline
       de cada bloco e concatenado corretamente.
    """
    texto_html = _ler_texto_bruto(path)
    soup = BeautifulSoup(texto_html, "html.parser")

    # Remove elementos que so atrapalham
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # CRITICO para paginas de "texto compilado" do Planalto:
    # a redacao REVOGADA aparece TACHADA ao lado da vigente. Visualmente da
    # para distinguir, mas ao extrair o texto puro as duas se misturam -- e o
    # agente pode responder com um limite que nao vale mais.
    # Removemos as tags de tachado e tambem estilos inline line-through.
    for tag in soup(["strike", "s", "del"]):
        tag.decompose()
    for tag in soup.find_all(style=True):
        if "line-through" in tag.get("style", "").lower():
            tag.decompose()

    blocos = []
    for elemento in soup.find_all(
        ["p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "blockquote"]
    ):
        # Pula blocos que contem outros blocos, para nao duplicar o texto
        if elemento.find(["p", "li", "td", "th", "blockquote"]):
            continue
        # get_text() SEM separador: junta os <span> internos sem quebrar palavras
        conteudo = re.sub(r"\s+", " ", elemento.get_text()).strip()
        if conteudo:
            blocos.append(conteudo)

    # Se o HTML nao usa blocos (raro), cai para a extracao simples
    if not blocos:
        blocos = [re.sub(r"\s+", " ", soup.get_text()).strip()]

    return [
        Document(
            page_content="\n".join(blocos),
            metadata={"source": str(path)},
        )
    ]


def load_document(path: str):
    """Carrega o documento de acordo com a extensao."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext == ".csv":
        return CSVLoader(path, encoding="utf-8").load()
    if ext in {".html", ".htm"}:
        return _carregar_html(path)
    if ext in {".txt", ".md"}:
        texto = _ler_texto_bruto(path)
        return [Document(page_content=texto, metadata={"source": str(path)})]
    raise ValueError(
        f"Extensao nao suportada: {ext}. Use: {', '.join(sorted(EXTENSOES))}"
    )


def limpar_texto(texto: str) -> str:
    """Remove ruidos comuns de documentos oficiais.

    Os padroes foram definidos olhando o texto real dos PDFs da Receita
    Federal e das paginas de legislacao do Planalto.
    """
    linhas_limpas = []
    for linha in texto.split("\n"):
        l = linha.strip()
        if not l:
            continue                              # linha vazia
        if re.fullmatch(r"Pág\.\s*\d+", l):
            continue                              # marcador "Pág. 12"
        if l == "Voltar ao Sumário":
            continue                              # rodape de navegacao
        if re.fullmatch(r"\d{1,3}", l):
            continue                              # numero de pagina isolado
        if re.search(r"\.{4,}", l):
            continue                              # linha do sumario (pontilhado)
        linhas_limpas.append(l)
    texto = "\n".join(linhas_limpas)
    texto = re.sub(r"[ \t]{2,}", " ", texto)      # colapsa espacos repetidos
    texto = re.sub(r"\n{3,}", "\n\n", texto)      # colapsa linhas em branco
    return texto


def nome_categoria(caminho_arquivo: Path, raiz: Path) -> str:
    """Deriva a categoria a partir da subpasta em que o arquivo esta.

    data/registro-empresarial/manual.pdf  ->  "Registro Empresarial"
    data/manual.pdf (solto na raiz)       ->  "Geral"
    """
    try:
        relativo = caminho_arquivo.relative_to(raiz)
    except ValueError:
        return "Geral"
    if len(relativo.parts) < 2:
        return "Geral"
    return relativo.parts[0].replace("-", " ").replace("_", " ").title()


def coletar_arquivos(caminho: str):
    """Retorna a lista de arquivos a indexar.

    Se 'caminho' for uma pasta, varre TODAS as subpastas recursivamente.
    Se for um arquivo, retorna so ele.
    """
    p = Path(caminho)
    if p.is_dir():
        arquivos = sorted(
            f for f in p.rglob("*")
            if f.suffix.lower() in EXTENSOES and f.name not in IGNORAR
        )
        if not arquivos:
            raise ValueError(
                f"Nenhum documento encontrado em '{caminho}'. "
                f"Formatos aceitos: {', '.join(sorted(EXTENSOES))}"
            )
        return arquivos
    return [p]


def build_vectorstore(path: str = PASTA_DADOS, index_dir: str = INDEX_DIR):
    """Le os documentos, limpa, divide em chunks, indexa e salva."""
    raiz = Path(path) if Path(path).is_dir() else Path(path).parent
    arquivos = coletar_arquivos(path)

    todos_documentos = []
    resumo_por_categoria = {}

    for arquivo in arquivos:
        categoria = nome_categoria(arquivo, raiz)
        documentos = load_document(str(arquivo))

        for doc in documentos:
            doc.page_content = limpar_texto(doc.page_content)
            # Enriquece os metadados: categoria e nome amigavel do arquivo
            doc.metadata["categoria"] = categoria
            doc.metadata["arquivo"] = arquivo.name

        # Descarta paginas que ficaram vazias apos a limpeza
        documentos = [d for d in documentos if d.page_content.strip()]
        todos_documentos.extend(documentos)
        resumo_por_categoria.setdefault(categoria, []).append(arquivo.name)

    print(f"\n{len(arquivos)} documento(s) lido(s):")
    for categoria, nomes in sorted(resumo_por_categoria.items()):
        print(f"  [{categoria}]")
        for nome in nomes:
            print(f"    - {nome}")

    # Separadores em ordem de prioridade. O splitter tenta cortar primeiro
    # nos pontos mais "naturais" do texto juridico -- inicio de artigo, de
    # paragrafo, de inciso -- e so quebra no meio da frase em ultimo caso.
    # Sem isso, o art. 3o ficava partido: o titulo num chunk e os valores
    # noutro, e a busca achava um sem o outro.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\nArt. ",     # inicio de artigo
            "\nArt.",
            "\n§ ",        # paragrafo
            "\nParágrafo", # paragrafo unico
            "\n\n",        # linha em branco
            "\n",
            ". ",
            " ",
            "",
        ],
        keep_separator=True,
    )
    chunks = splitter.split_documents(todos_documentos)
    print(f"\nDividido em {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)

    salvar_chunks(chunks)
    print(f"Indice salvo em '{index_dir}/'.\n")
    return vectorstore


def salvar_chunks(chunks, caminho: str = CHUNKS_FILE):
    """Grava os chunks em JSON para o BM25 poder ser reconstruido depois."""
    dados = [
        {"page_content": c.page_content, "metadata": c.metadata} for c in chunks
    ]
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    Path(caminho).write_text(
        json.dumps(dados, ensure_ascii=False), encoding="utf-8"
    )


def carregar_chunks(caminho: str = CHUNKS_FILE):
    """Le os chunks salvos. Devolve [] se o arquivo nao existir."""
    p = Path(caminho)
    if not p.exists():
        return []
    dados = json.loads(p.read_text(encoding="utf-8"))
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in dados
    ]


if __name__ == "__main__":
    alvo = sys.argv[1] if len(sys.argv) > 1 else PASTA_DADOS
    build_vectorstore(alvo)
