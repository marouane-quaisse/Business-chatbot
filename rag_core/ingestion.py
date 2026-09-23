"""
Ingestion des documents : chargement depuis un dossier + découpage en chunks.

RÔLE DE CE MODULE DANS LE RAG
-----------------------------
C'est la PREMIÈRE étape du pipeline. Avant de pouvoir "chercher" dans des
documents, il faut :
  1. Les LIRE (extraire le texte brut d'un PDF, d'un .md, etc.)
  2. Les DÉCOUPER en petits morceaux (chunks) qu'on pourra indexer.

On ne fait PAS encore d'embeddings ici : ce module produit juste des
"Documents" LangChain (texte + métadonnées), prêts à être vectorisés.

Formats supportés pour l'instant : .txt, .md, .pdf
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE

# Extensions de fichiers qu'on sait lire. Toute autre extension sera ignorée.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


# =============================================================================
# 1. CHARGEMENT : transformer un fichier en texte brut
# =============================================================================
def load_txt(path: Path) -> str:
    """Charge un fichier texte brut (.txt ou .md)."""
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    """
    Charge un PDF et concatène le texte de toutes ses pages.

    Note : pypdf n'extrait que le TEXTE. Un PDF scanné (image) ne donnera
    rien ici — il faudrait de l'OCR (Tesseract) pour ça.

    ⚠️ On ouvre le fichier nous-mêmes et on le referme explicitement.
    Sous Windows, un handle encore ouvert empêche la suppression du
    fichier (WinError 32) — ce qui faisait échouer l'upload juste après
    la lecture du PDF.
    """
    from pypdf import PdfReader

    with path.open("rb") as handle:
        reader = PdfReader(handle)
        # extract_text() peut renvoyer None sur une page vide → on met "".
        pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_document(path: Path) -> str:
    """
    Aiguillage : choisit la bonne méthode de lecture selon l'extension.
    C'est ici qu'on ajouterait .docx, .csv, etc. plus tard.
    """
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return load_txt(path)
    if ext == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Format non supporté : {ext}")


def load_documents_from_folder(folder: Path) -> list[Document]:
    """
    Parcourt un dossier (récursivement) et retourne une liste de Documents.

    Un "Document" LangChain = un objet avec :
      - page_content : le texte
      - metadata     : des infos annexes (nom du fichier, chemin...)

    Les métadonnées sont ESSENTIELLES : c'est grâce à elles qu'on pourra
    plus tard citer la source d'une réponse, ou filtrer par business.
    """
    documents: list[Document] = []

    # rglob("*") parcourt aussi les sous-dossiers.
    for path in sorted(folder.rglob("*")):
        # On ignore les dossiers et les extensions non supportées.
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            text = load_document(path)
        except Exception as e:
            # Un fichier corrompu ne doit pas arrêter toute l'ingestion.
            print(f"  ⚠️  Échec du chargement de {path.name} : {e}")
            continue

        # Un fichier vide n'apporte rien : on l'ignore.
        if not text.strip():
            print(f"  ⚠️  {path.name} est vide, ignoré.")
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "path": str(path)},
            )
        )
        print(f"  ✅ Chargé : {path.name} ({len(text)} caractères)")

    return documents


# =============================================================================
# 2. CHUNKING : découper les documents en morceaux
# =============================================================================
def split_documents(documents: list[Document]) -> list[Document]:
    """
    Découpe chaque document en chunks de taille contrôlée.

    POURQUOI DÉCOUPER ?
      - Un document entier est trop long pour être envoyé au LLM.
      - La recherche vectorielle est plus PRÉCISE sur de petits morceaux :
        on veut retrouver LE paragraphe qui répond, pas tout le document.

    RecursiveCharacterTextSplitter essaie de couper "intelligemment" :
    d'abord aux paragraphes (\\n\\n), puis aux lignes, puis aux phrases,
    et seulement en dernier recours au milieu d'un mot. Cela préserve le
    sens du texte autant que possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Ordre de priorité des séparateurs : du plus "gros" au plus fin.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # On ajoute un numéro d'ordre à chaque chunk (utile pour le debug et
    # pour reconstituer l'ordre d'origine si besoin).
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
