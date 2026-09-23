"""
Vector store : génération des embeddings et stockage dans ChromaDB.

RÔLE DE CE MODULE DANS LE RAG
-----------------------------
C'est la DEUXIÈME étape du pipeline (après le chunking).

Le principe :
  1. On transforme chaque chunk de texte en VECTEUR (une liste de nombres).
     C'est ce qu'on appelle un "embedding".
  2. Deux textes qui parlent de la même chose auront des vecteurs PROCHES
     dans l'espace. C'est ce qui permet la recherche "sémantique" : on ne
     cherche pas les mots exacts, mais le SENS.
  3. On stocke tous ces vecteurs dans une base spécialisée (Chroma) capable
     de retrouver très vite les vecteurs les plus proches d'une question.

Chroma est PERSISTANT sur disque (dossier vector_db/) : les embeddings sont
conservés entre deux exécutions, pas besoin de tout recalculer.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_DB_DIR


# =============================================================================
# 1. LE MODÈLE D'EMBEDDINGS
# =============================================================================
# ⚡ OPTIMISATION CRITIQUE : on met le modèle en CACHE.
#
# Sans cache, HuggingFaceEmbeddings recharge le modèle (~90 Mo) à CHAQUE appel
# de get_embeddings(). Comme get_vectorstore() appelle get_embeddings(), et que
# get_retriever() appelle get_vectorstore(), le modèle était rechargé à chaque
# question → ~2.5s de latence inutile par requête.
#
# Avec le cache, le modèle est chargé UNE SEULE FOIS au premier usage, puis
# réutilisé. Gain mesuré : ~8x plus rapide.
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Retourne le modèle qui transforme le texte en vecteurs.

    On utilise un modèle LOCAL (sentence-transformers) :
      - gratuit (pas d'appel API, pas de coût par token)
      - tourne sur ta machine
      - le modèle est téléchargé automatiquement au premier usage (~90 Mo)

    normalize_embeddings=True : on ramène tous les vecteurs à une longueur
    de 1. Cela rend la similarité cosinus plus fiable et comparable.

    ⚡ Le modèle est mis en cache : chargé une seule fois pour toute la session.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# =============================================================================
# 2. LA BASE VECTORIELLE (Chroma)
# =============================================================================
# ⚡ Même logique : on garde l'objet Chroma en cache. Le recréer à chaque appel
# rouvrait la connexion au disque et reconstruisait l'index en mémoire.
_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """
    Ouvre (ou crée) la collection Chroma persistante.

    Si le dossier vector_db/ n'existe pas encore, Chroma le crée.
    Si la collection existe déjà, on la réutilise telle quelle.

    ⚡ L'objet est mis en cache : une seule connexion pour toute la session.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(VECTOR_DB_DIR),
        )
    return _vectorstore


# =============================================================================
# 3. INDEXATION : écrire les chunks dans la base
# =============================================================================
def index_documents(chunks: list[Document], reset: bool = False) -> int:
    """
    Indexe les chunks dans Chroma.

    Pour chaque chunk, Chroma va :
      1. calculer son embedding (via le modèle ci-dessus)
      2. stocker le vecteur + le texte + les métadonnées

    Args:
        chunks: liste de Documents à indexer.
        reset: si True, supprime la collection existante avant d'indexer
               (utile pour repartir de zéro et éviter les doublons).

    Returns:
        Le nombre de chunks indexés.
    """
    if reset:
        reset_collection()

    if not chunks:
        return 0

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def delete_user_chunks(user_id: int) -> None:
    """
    Supprime de Chroma tous les chunks appartenant à un utilisateur.

    ⚠️ POURQUOI PAS UN SIMPLE `reset_collection()` ?
    Parce que la collection est PARTAGÉE entre tous les utilisateurs.
    La vider supprimerait aussi les documents des autres. On supprime donc
    uniquement les vecteurs dont la métadonnée `user_id` correspond.

    C'est ce qui rend la réindexation d'un utilisateur possible sans
    impacter les autres.
    """
    vectorstore = get_vectorstore()
    vectorstore._collection.delete(where={"user_id": user_id})


def delete_document_chunks(user_id: int, filename: str) -> None:
    """
    Supprime de Chroma les chunks d'UN SEUL document d'un utilisateur.

    ⚠️ POURQUOI PAS `delete_user_chunks()` ?
    Parce que celle-ci efface TOUS les chunks de l'utilisateur. L'utiliser
    lors d'un upload supprimerait les autres documents de l'utilisateur !
    Ici on cible précisément le fichier concerné, via sa métadonnée `source`.

    C'est indispensable pour ré-uploader un document corrigé : on retire
    l'ancienne version sans toucher au reste de la bibliothèque.
    """
    vectorstore = get_vectorstore()
    vectorstore._collection.delete(where={"$and": [{"user_id": user_id}, {"source": filename}]})


def reset_collection() -> None:
    """
    Supprime toute la collection (pour réindexer proprement).

    ⚡ Important : on invalide le cache après suppression. Sinon on garderait
    une référence vers une collection qui n'existe plus, et les appels
    suivants échoueraient.
    """
    global _vectorstore
    vectorstore = get_vectorstore()
    vectorstore.delete_collection()
    _vectorstore = None  # force la recréation au prochain appel


# =============================================================================
# 4. RETRIEVAL : lire dans la base
# =============================================================================
def get_retriever(k: int, user_id: int | None = None):
    """
    Retourne un "retriever" : un objet qui, à partir d'une question,
    renvoie les k chunks les plus proches sémantiquement.

    C'est le pont entre la base vectorielle et le module rag.py.

    Args:
        k: nombre de chunks à récupérer.
        user_id: si fourni, on ne cherche QUE dans les chunks de cet
                 utilisateur. C'est le cœur de l'isolation : sans ce
                 filtre, un utilisateur pourrait recevoir des extraits
                 des documents d'un autre.

    ⚠️ Le filtre est appliqué par Chroma AVANT le calcul de similarité,
    ce qui est à la fois plus sûr et plus rapide (moins de vecteurs à
    comparer).
    """
    vectorstore = get_vectorstore()

    search_kwargs: dict = {"k": k}
    if user_id is not None:
        search_kwargs["filter"] = {"user_id": user_id}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def collection_count() -> int:
    """Retourne le nombre de chunks actuellement indexés (pour l'affichage)."""
    vectorstore = get_vectorstore()
    return vectorstore._collection.count()
