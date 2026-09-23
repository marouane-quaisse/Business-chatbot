"""
=============================================================================
RERANKING — Cross-Encoder
=============================================================================

POURQUOI LE RERANKING ?
-----------------------
La recherche vectorielle (bi-encoder) est RAPIDE mais approximative :
elle encode la question et les documents SÉPARÉMENT, puis compare les
vecteurs. Elle ne "voit" jamais la question et le document ensemble.

Le cross-encoder fait l'inverse : il prend la PAIRE (question, document)
et calcule un score de pertinence en les lisant ENSEMBLE.
→ Beaucoup plus précis, mais trop lent pour scanner toute la base.

STRATÉGIE EN DEUX TEMPS (le pattern classique du RAG pro)
---------------------------------------------------------
    1. RETRIEVE  : la recherche vectorielle ramène ~20 candidats (rapide)
    2. RERANK    : le cross-encoder note ces 20 paires et garde les 4
                   meilleures (précis)

C'est le "garde-fou" qui permet d'activer Multi-Query + HyDE sans que le
bruit ne pollue le contexte final envoyé au LLM.

COÛT
----
Modèle local (cross-encoder/ms-marco-MiniLM-L-6-v2, ~80 Mo).
Aucun appel API, aucun coût par requête. Juste un peu de CPU.
=============================================================================
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import config


# =============================================================================
# 1. CHARGEMENT DU MODÈLE (mis en cache)
# =============================================================================
# Le modèle est lourd à charger (~1-2 s). On le charge UNE SEULE FOIS et on
# le réutilise pour toutes les requêtes suivantes.
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Retourne le cross-encoder (chargé une seule fois)."""
    global _reranker
    if _reranker is None:
        print(f"[reranker] Chargement du modèle {config.RERANKER_MODEL}...")
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


# =============================================================================
# 2. FONCTION DE RERANKING
# =============================================================================
def rerank(
    question: str,
    documents: list[Document],
    top_k: int | None = None,
) -> list[Document]:
    """
    Reclasse les documents par pertinence réelle vis-à-vis de la question.

    Args:
        question: la question originale de l'utilisateur.
        documents: les candidats ramenés par la recherche vectorielle.
        top_k: nombre de documents à garder (sinon config.TOP_K).

    Returns:
        Les top_k documents les plus pertinents, triés du meilleur au pire.
    """
    if not documents:
        return []

    top_k = top_k or config.TOP_K

    # Le cross-encoder attend une liste de PAIRES (question, texte_du_doc).
    pairs = [(question, doc.page_content) for doc in documents]

    # predict() retourne un score par paire (plus grand = plus pertinent).
    scores = get_reranker().predict(pairs)

    # On associe chaque document à son score, puis on trie décroissant.
    scored = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    # On garde les top_k meilleurs et on attache le score dans les métadonnées
    # (utile pour le debug et pour afficher la confiance plus tard).
    results: list[Document] = []
    for doc, score in scored[:top_k]:
        doc.metadata["rerank_score"] = float(score)
        results.append(doc)

    return results
