"""
Cœur du RAG : récupération des chunks pertinents + génération de la réponse.

RÔLE DE CE MODULE DANS LE RAG
-----------------------------
C'est la TROISIÈME étape, celle qui se produit à CHAQUE question posée.

RAG = Retrieval-Augmented Generation, soit en français :
      "Génération Augmentée par Récupération".

L'idée en 3 temps :
  1. RETRIEVAL  : on cherche dans la base vectorielle les chunks les plus
                  proches de la question (le "matching").
  2. AUGMENTATION : on insère ces chunks dans le prompt envoyé au LLM.
                    Le LLM reçoit donc la question + le contexte pertinent.
  3. GENERATION : le LLM rédige une réponse en se basant sur ce contexte.

Pourquoi c'est puissant : le LLM n'a pas besoin d'avoir "appris" tes
documents. On lui fournit l'information au moment de la question. Résultat :
des réponses à jour, sur TES données, sans réentraîner de modèle.
"""

import time

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import (
    ENABLE_HYDE,
    ENABLE_MULTI_QUERY,
    ENABLE_RERANK,
    GROQ_API_KEY,
    GROQ_MODEL,
    RETRIEVAL_CANDIDATES,
    TOP_K,
)
from query_transform import build_search_queries
from reranker import rerank
from vectorstore import get_retriever

# Type de l'historique conversationnel : liste de (role, contenu).
# Exemple : [("user", "Quel est le délai ?"), ("assistant", "30 jours.")]
History = list[tuple[str, str]]

# =============================================================================
# 1. LE PROMPT SYSTÈME (les règles données au LLM)
# =============================================================================
# C'est LE point critique pour éviter les hallucinations. On impose au LLM :
#   - de ne répondre QUE depuis le contexte fourni
#   - de dire "je ne sais pas" si l'info est absente
#   - de citer ses sources
# Le {context} sera remplacé par les chunks récupérés.
SYSTEM_PROMPT = """Tu es un assistant qui répond aux questions des clients \
d'une entreprise en te basant UNIQUEMENT sur le contexte fourni.

Règles strictes :
1. Réponds uniquement à partir du contexte ci-dessous.
2. Si l'information n'est pas dans le contexte, réponds exactement :
   "Je ne dispose pas de cette information dans les documents fournis."
3. N'invente jamais d'information.
4. Réponds de façon claire, concise et professionnelle.
5. Si possible, indique la source (nom du document) utilisée.

Contexte :
{context}
"""


# =============================================================================
# 2. OUTILS INTERNES
# =============================================================================
def format_docs(docs: list[Document]) -> str:
    """
    Transforme la liste de chunks en un seul bloc de texte.

    On préfixe chaque chunk par sa source pour que le LLM puisse la citer.
    C'est l'étape "Augmentation" du RAG.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "inconnu")
        parts.append(f"[Source : {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def get_llm() -> ChatGroq:
    """
    Retourne le LLM Groq.

    temperature=0 : on veut des réponses factuelles et reproductibles,
    pas de créativité (essentiel pour un chatbot business).
    """
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )


def build_prompt(history: History | None = None) -> ChatPromptTemplate:
    """
    Construit le prompt en insérant l'historique conversationnel.

    STRUCTURE DU PROMPT
    -------------------
        [system]  → les règles (répondre uniquement depuis le contexte...)
        [human]   → question précédente 1
        [ai]      → réponse précédente 1
        [human]   → question précédente 2
        [ai]      → réponse précédente 2
        [human]   → {question}  ← la question actuelle

    POURQUOI L'HISTORIQUE ?
    Sans lui, le LLM ne peut pas comprendre les questions de suivi :
        "Et pour les DOM-TOM ?"  →  il ne sait pas de quoi on parle.

    ⚠️ L'historique est placé APRÈS le prompt système et AVANT la question
    actuelle : c'est l'ordre attendu par les modèles de chat.

    Si `history` est None ou vide, on retombe exactement sur l'ancien
    comportement (system + question) : la rétrocompatibilité est totale.
    """
    messages: list[tuple[str, str]] = [("system", SYSTEM_PROMPT)]

    if history:
        # Import local pour éviter un import circulaire (history.py importe
        # config, et rag.py est importé par api.py).
        from history import to_langchain_messages

        messages.extend(to_langchain_messages(history))

    messages.append(("human", "{question}"))

    return ChatPromptTemplate.from_messages(messages)


# =============================================================================
# 3. ÉTAPE RETRIEVAL : trouver les chunks pertinents
# =============================================================================
# Le retrieval "optimisé" se fait en 4 sous-étapes :
#
#   (a) QUERY TRANSFORMATION : on transforme 1 question en N requêtes
#       (question originale + variantes Multi-Query + texte HyDE)
#   (b) RECHERCHE MULTIPLE    : on interroge Chroma avec CHAQUE requête
#   (c) FUSION RRF            : on fusionne les N listes de résultats en une
#   (d) RERANKING             : le cross-encoder reclasse et garde le top-K
#
# Chaque sous-étape est désactivable via config.py / .env, ce qui permet de
# mesurer l'apport réel de chaque technique.
# =============================================================================


def _rrf_fuse(result_lists: list[list[Document]], k: int) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) : fusionne plusieurs listes de résultats.

    PROBLÈME : si on cherche avec 5 requêtes, on obtient 5 listes de chunks.
    Comment les combiner ? Les scores de similarité ne sont pas comparables
    d'une requête à l'autre. RRF résout ça en n'utilisant que les RANGS.

    FORMULE :  score(doc) = Σ  1 / (60 + rang_du_doc_dans_la_liste)

    - Un doc bien classé (rang 0) dans plusieurs listes obtient un gros score.
    - Un doc présent dans une seule liste est pénalisé.
    - Le "60" est une constante de lissage standard (issue de la littérature).

    C'est la méthode de fusion la plus robuste et la plus simple.
    """
    scores: dict[str, float] = {}
    docs_by_id: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            # Clé d'identité du chunk : source + début du contenu.
            # (Chroma ne nous donne pas toujours un id stable ici.)
            doc_id = f"{doc.metadata.get('source', '')}::{doc.page_content[:80]}"
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
            docs_by_id.setdefault(doc_id, doc)

    # Tri décroissant par score RRF, puis on garde les k meilleurs.
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [docs_by_id[doc_id] for doc_id, _ in ranked[:k]]


def retrieve(question: str, k: int = TOP_K, user_id: int | None = None) -> list[Document]:
    """
    Récupère les k chunks les plus pertinents pour la question.

    Pipeline :
      1. Query transformation  → liste de requêtes (si activé)
      2. Recherche vectorielle → une liste de candidats par requête
      3. Fusion RRF            → une seule liste fusionnée
      4. Reranking             → les k meilleurs (si activé)

    Si toutes les optimisations sont désactivées, on retombe exactement sur
    l'ancien comportement : une seule recherche vectorielle top-k.

    Args:
        question: la question (reformulée) de l'utilisateur.
        k: nombre de chunks à récupérer.
        user_id: si fourni, on ne cherche QUE dans les documents de cet
                 utilisateur. C'est la garantie d'isolation : un client
                 ne peut pas recevoir d'extraits des documents d'un autre.
    """
    # --- (a) Query transformation ---
    queries = build_search_queries(question)

    # Nombre de candidats à ramener par requête. On prend large AVANT
    # reranking : c'est le reranker qui fera le tri final.
    candidates_per_query = RETRIEVAL_CANDIDATES if ENABLE_RERANK else k

    # --- (b) Recherche vectorielle pour chaque requête ---
    retriever = get_retriever(candidates_per_query, user_id=user_id)
    result_lists = [retriever.invoke(q) for q in queries]

    # --- (c) Fusion des résultats ---
    if len(result_lists) == 1:
        # Une seule requête : pas besoin de fusionner.
        candidates = result_lists[0]
    else:
        # On fusionne en gardant assez de candidats pour le reranker.
        candidates = _rrf_fuse(result_lists, RETRIEVAL_CANDIDATES)

    # --- (d) Reranking ---
    if ENABLE_RERANK and candidates:
        return rerank(question, candidates, top_k=k)

    # Sans reranker : on coupe simplement au top-k.
    return candidates[:k]


# =============================================================================
# 4. PIPELINE COMPLET : retrieval + génération
# =============================================================================
def answer(
    question: str,
    k: int = TOP_K,
    history: History | None = None,
    user_id: int | None = None,
) -> dict:
    """
    Pipeline RAG complet : de la question à la réponse.

    Args:
        question: la question de l'utilisateur.
        k: nombre de chunks à récupérer.
        history: les messages précédents [(role, content), ...].
                 Si None, le comportement est identique à avant
                 (question isolée, sans contexte conversationnel).
        user_id: si fourni, la recherche est restreinte aux documents de
                 cet utilisateur (isolation multi-tenant).

    Returns:
        dict avec :
          - answer  : la réponse générée par le LLM
          - sources : liste des documents sources utilisés
          - chunks  : les chunks bruts récupérés (utile pour le debug)
    """
    # --- Étape 1 : RETRIEVAL ---
    # ⚠️ On cherche avec la question REFORMULÉE (faite en amont par
    # history.reformulate) : une question de suivi comme "Et pour les
    # DOM-TOM ?" ne matcherait rien telle quelle.
    docs = retrieve(question, k, user_id=user_id)

    # Cas particulier : base vide → on ne peut rien répondre.
    if not docs:
        return {
            "answer": "Aucun document n'est indexé. Veuillez d'abord ingérer des ressources.",
            "sources": [],
            "chunks": [],
        }

    # --- Étape 2 : construction du prompt (AUGMENTATION) ---
    prompt = build_prompt(history)

    # --- Étape 3 : GENERATION ---
    # La syntaxe "|" est un "pipe" LangChain : la sortie de chaque étape
    # devient l'entrée de la suivante.
    #   prompt → LLM → extraction du texte
    chain = prompt | get_llm() | StrOutputParser()

    response = chain.invoke(
        {"context": format_docs(docs), "question": question}
    )

    # On déduplique les sources (plusieurs chunks peuvent venir du même fichier).
    sources = sorted({doc.metadata.get("source", "inconnu") for doc in docs})

    return {"answer": response, "sources": sources, "chunks": docs}


# =============================================================================
# 5. VERSION STREAMING (pour l'interface web)
# =============================================================================
# POURQUOI ?
# L'utilisateur n'aime pas attendre 2-3 secondes devant un écran figé.
# Avec le streaming, on affiche les mots AU FUR ET À MESURE que le LLM les
# produit. La latence perçue passe de ~2s à ~0.3s.
#
# COMMENT ?
# C'est un GÉNÉRATEUR Python (mot-clé `yield`). Chaque `yield` produit un
# "événement" que l'API FastAPI convertit en Server-Sent Event (SSE) et
# envoie au navigateur en temps réel.


def stream_answer(
    question: str,
    k: int = TOP_K,
    history: History | None = None,
    user_id: int | None = None,
):
    """
    Pipeline RAG complet, mais en streaming token par token.

    Args:
        question: la question de l'utilisateur.
        k: nombre de chunks à récupérer.
        history: les messages précédents [(role, content), ...].
                 Si None, comportement identique à avant (sans contexte).
        user_id: si fourni, la recherche est restreinte aux documents de
                 cet utilisateur (isolation multi-tenant).

    Événements émis (tuples) :
        ("sources", [...])             → dès que le retrieval est terminé
        ("token",   "morceau texte")   → répété pendant la génération
        ("done",    {"latency": 1.23}) → fin normale
        ("error",   "message")         → en cas de problème

    Usage :
        for event, payload in stream_answer("ma question"):
            print(event, payload)
    """
    start = time.perf_counter()

    # --- Étape 1 : RETRIEVAL (bloquant, mais rapide ~0.1s sans MQ/HyDE) ---
    try:
        docs = retrieve(question, k, user_id=user_id)
    except Exception as exc:  # noqa: BLE001
        yield ("error", f"Erreur pendant la recherche : {exc}")
        return

    # Base vide → message explicite plutôt qu'une réponse vide.
    if not docs:
        yield ("token", "Aucun document n'est indexé. Veuillez d'abord ingérer des ressources.")
        yield ("done", {"latency": time.perf_counter() - start})
        return

    # On envoie les sources AVANT le texte : l'UI peut les afficher tout de suite.
    sources = sorted({doc.metadata.get("source", "inconnu") for doc in docs})
    yield ("sources", sources)

    # --- Étape 2 : AUGMENTATION (avec l'historique conversationnel) ---
    prompt = build_prompt(history)

    # --- Étape 3 : GÉNÉRATION en streaming ---
    # `.stream()` renvoie les morceaux au fur et à mesure (vs `.invoke()`
    # qui attend la réponse complète).
    chain = prompt | get_llm() | StrOutputParser()

    try:
        for chunk in chain.stream({"context": format_docs(docs), "question": question}):
            yield ("token", chunk)
    except Exception as exc:  # noqa: BLE001
        yield ("error", f"Erreur pendant la génération : {exc}")
        return

    yield ("done", {"latency": time.perf_counter() - start})
