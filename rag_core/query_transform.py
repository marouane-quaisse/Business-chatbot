"""
=============================================================================
QUERY TRANSFORMATION — Multi-Query + HyDE
=============================================================================

PROBLÈME RÉSOLU
---------------
Une question d'utilisateur et un document métier ne sont PAS écrits de la
même façon :

    Question : "Je peux renvoyer un article ?"
    Document : "Politique de retour : le client dispose de 30 jours..."

La recherche vectorielle compare des embeddings. Si les mots/formulations
sont trop différents, le bon document peut être mal classé (mauvais recall).

DEUX TECHNIQUES COMPLÉMENTAIRES
-------------------------------
1. MULTI-QUERY : on demande au LLM de reformuler la question de N façons
   différentes (synonymes, angles). On cherche avec TOUTES les variantes.
   → Diversité de VOCABULAIRE.

2. HyDE (Hypothetical Document Embeddings) : on demande au LLM d'écrire une
   réponse hypothétique à la question, puis on cherche avec CE texte.
   → Diversité de FORME (question → document).

Les deux attaquent le même problème sous deux angles différents, donc elles
se cumulent bien. Le reranker (voir reranker.py) nettoie ensuite le bruit.

COÛT
----
Chaque technique = 1 appel LLM supplémentaire (modèle auxiliaire rapide).
C'est acceptable car ces appels sont petits et parallélisables plus tard.
=============================================================================
"""

from langchain_groq import ChatGroq

import config


# =============================================================================
# 1. LLM AUXILIAIRE (petit modèle rapide)
# =============================================================================
# On utilise un modèle différent du modèle principal de génération :
#   - plus rapide et moins cher (llama-3.1-8b-instant)
#   - température plus haute (0.3) car on veut de la DIVERSITÉ, pas de la
#     précision. Pour générer des variantes, un peu d'aléatoire aide.
_aux_llm = None


def get_aux_llm() -> ChatGroq:
    """Retourne le LLM auxiliaire (créé une seule fois, mis en cache)."""
    global _aux_llm
    if _aux_llm is None:
        _aux_llm = ChatGroq(
            model=config.AUX_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0.3,
        )
    return _aux_llm


# =============================================================================
# 2. PROMPTS
# =============================================================================
MULTI_QUERY_PROMPT = """Tu es un assistant qui aide à la recherche documentaire.

Génère {count} reformulations différentes de la question suivante.
Chaque reformulation doit :
- garder le MÊME sens que la question originale
- utiliser des mots-clés et formulations différents (synonymes, angles)
- être une question ou une courte phrase de recherche

Réponds UNIQUEMENT avec les reformulations, une par ligne, sans numérotation,
sans tirets, sans texte supplémentaire.

Question originale : {question}
"""

HYDE_PROMPT = """Tu es un expert qui rédige de la documentation d'entreprise.

Écris un COURT paragraphe (3-5 phrases) qui répondrait à la question suivante,
comme si tu étais un document officiel de l'entreprise.

Ne dis pas "je ne sais pas". Invente une réponse plausible et factuelle dans
le style d'un document métier. Ce texte servira uniquement à la recherche.

Question : {question}

Paragraphe :"""


# =============================================================================
# 3. MULTI-QUERY
# =============================================================================
def generate_query_variants(question: str, count: int | None = None) -> list[str]:
    """
    Demande au LLM de reformuler la question en plusieurs variantes.

    Retourne une liste de variantes (sans la question originale).
    En cas d'erreur, retourne une liste vide (dégradation gracieuse).
    """
    count = count or config.MULTI_QUERY_COUNT
    try:
        prompt = MULTI_QUERY_PROMPT.format(count=count, question=question)
        response = get_aux_llm().invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        # Nettoyage : une variante par ligne, on ignore les lignes vides.
        variants = [
            line.strip().lstrip("-•*0123456789. ").strip()
            for line in text.splitlines()
        ]
        variants = [v for v in variants if len(v) > 3]
        return variants[:count]
    except Exception as exc:  # noqa: BLE001
        print(f"[query_transform] Multi-Query échoué : {exc}")
        return []


# =============================================================================
# 4. HyDE
# =============================================================================
def generate_hypothetical_document(question: str) -> str | None:
    """
    Demande au LLM d'écrire une réponse hypothétique (style document).

    Retourne le texte généré, ou None en cas d'erreur.
    """
    try:
        prompt = HYDE_PROMPT.format(question=question)
        response = get_aux_llm().invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()
        return text if len(text) > 10 else None
    except Exception as exc:  # noqa: BLE001
        print(f"[query_transform] HyDE échoué : {exc}")
        return None


# =============================================================================
# 5. FONCTION PRINCIPALE
# =============================================================================
def build_search_queries(
    question: str,
    use_multi_query: bool | None = None,
    use_hyde: bool | None = None,
) -> list[str]:
    """
    Construit la liste de TOUTES les requêtes à envoyer à la recherche.

    La question originale est TOUJOURS en première position : c'est notre
    "filet de sécurité". Si les techniques échouent ou nuisent, on garde
    au minimum la recherche de base.

    Args:
        question: la question de l'utilisateur.
        use_multi_query: force l'activation (sinon utilise config).
        use_hyde: force l'activation (sinon utilise config).

    Returns:
        Liste de requêtes, la première étant la question originale.
    """
    if use_multi_query is None:
        use_multi_query = config.ENABLE_MULTI_QUERY
    if use_hyde is None:
        use_hyde = config.ENABLE_HYDE

    queries = [question]  # toujours la question originale

    if use_multi_query:
        queries.extend(generate_query_variants(question))

    if use_hyde:
        hypothetical = generate_hypothetical_document(question)
        if hypothetical:
            queries.append(hypothetical)

    return queries
