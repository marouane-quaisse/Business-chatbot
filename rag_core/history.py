"""
=============================================================================
HISTORIQUE CONVERSATIONNEL — sauvegarde, chargement, reformulation
=============================================================================

LE PROBLÈME RÉSOLU
------------------
Avant ce module, chaque question était traitée de façon ISOLÉE :

    Vous : "Quel est le délai de retour ?"
    Bot  : "30 jours calendaires."
    Vous : "Et pour les DOM-TOM ?"        ← question de suivi
    Bot  : ❌ ne comprend pas "Et pour..." (aucun contexte)

Deux problèmes distincts se cachent derrière ce symptôme :

  1. LE LLM ne voit pas les messages précédents → il ne peut pas comprendre
     les références comme "et pour...", "dans ce cas", "et si je...".

  2. LE RETRIEVAL cherche avec une question incomplète. Même si le LLM
     comprenait, la recherche vectorielle partirait avec "Et pour les
     DOM-TOM ?" et ne retrouverait AUCUN document pertinent.

C'est pourquoi ce module fait DEUX choses :

  - `load_history()`   → fournit les messages précédents au LLM
  - `reformulate()`    → transforme la question de suivi en question
                         AUTONOME, utilisée pour la recherche

EXEMPLE DE REFORMULATION
------------------------
    Historique : "Quel est le délai de retour ?" / "30 jours."
    Question   : "Et pour les DOM-TOM ?"
    → Reformulée : "Quel est le délai de retour pour les DOM-TOM ?"

C'est cette version reformulée qui part dans la recherche vectorielle.

MODE ANONYME
------------
Si `user_id` est None, RIEN n'est écrit en base. L'historique est alors
fourni par le client (navigateur) et transmis à chaque requête.
Les fonctions de ce module acceptent donc un historique "externe".
=============================================================================
"""

import json

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from models import ChatSession, Message

# =============================================================================
# 1. SAUVEGARDE
# =============================================================================
def save_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    sources: list[str] | None = None,
    latency_ms: int | None = None,
) -> Message:
    """
    Enregistre un message dans la base.

    Args:
        db: session SQLAlchemy.
        session_id: identifiant de la conversation.
        role: "user" ou "assistant".
        content: le texte du message.
        sources: documents cités (uniquement pour les réponses).
        latency_ms: temps de réponse en millisecondes.

    Returns:
        Le message créé.
    """
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        # On sérialise la liste en JSON : SQLite n'a pas de type tableau.
        sources=json.dumps(sources, ensure_ascii=False) if sources else None,
        latency_ms=latency_ms,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def ensure_session(db: Session, session_id: str, user_id: int | None, first_message: str = "") -> ChatSession:
    """
    Récupère une conversation existante, ou la crée si elle n'existe pas.

    POURQUOI "ensure" ET PAS "create" ?
    Le `session_id` est généré par le NAVIGATEUR. L'API ne sait donc pas
    si la conversation existe déjà. Plutôt que d'imposer un appel
    "crée-moi une session" avant chaque première question, on crée la
    session à la volée au premier message. C'est un aller-retour en moins.

    Le titre reprend le premier message de l'utilisateur (tronqué) : c'est
    ce qui s'affichera dans la liste des conversations.
    """
    session = db.get(ChatSession, session_id)
    if session is not None:
        return session

    title = first_message.strip()[:80] or "Nouvelle conversation"
    session = ChatSession(id=session_id, user_id=user_id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


# =============================================================================
# 2. CHARGEMENT
# =============================================================================
def load_history(
    db: Session,
    session_id: str,
    max_messages: int | None = None,
) -> list[tuple[str, str]]:
    """
    Charge les derniers messages d'une conversation.

    Args:
        db: session SQLAlchemy.
        session_id: identifiant de la conversation.
        max_messages: nombre maximum de messages à charger
                      (défaut : config.HISTORY_MAX_TURNS).

    Returns:
        Liste de tuples (role, content), du plus ancien au plus récent.
        Exemple : [("user", "Quel est le délai ?"), ("assistant", "30 jours.")]

    POURQUOI LES DERNIERS MESSAGES ET PAS TOUS ?
    Le contexte du LLM est limité et coûteux. Les échanges très anciens
    apportent peu et diluent l'information récente. On garde donc une
    fenêtre glissante (6 messages par défaut = 3 échanges).
    """
    max_messages = max_messages or config.HISTORY_MAX_TURNS

    # On trie par id DÉCROISSANT pour prendre les plus récents, puis on
    # remet dans l'ordre chronologique avant de renvoyer.
    rows = db.scalars(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.id.desc())
        .limit(max_messages)
    ).all()

    return [(m.role, m.content) for m in reversed(rows)]


def to_langchain_messages(history: list[tuple[str, str]]) -> list:
    """
    Convertit l'historique en objets de messages LangChain.

    C'est ce format qu'attend ChatPromptTemplate.from_messages() :
        [HumanMessage("..."), AIMessage("..."), ...]
    """
    messages = []
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


# =============================================================================
# 3. REFORMULATION DE LA QUESTION
# =============================================================================
# C'est LA pièce essentielle pour que les questions de suivi fonctionnent
# en RAG. Sans elle, "Et pour les DOM-TOM ?" ne retrouve aucun document.
REFORMULATION_PROMPT = """Tu reformules une question de suivi en une question AUTONOME.

Voici la conversation précédente :
{history}

Nouvelle question de l'utilisateur : {question}

Règles :
- Si la nouvelle question est déjà complète et compréhensible seule, renvoie-la TELLE QUELLE.
- Sinon, remplace les références implicites ("et pour...", "dans ce cas", "celui-ci")
  par les éléments concrets de la conversation.
- Garde le MÊME sens et la MÊME langue.
- Réponds UNIQUEMENT avec la question reformulée, sans préambule ni explication.

Question reformulée :"""


def reformulate(question: str, history: list[tuple[str, str]]) -> str:
    """
    Transforme une question de suivi en question autonome.

    Args:
        question: la question brute de l'utilisateur.
        history: les messages précédents [(role, content), ...].

    Returns:
        La question reformulée. En cas d'échec ou d'historique vide,
        renvoie la question ORIGINALE (dégradation gracieuse).

    POURQUOI C'EST IMPORTANT
    ------------------------
    Le retrieval compare la question aux chunks par similarité sémantique.
    Une question incomplète ("Et pour les DOM-TOM ?") a un embedding pauvre
    et ne matche rien. La version reformulée ("Quel est le délai de retour
    pour les DOM-TOM ?") matche correctement.

    COÛT
    ----
    1 appel LLM supplémentaire (modèle auxiliaire rapide), uniquement
    lorsqu'il y a un historique. Sur la première question, aucun appel.
    """
    # Pas d'historique → rien à reformuler, la question est déjà autonome.
    if not history:
        return question

    if not config.ENABLE_REFORMULATION:
        return question

    # On formate l'historique en texte lisible pour le LLM.
    history_text = "\n".join(
        f"{'Utilisateur' if role == 'user' else 'Assistant'} : {content}"
        for role, content in history
    )

    try:
        # Import local : évite un import circulaire au chargement du module.
        from query_transform import get_aux_llm

        prompt = REFORMULATION_PROMPT.format(history=history_text, question=question)
        response = get_aux_llm().invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip().strip('"').strip()

        # Garde-fou : si le LLM renvoie quelque chose d'aberrant (vide, trop
        # long, ou manifestement hors sujet), on garde la question originale.
        if not text or len(text) > 500:
            return question

        return text
    except Exception as exc:  # noqa: BLE001
        # Une panne de reformulation ne doit JAMAIS casser le chat.
        print(f"[history] Reformulation échouée : {exc}")
        return question


# =============================================================================
# 4. GESTION DES CONVERSATIONS
# =============================================================================
def list_sessions(db: Session, user_id: int) -> list[ChatSession]:
    """Liste les conversations d'un utilisateur, de la plus récente à la plus ancienne."""
    return list(
        db.scalars(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
        ).all()
    )


def get_session_messages(
    db: Session,
    session_id: str,
    user_id: int,
    limit: int | None = None,
    offset: int = 0,
) -> list[Message]:
    """
    Charge les messages d'une conversation, par pages.

    C'est ce qui alimente le bouton "Voir plus" : on charge les 20 derniers
    messages, puis 20 de plus à chaque clic (offset += 20).

    ⚠️ SÉCURITÉ : on vérifie que la conversation appartient bien à
    l'utilisateur. Sans ce contrôle, n'importe qui pourrait lire les
    conversations des autres en devinant un session_id.
    """
    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != user_id:
        return []

    limit = limit or config.HISTORY_PAGE_SIZE

    # On trie par id décroissant (les plus récents d'abord) pour la pagination,
    # puis on remet dans l'ordre chronologique pour l'affichage.
    rows = db.scalars(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()

    return list(reversed(rows))


def delete_session(db: Session, session_id: str, user_id: int) -> bool:
    """
    Supprime une conversation et tous ses messages.

    Returns:
        True si la suppression a eu lieu, False si la conversation
        n'existe pas ou n'appartient pas à l'utilisateur.
    """
    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != user_id:
        return False

    # cascade="all, delete-orphan" (voir models.py) supprime les messages liés.
    db.delete(session)
    db.commit()
    return True
