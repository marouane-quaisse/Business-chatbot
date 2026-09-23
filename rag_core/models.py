"""
=============================================================================
MODÈLES DE DONNÉES — les tables de la base
=============================================================================

TROIS TABLES, TROIS RÔLES
-------------------------
    users          → qui utilise le chatbot (email + mot de passe hashé)
    auth_tokens    → les jetons de connexion (un par session de navigateur)
    chat_sessions  → une conversation (regroupe plusieurs messages)
    messages       → un message individuel (question ou réponse)
    documents      → un fichier uploadé par un utilisateur (métadonnées)

RELATIONS
---------
    User 1 ──── N ChatSession 1 ──── N Message
    User 1 ──── N AuthToken
    User 1 ──── N Document

POURQUOI SÉPARER "SESSION" ET "MESSAGE" ?
-----------------------------------------
Une session = une conversation complète (comme un fil dans ChatGPT).
Un message = un tour de parole dans cette conversation.

Cette séparation permet :
  - d'afficher la liste des conversations dans une sidebar
  - de charger les messages par pages ("Voir plus")
  - de calculer des analytics par conversation (latence, sources...)

⚠️ NOTE SUR LE MODE ANONYME
---------------------------
Un utilisateur non authentifié peut utiliser le chatbot, mais RIEN n'est
écrit en base : son historique vit uniquement dans le navigateur.
La colonne `user_id` est donc nullable, mais en pratique les sessions
anonymes ne sont jamais persistées (voir history.py).

POURQUOI UNE TABLE "documents" ET PAS LES FICHIERS EN BASE ?
------------------------------------------------------------
On stocke les MÉTADONNÉES en base et les OCTETS sur disque. C'est la
pratique standard (et ce que font tous les SaaS) :

  - la base reste légère et rapide à sauvegarder
  - les fichiers se lisent en flux (streaming), sans charger 10 Mo en RAM
  - on peut ouvrir un fichier à la main pour déboguer
  - on sait QUI a uploadé QUOI et QUAND (impossible avec un simple dossier)

Le fichier lui-même vit dans ressources/user_{id}/, et `stored_path` garde
le chemin relatif pour le retrouver.
=============================================================================
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def utcnow() -> datetime:
    """
    Retourne l'heure actuelle en UTC.

    ⚠️ On utilise timezone.utc explicitement : datetime.utcnow() est
    déprécié en Python 3.12+ et renvoie une date "naïve" (sans fuseau),
    ce qui provoque des bugs de comparaison plus tard.
    """
    return datetime.now(timezone.utc)


# =============================================================================
# 1. UTILISATEUR
# =============================================================================
class User(Base):
    """
    Un compte utilisateur.

    Le mot de passe n'est JAMAIS stocké en clair : on stocke un hash bcrypt
    (voir auth.py). Même en cas de fuite de la base, les mots de passe
    restent inexploitables.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Relations : permet d'écrire user.sessions ou user.tokens.
    # cascade="all, delete-orphan" : supprimer un user supprime ses données.
    sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    tokens: Mapped[list["AuthToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


# =============================================================================
# 2. TOKEN D'AUTHENTIFICATION
# =============================================================================
class AuthToken(Base):
    """
    Un jeton de connexion (équivalent simplifié d'un JWT).

    POURQUOI PAS UN JWT ?
    Un JWT est auto-porteur (signé) mais ne peut pas être révoqué avant
    expiration. Ici, le token est stocké en base : on peut donc le supprimer
    à tout moment (déconnexion immédiate, bannissement...).

    Le token est une chaîne aléatoire de 32 octets (voir auth.py).
    """

    __tablename__ = "auth_tokens"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)

    user: Mapped["User"] = relationship(back_populates="tokens")


# =============================================================================
# 3. CONVERSATION
# =============================================================================
class ChatSession(Base):
    """
    Une conversation complète.

    `id` est un UUID (et non un entier auto-incrémenté) car il est généré
    côté client : le navigateur crée un identifiant, l'envoie à l'API, et
    la session est créée à la volée au premier message. Cela évite un
    aller-retour supplémentaire ("crée-moi une session").

    `title` reprend le premier message de l'utilisateur : c'est ce qui
    s'affiche dans la liste des conversations.
    """

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # Nullable : une session anonyme n'a pas d'utilisateur (et n'est pas persistée).
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(200), default="Nouvelle conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped["User | None"] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        # Les messages sont presque toujours lus dans l'ordre chronologique.
        order_by="Message.id",
    )


# =============================================================================
# 4. MESSAGE
# =============================================================================
class Message(Base):
    """
    Un message dans une conversation.

    role = "user"      → la question posée
    role = "assistant" → la réponse du bot

    `sources` stocke la liste des documents cités, sérialisée en JSON
    (ex: '["politique_retour.md"]'). On garde du texte brut plutôt qu'un
    type JSON natif pour rester compatible SQLite ET PostgreSQL.

    `latency_ms` mesure le temps de réponse : c'est la base des analytics
    de performance prévues dans le plan.
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


# =============================================================================
# 5. DOCUMENT
# =============================================================================
class Document(Base):
    """
    Un fichier uploadé par un utilisateur.

    ⚠️ NE PAS CONFONDRE avec `langchain_core.documents.Document`, qui
    représente un CHUNK de texte en mémoire. Ici, il s'agit de la ligne
    en base qui décrit un FICHIER sur disque.

    POURQUOI CETTE TABLE ?
    Sans elle, impossible de savoir qui a uploadé quoi : le dossier
    ressources/ est partagé et anonyme. Cette table apporte :

      - l'ISOLATION : chaque utilisateur ne voit que ses documents
      - la TRAÇABILITÉ : qui, quand, quelle taille, combien de chunks
      - la ROBUSTESSE : si un fichier est supprimé à la main sur le disque,
        on peut détecter l'incohérence entre la base et le dossier

    `stored_path` est un chemin RELATIF à ressources/ (ex: "user_3/faq.md").
    On ne stocke jamais de chemin absolu : le projet doit rester portable
    d'une machine à l'autre.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    # Nom affiché à l'utilisateur (ex: "faq.md").
    filename: Mapped[str] = mapped_column(String(255))
    # Chemin relatif à RESSOURCES_DIR (ex: "user_3/faq.md").
    stored_path: Mapped[str] = mapped_column(String(512))
    size: Mapped[int] = mapped_column(Integer, default=0)
    chunks_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped["User"] = relationship(back_populates="documents")
