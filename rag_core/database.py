"""
=============================================================================
BASE DE DONNÉES — SQLite + SQLAlchemy
=============================================================================

POURQUOI CE FICHIER ?
---------------------
Jusqu'ici, le projet était STATELESS : chaque question était traitée de façon
isolée, et rien n'était conservé. Impossible donc de :
  - gérer des utilisateurs (inscription / connexion)
  - conserver l'historique d'une conversation
  - afficher les conversations passées

Ce module fournit la brique manquante : une base de données relationnelle.

POURQUOI SQLITE ?
-----------------
  - Zéro installation : c'est un simple FICHIER (app.db)
  - Zéro serveur à lancer (contrairement à PostgreSQL)
  - Parfait pour un MVP et pour le développement local
  - Migrable vers PostgreSQL plus tard SANS changer le code métier,
    car SQLAlchemy abstrait le moteur (il suffit de changer DATABASE_URL)

⚠️ LIMITE : SQLite ne gère pas bien les écritures concurrentes. Pour un vrai
   déploiement multi-utilisateurs, on passera à PostgreSQL (voir Plan.md).

ARCHITECTURE
------------
    database.py  → le moteur + les sessions (ce fichier)
    models.py    → les tables (User, ChatSession, Message)
    auth.py      → l'authentification (hash, tokens)
    history.py   → la logique de l'historique conversationnel
=============================================================================
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL

# =============================================================================
# 1. LE MOTEUR (engine)
# =============================================================================
# L'engine gère la connexion au fichier SQLite.
#
# connect_args={"check_same_thread": False} :
#   Par défaut, SQLite interdit d'utiliser une connexion depuis un thread
#   différent de celui qui l'a créée. FastAPI exécute les endpoints
#   synchrones dans un pool de threads → il FAUT désactiver cette
#   vérification, sinon on obtient l'erreur "SQLite objects created in a
#   thread can only be used in that same thread".
#
#   C'est sans danger ici car SQLAlchemy gère lui-même la sérialisation
#   des accès via son pool de connexions.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,  # passe à True pour voir le SQL généré (debug)
)

# =============================================================================
# 2. LA FABRIQUE DE SESSIONS
# =============================================================================
# Une "session" SQLAlchemy = une unité de travail (transaction).
# On en ouvre une par requête HTTP, puis on la ferme.
#
# autoflush=False : on contrôle explicitement quand les données partent en DB.
# expire_on_commit=False : après un commit, les objets restent lisibles
#   (sinon SQLAlchemy les invalide et recharge tout à la prochaine lecture,
#    ce qui provoque des erreurs si la session est déjà fermée).
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


# =============================================================================
# 3. CLASSE DE BASE DES MODÈLES
# =============================================================================
# Tous les modèles (User, ChatSession, Message) héritent de cette classe.
# SQLAlchemy s'en sert pour connaître la liste des tables à créer.
class Base(DeclarativeBase):
    pass


# =============================================================================
# 4. INITIALISATION
# =============================================================================
def init_db() -> None:
    """
    Crée les tables si elles n'existent pas.

    Appelée UNE FOIS au démarrage de l'API (voir api.py).
    Si le fichier app.db existe déjà, cette fonction ne fait rien :
    elle ne supprime JAMAIS de données.

    ⚠️ On importe models ici (et pas en haut du fichier) pour éviter un
    import circulaire : models.py importe Base depuis ce module.
    """
    import models  # noqa: F401  (l'import suffit à enregistrer les tables)

    Base.metadata.create_all(bind=engine)


# =============================================================================
# 5. DÉPENDANCE FASTAPI
# =============================================================================
def get_db() -> Generator[Session, None, None]:
    """
    Fournit une session de base de données à un endpoint FastAPI.

    Usage dans un endpoint :

        @app.get("/exemple")
        def exemple(db: Session = Depends(get_db)):
            ...

    Le bloc `finally` garantit que la session est TOUJOURS fermée, même si
    l'endpoint lève une exception. Sans ça, on finirait par épuiser le pool
    de connexions.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
