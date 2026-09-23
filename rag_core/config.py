"""
Configuration centrale du projet RAG.

Toutes les valeurs sont surchargeables via des variables d'environnement
(voir .env à la racine du projet).

POURQUOI CE FICHIER ?
---------------------
On centralise ici TOUS les paramètres réglables du RAG. Ainsi, on peut
changer le modèle d'embeddings, la taille des chunks ou le nombre de
résultats récupérés SANS toucher au reste du code. C'est une bonne pratique
qui rend le projet facile à ajuster et à tester.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Charge les variables du fichier .env (ex: GROQ_API_KEY) dans l'environnement.
load_dotenv()

# =============================================================================
# 1. CHEMINS DU PROJET
# =============================================================================
# BASE_DIR = dossier où se trouve ce fichier (rag_core/).
# On construit tous les autres chemins à partir de lui pour que le projet
# fonctionne quelle que soit la machine ou le dossier courant.
BASE_DIR = Path(__file__).resolve().parent
RESSOURCES_DIR = BASE_DIR / "ressources"          # dossier des documents à ingérer
VECTOR_DB_DIR = BASE_DIR / "vector_db"            # stockage persistant Chroma

# =============================================================================
# 2. MODÈLES
# =============================================================================
# EMBEDDING_MODEL : transforme un texte en vecteur (liste de nombres).
#   On utilise un modèle LOCAL (sentence-transformers) : gratuit, pas d'appel
#   API, tourne sur ta machine. "all-MiniLM-L6-v2" est petit (~90 Mo) et rapide.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# GROQ_MODEL : le LLM qui rédige la réponse finale à partir du contexte trouvé.
#   C'est la partie "G" de RAG (Generation).
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =============================================================================
# 3. CHUNKING (découpage des documents)
# =============================================================================
# Un document entier est trop long pour être envoyé au LLM et pour être
# "matché" précisément. On le découpe donc en morceaux (chunks).
#
# CHUNK_SIZE    : taille d'un chunk en caractères (~800 = environ 150-200 mots).
# CHUNK_OVERLAP : nombre de caractères répétés entre deux chunks voisins.
#                 L'overlap évite de "couper" une idée en deux : si une phrase
#                 importante est à cheval sur deux chunks, elle apparaît
#                 entièrement dans au moins l'un des deux.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# =============================================================================
# 4. RETRIEVAL (recherche des chunks pertinents)
# =============================================================================
# TOP_K : combien de chunks on récupère pour répondre à une question.
#   Trop peu → on rate l'info. Trop → on noie le LLM et on paie plus cher.
#   4 est un bon compromis pour démarrer.
TOP_K = int(os.getenv("TOP_K", "4"))

# =============================================================================
# 5. COLLECTION CHROMA
# =============================================================================
# Une "collection" est comme une table dans une base de données : c'est là
# que sont rangés tous nos vecteurs. Plus tard (multi-tenant), on pourra
# utiliser une collection par business pour isoler les données.
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "business_docs")

# =============================================================================
# 6. OPTIMISATIONS DU RETRIEVAL (activables/désactivables)
# =============================================================================
# Chaque technique peut être activée indépendamment pour MESURER son impact.
# Mets une valeur à "0" (ou "false") pour la désactiver.
#
#   ENABLE_MULTI_QUERY : génère plusieurs reformulations de la question.
#                        → améliore le RECALL (couvre les synonymes).
#   ENABLE_HYDE        : génère une réponse hypothétique et cherche avec.
#                        → améliore le RECALL (question → forme "document").
#   ENABLE_RERANK      : reclasse les candidats avec un cross-encoder local.
#                        → améliore la PRÉCISION (filtre le bruit).
#
# ⚠️ Sans reranker, activer Multi-Query ET HyDE peut introduire du bruit
#    dans le top-K. Le reranker est le "garde-fou" qui nettoie les candidats.
def _env_bool(name: str, default: bool) -> bool:
    """Lit une variable d'environnement booléenne ('1', 'true', 'yes' = True)."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "oui", "on"}


ENABLE_MULTI_QUERY = _env_bool("ENABLE_MULTI_QUERY", True)
ENABLE_HYDE = _env_bool("ENABLE_HYDE", True)
ENABLE_RERANK = _env_bool("ENABLE_RERANK", True)

# Nombre de reformulations générées par Multi-Query (hors question originale).
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))

# Nombre de candidats récupérés AVANT reranking (on prend large, puis on filtre).
# Ex: on récupère 20 candidats, le reranker garde les TOP_K meilleurs.
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "20"))

# Modèle utilisé pour les tâches auxiliaires (générer variantes / HyDE).
# Un petit modèle rapide suffit et réduit la latence.
# ⚠️ Groq retire régulièrement des modèles : si erreur 404, vérifie la liste
#    disponible via l'API (client.models.list()) et mets à jour cette valeur.
AUX_MODEL = os.getenv("AUX_MODEL", "openai/gpt-oss-20b")

# Modèle de reranking local (cross-encoder, gratuit, pas d'appel API).
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# =============================================================================
# 7. BASE DE DONNÉES (users, sessions, messages)
# =============================================================================
# SQLite : un simple fichier sur disque, aucun serveur à lancer.
# C'est suffisant pour un MVP et migrable vers PostgreSQL plus tard sans
# changer le code (SQLAlchemy abstrait le moteur).
DB_PATH = BASE_DIR / os.getenv("DB_PATH", "app.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Durée de validité d'un token d'authentification (en jours).
TOKEN_EXPIRE_DAYS = int(os.getenv("TOKEN_EXPIRE_DAYS", "30"))

# =============================================================================
# 8. HISTORIQUE CONVERSATIONNEL
# =============================================================================
# Nombre de messages (user + assistant) envoyés au LLM par défaut.
# 6 messages = 3 échanges. Au-delà, le contexte devient long et coûteux
# sans apporter grand-chose à la qualité de la réponse.
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "6"))

# Nombre de messages chargés par page dans l'interface ("Voir plus").
HISTORY_PAGE_SIZE = int(os.getenv("HISTORY_PAGE_SIZE", "20"))

# Reformulation de la question à partir de l'historique.
# Indispensable en RAG : sans ça, une question de suivi comme
# "Et pour les DOM-TOM ?" ne retrouve aucun document pertinent.
ENABLE_REFORMULATION = _env_bool("ENABLE_REFORMULATION", True)
