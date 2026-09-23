"""
=============================================================================
API REST — EXPOSER LE RAG COMME UN SERVICE WEB
=============================================================================

POURQUOI CE FICHIER ?
---------------------
Jusqu'ici, le RAG ne se parlait qu'en ligne de commande (chat.py, eval_*.py).
Pour en faire un VRAI PRODUIT, il faut exposer ses capacités via HTTP, afin
qu'une interface web (ou une app mobile, ou un client externe) puisse l'utiliser.

C'est exactement le principe d'un SaaS : le "core" (le RAG) tourne sur un
serveur, et les utilisateurs s'y connectent via une API.

ENDPOINTS EXPOSÉS
-----------------
  GET  /                  → l'interface de chat (fichier HTML)
  GET  /api/status        → état de la base + configuration active
  POST /api/chat          → question → réponse complète (JSON)
  POST /api/chat/stream   → question → réponse EN STREAMING (SSE)
  POST /api/upload        → ajouter un document à son espace
  GET  /api/documents     → lister SES documents
  GET  /api/documents/{name}/download → télécharger un de ses documents
  DELETE /api/documents/{name} → supprimer un de ses documents (réindexe)
  POST /api/reindex       → réindexer SES documents

ISOLATION MULTI-UTILISATEUR
---------------------------
Chaque utilisateur possède son propre espace :

    ressources/user_{id}/     → ses fichiers
    table `documents`         → ses métadonnées (qui, quand, taille)
    métadonnée `user_id`      → sur chaque chunk Chroma

Le retrieval filtre sur `user_id` : un utilisateur ne peut donc JAMAIS
recevoir d'extraits des documents d'un autre, même si la collection
vectorielle est physiquement partagée.

POURQUOI LE STREAMING (SSE) ?
-----------------------------
Un LLM met 1 à 3 secondes à rédiger une réponse complète. Sans streaming,
l'utilisateur voit un écran figé puis le texte d'un coup. Avec le streaming,
il voit les mots apparaître progressivement → la latence PERÇUE chute
énormément. C'est ce que font ChatGPT, Claude, etc.

Note technique : contrairement à un POST classique, on renvoie ici un flux
continu de texte. Le format utilisé est "Server-Sent Events" (SSE), un
standard simple : chaque ligne commence par `data: `.

LANCEMENT
---------
Depuis la racine du projet :

    .venv/Scripts/python.exe -m uvicorn api:app --port 8000 \
        --app-dir rag_core

Puis ouvrir http://localhost:8000
=============================================================================
"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

import config
from auth import (
    authenticate,
    create_token,
    create_user,
    get_current_user,
    get_current_user_optional,
    revoke_token,
)
from database import get_db, init_db
from documents import (
    delete_document as delete_user_document,
    get_document,
    list_documents as list_user_documents,
    load_user_documents,
    register_document,
    resolve_document_path,
    user_dir,
)
from history import (
    delete_session,
    ensure_session,
    get_session_messages,
    list_sessions,
    load_history,
    reformulate,
    save_message,
)
from ingestion import SUPPORTED_EXTENSIONS, load_document, split_documents
from models import User
from rag import answer, stream_answer
from vectorstore import (
    collection_count,
    delete_document_chunks,
    delete_user_chunks,
    index_documents,
)

# =============================================================================
# 1. APPLICATION FASTAPI
# =============================================================================
app = FastAPI(
    title="RAG Chatbot API",
    description="API d'un chatbot RAG sur documents d'entreprise.",
    version="1.0.0",
)

# Dossier contenant l'interface web (index.html, etc.).
WEB_DIR = Path(__file__).resolve().parent / "web"


@app.on_event("startup")
def on_startup() -> None:
    """
    Crée les tables de la base au démarrage (si elles n'existent pas).

    C'est l'équivalent d'une migration, en beaucoup plus simple : SQLAlchemy
    crée les tables manquantes et ne touche jamais aux données existantes.
    """
    init_db()


# =============================================================================
# 2. SCHÉMAS DE DONNÉES (validation automatique par Pydantic)
# =============================================================================
# Pydantic valide automatiquement les requêtes entrantes. Si un client envoie
# un JSON mal formé, FastAPI renvoie une erreur 422 claire, sans que le code
# du RAG ne soit jamais exécuté. C'est une protection GRATUITE.
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Question de l'utilisateur")
    k: int = Field(default=config.TOP_K, ge=1, le=20, description="Nombre de chunks à récupérer")
    session_id: str | None = Field(
        default=None,
        max_length=36,
        description="Identifiant de la conversation (généré par le navigateur).",
    )
    history: list[tuple[str, str]] | None = Field(
        default=None,
        description=(
            "Historique envoyé par le client. Utilisé UNIQUEMENT en mode anonyme "
            "(en mode authentifié, l'historique est lu depuis la base)."
        ),
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    latency: float


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class TokenResponse(BaseModel):
    token: str
    email: str


class SessionSummary(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: int


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[str]
    latency_ms: int | None
    created_at: str


# =============================================================================
# 3. AUTHENTIFICATION
# =============================================================================
@app.post("/api/auth/signup", response_model=TokenResponse)
def signup(request: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """
    Crée un compte et renvoie immédiatement un token de connexion.

    L'utilisateur n'a donc pas besoin de se reconnecter après l'inscription.
    """
    try:
        user = create_user(db, request.email, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = create_token(db, user)
    return TokenResponse(token=token, email=user.email)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(request: SignupRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """
    Vérifie les identifiants et renvoie un token.

    ⚠️ Message d'erreur volontairement générique : on ne révèle pas si
    c'est l'email ou le mot de passe qui est incorrect (anti-énumération).
    """
    user = authenticate(db, request.email, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect.")

    token = create_token(db, user)
    return TokenResponse(token=token, email=user.email)


@app.post("/api/auth/logout")
def logout(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> dict:
    """
    Révoque le token courant (déconnexion immédiate côté serveur).

    ⚠️ `authorization` DOIT être déclaré avec `Header(...)` : sans cela,
    FastAPI l'interprète comme un paramètre de query string et le token
    envoyé dans l'en-tête HTTP n'est jamais lu — la déconnexion serait
    silencieusement sans effet.
    """
    if authorization:
        revoke_token(db, authorization.removeprefix("Bearer ").strip())
    return {"status": "ok"}


@app.get("/api/auth/me")
def me(user: User = Depends(get_current_user)) -> dict:
    """Renvoie les informations de l'utilisateur connecté."""
    return {"id": user.id, "email": user.email, "created_at": user.created_at.isoformat()}


# =============================================================================
# 4. ENDPOINTS D'INFORMATION
# =============================================================================
@app.get("/api/status")
def get_status() -> dict:
    """
    Retourne l'état du service : nombre de chunks indexés + config active.

    L'UI s'en sert au chargement pour afficher "146 documents indexés" et
    pour montrer quelles optimisations sont activées.
    """
    return {
        "status": "ok",
        "chunks_indexed": collection_count(),
        "config": {
            "generator_model": config.GROQ_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
            "multiquery": config.ENABLE_MULTI_QUERY,
            "hyde": config.ENABLE_HYDE,
            "rerank": config.ENABLE_RERANK,
            "top_k": config.TOP_K,
            "reformulation": config.ENABLE_REFORMULATION,
        },
    }


# =============================================================================
# 5. CHAT — VERSION SIMPLE (tout d'un coup)
# =============================================================================
def _prepare_context(
    request: ChatRequest,
    user: User | None,
    db: Session,
) -> tuple[str, list[tuple[str, str]]]:
    """
    Prépare le contexte conversationnel avant d'appeler le RAG.

    C'est ici que se joue la différence entre les deux modes :

      AUTHENTIFIÉ : l'historique est lu depuis la BASE (persistant).
      ANONYME     : l'historique vient du CLIENT (éphémère, non sauvegardé).

    Returns:
        (question_reformulée, historique)
    """
    if user is not None and request.session_id:
        # --- Mode authentifié : historique depuis la base ---
        history = load_history(db, request.session_id)
    else:
        # --- Mode anonyme : historique fourni par le navigateur ---
        # On le tronque pour éviter qu'un client malveillant n'envoie
        # 10 000 messages et ne fasse exploser le coût en tokens.
        history = (request.history or [])[-config.HISTORY_MAX_TURNS:]

    # Reformulation : transforme "Et pour les DOM-TOM ?" en question autonome.
    # Indispensable pour que le retrieval trouve les bons documents.
    search_question = reformulate(request.question, history)

    return search_question, history


def _persist_exchange(
    db: Session,
    user: User | None,
    request: ChatRequest,
    answer_text: str,
    sources: list[str],
    latency_ms: int,
) -> None:
    """
    Sauvegarde l'échange (question + réponse) en base.

    ⚠️ RIEN n'est écrit en mode anonyme : c'est la promesse faite à
    l'utilisateur ("vous perdrez vos conversations"). On ne stocke donc
    aucune donnée le concernant.
    """
    if user is None or not request.session_id:
        return

    # Crée la conversation à la volée si c'est le premier message.
    ensure_session(db, request.session_id, user.id, first_message=request.question)

    save_message(db, request.session_id, "user", request.question)
    save_message(db, request.session_id, "assistant", answer_text, sources, latency_ms)


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Répond à une question et renvoie la réponse COMPLÈTE d'un seul coup.

    C'est l'endpoint le plus simple à consommer (utile pour des tests, des
    scripts, ou un client qui n'a pas besoin de streaming).
    """
    import time

    start = time.perf_counter()

    search_question, history = _prepare_context(request, user, db)
    result = answer(
        search_question,
        k=request.k,
        history=history,
        # Isolation : on ne cherche que dans les documents de l'utilisateur.
        user_id=user.id if user is not None else None,
    )

    latency = time.perf_counter() - start
    _persist_exchange(db, user, request, result["answer"], result["sources"], int(latency * 1000))

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency=latency,
    )


# =============================================================================
# 6. CHAT — VERSION STREAMING (SSE)
# =============================================================================
@app.post("/api/chat/stream")
def chat_stream(
    request: ChatRequest,
    user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Répond à une question en envoyant le texte AU FUR ET À MESURE.

    FORMAT SSE : chaque événement est une ligne `data: {json}\\n\\n`.
      data: {"type": "sources", "data": ["politique_retour.md"]}
      data: {"type": "token",   "data": "Le délai"}
      data: {"type": "token",   "data": " de retour est"}
      data: {"type": "done",    "data": {"latency": 1.24}}

    Le client lit ce flux et affiche le texte progressivement.
    """
    # ⚠️ IMPORTANT : on prépare le contexte AVANT de créer le générateur.
    # Une fois le StreamingResponse renvoyé, la session de base de données
    # est fermée par FastAPI (la dépendance get_db est libérée). On ne peut
    # donc plus l'utiliser dans le générateur.
    search_question, history = _prepare_context(request, user, db)

    def event_generator():
        # Accumulateurs : on reconstitue la réponse complète pour la sauvegarder.
        full_answer: list[str] = []
        collected_sources: list[str] = []
        latency_ms = 0

        # Chaque événement renvoyé par stream_answer() est converti en SSE.
        for event_type, payload in stream_answer(
            search_question,
            k=request.k,
            history=history,
            # Isolation : on ne cherche que dans les documents de l'utilisateur.
            user_id=user.id if user is not None else None,
        ):
            if event_type == "token":
                full_answer.append(payload)
            elif event_type == "sources":
                collected_sources = payload
            elif event_type == "done":
                latency_ms = int(payload.get("latency", 0) * 1000)

            message = json.dumps({"type": event_type, "data": payload}, ensure_ascii=False)
            yield f"data: {message}\n\n"

        # --- Sauvegarde APRÈS la fin du flux ---
        # On ouvre une NOUVELLE session de base : celle de la requête est
        # déjà fermée à ce stade.
        if user is not None and request.session_id and full_answer:
            from database import SessionLocal

            db2 = SessionLocal()
            try:
                _persist_exchange(
                    db2, user, request, "".join(full_answer), collected_sources, latency_ms
                )
            except Exception as exc:  # noqa: BLE001
                # Un échec de sauvegarde ne doit pas casser la réponse déjà envoyée.
                print(f"[api] Sauvegarde de l'échange échouée : {exc}")
            finally:
                db2.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # Ces en-têtes empêchent la mise en cache du flux par le navigateur
            # ou un proxy intermédiaire.
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# 7. CONVERSATIONS (sessions)
# =============================================================================
@app.get("/api/sessions", response_model=list[SessionSummary])
def get_sessions(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SessionSummary]:
    """
    Liste les conversations de l'utilisateur connecté.

    ⚠️ On filtre TOUJOURS par user.id : un utilisateur ne peut voir que
    ses propres conversations.
    """
    sessions = list_sessions(db, user.id)
    return [
        SessionSummary(
            id=s.id,
            title=s.title,
            created_at=s.created_at.isoformat(),
            message_count=len(s.messages),
        )
        for s in sessions
    ]


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageOut])
def get_messages(
    session_id: str,
    limit: int = Query(default=config.HISTORY_PAGE_SIZE, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[MessageOut]:
    """
    Charge les messages d'une conversation, par pages.

    C'est ce qui alimente le bouton "Voir plus" :
      - 1er chargement : offset=0
      - clic suivant   : offset += limit

    ⚠️ La vérification de propriété est faite dans history.get_session_messages.
    """
    messages = get_session_messages(db, session_id, user.id, limit=limit, offset=offset)

    return [
        MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            sources=json.loads(m.sources) if m.sources else [],
            latency_ms=m.latency_ms,
            created_at=m.created_at.isoformat(),
        )
        for m in messages
    ]


@app.delete("/api/sessions/{session_id}")
def remove_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Supprime une conversation et tous ses messages."""
    if not delete_session(db, session_id, user.id):
        raise HTTPException(status_code=404, detail="Conversation introuvable.")
    return {"status": "ok"}


# =============================================================================
# 8. INGESTION — AJOUTER UN DOCUMENT
# =============================================================================
@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Ajoute un document à l'espace de l'utilisateur connecté.

    Pipeline : fichier reçu → sauvegardé dans ressources/user_{id}/ → découpé
               en chunks → vectorisé (avec la métadonnée user_id) → Chroma.

    ⚠️ L'upload exige désormais une authentification : sans propriétaire
    identifié, impossible de garantir l'isolation des documents.
    """
    suffix = Path(file.filename or "").suffix.lower()

    # On refuse les formats qu'on ne sait pas lire (PDF scanné, .docx, images...).
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté ({suffix}). Formats acceptés : {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # On écrit d'abord dans un fichier temporaire, puis on déplace. Si l'upload
    # échoue en cours de route, on ne laisse pas de fichier à moitié écrit.
    tmp_path = Path(tempfile.mkstemp(suffix=suffix)[1])
    try:
        with tmp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = load_document(tmp_path)
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Le fichier ne contient aucun texte exploitable (PDF scanné ?).",
            )

        # Copie définitive dans le dossier de l'utilisateur + ligne en base.
        filename = Path(file.filename or "document.txt").name
        document = register_document(db, user.id, filename, tmp_path, chunks_count=0)
    finally:
        # Le nettoyage du fichier temporaire ne doit JAMAIS faire échouer
        # l'upload : sous Windows, un handle encore ouvert (antivirus,
        # indexeur) peut bloquer la suppression. On ignore l'erreur.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    # Chunking + indexation, avec la métadonnée user_id sur chaque chunk.
    from langchain_core.documents import Document

    chunks = split_documents(
        [
            Document(
                page_content=text,
                metadata={
                    "source": document.filename,
                    "path": document.stored_path,
                    "user_id": user.id,
                },
            )
        ]
    )

    # On retire d'abord les anciens chunks de CE fichier : ré-uploader un
    # document corrigé ne doit pas laisser traîner la version précédente.
    # ⚠️ On cible le fichier précis, PAS tout l'utilisateur : sinon les
    # autres documents de l'utilisateur seraient effacés de l'index.
    delete_document_chunks(user.id, document.filename)
    added = index_documents(chunks)

    # On met à jour le nombre de chunks réellement indexés.
    document.chunks_count = added
    db.commit()

    return {
        "filename": document.filename,
        "chunks_added": added,
        "chunks_total": collection_count(),
    }


@app.post("/api/reindex")
def reindex(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Réindexe les documents de l'utilisateur connecté.

    ⚠️ On ne touche QUE ses propres chunks : les documents des autres
    utilisateurs restent intacts dans la collection partagée.
    """
    documents = load_user_documents(user.id)
    if not documents:
        raise HTTPException(status_code=400, detail="Aucun document à réindexer.")

    # On ajoute la métadonnée user_id à chaque chunk (indispensable pour
    # que le filtrage au retrieval fonctionne).
    for doc in documents:
        doc.metadata["user_id"] = user.id

    chunks = split_documents(documents)

    # Suppression ciblée puis réindexation : pas de doublons, pas d'impact
    # sur les autres utilisateurs.
    delete_user_chunks(user.id)
    index_documents(chunks)

    return {"documents": len(documents), "chunks_indexed": collection_count()}


@app.get("/api/documents")
def list_documents(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Liste les documents de l'utilisateur connecté.

    ⚠️ On lit la BASE (table `documents`) et non le dossier : c'est elle
    qui sait à QUI appartient chaque fichier. Un simple parcours de
    ressources/ exposerait les documents de tout le monde.
    """
    return [
        {
            "name": doc.filename,
            "size": doc.size,
            "chunks_count": doc.chunks_count,
            "modified_at": doc.created_at.isoformat(),
        }
        for doc in list_user_documents(db, user.id)
    ]


@app.delete("/api/documents/{filename}")
def delete_document(
    filename: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Supprime un document de l'utilisateur puis réindexe son espace.

    ⚠️ `get_document` filtre par user_id : un utilisateur ne peut pas
    supprimer le document d'un autre, même en connaissant son nom.
    """
    document = delete_user_document(db, user.id, filename)
    if document is None:
        raise HTTPException(status_code=404, detail="Document introuvable.")

    # Réindexation de l'espace de l'utilisateur uniquement.
    documents = load_user_documents(user.id)
    for doc in documents:
        doc.metadata["user_id"] = user.id

    chunks = split_documents(documents) if documents else []
    delete_user_chunks(user.id)
    if chunks:
        index_documents(chunks)

    return {
        "deleted": document.filename,
        "documents": len(documents),
        "chunks_indexed": collection_count(),
    }


@app.get("/api/documents/{filename}/download")
def download_document(
    filename: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FileResponse:
    """
    Télécharge un document de l'utilisateur connecté.

    ⚠️ On vérifie d'abord que le document appartient bien à l'utilisateur
    (via la base), puis on construit le chemin sur disque. Sans cette
    double vérification, un utilisateur pourrait télécharger le fichier
    d'un autre en devinant son nom.
    """
    document = get_document(db, user.id, filename)
    if document is None:
        raise HTTPException(status_code=404, detail="Document introuvable.")

    try:
        target = resolve_document_path(user.id, document.filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide.")

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Fichier absent du disque.")

    return FileResponse(
        target,
        filename=document.filename,
        media_type="application/octet-stream",
    )


# =============================================================================
# 9. INTERFACE WEB
# =============================================================================
# On sert le dossier web/ en statique (CSS, JS, images) et on renvoie
# index.html à la racine.


class NoCacheStaticFiles(StaticFiles):
    """
    Sert les fichiers statiques en forçant leur revalidation.

    Sans en-tête `Cache-Control`, le navigateur applique une heuristique de
    fraîcheur et peut réutiliser un `style.css` ou un `app.js` périmé pendant
    plusieurs minutes après une modification — l'utilisateur voit alors
    l'ancienne interface malgré un rechargement (F5).

    `no-cache` n'interdit pas la mise en cache : il impose une revalidation
    conditionnelle (ETag / Last-Modified). Le serveur répond donc 304 tant que
    le fichier n'a pas changé, et 200 avec la nouvelle version sinon.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


if WEB_DIR.exists():
    app.mount(
        "/static",
        NoCacheStaticFiles(directory=str(WEB_DIR)),
        name="static",
    )


@app.get("/")
def serve_ui() -> FileResponse:
    """Renvoie l'interface de chat."""
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Interface web introuvable (web/index.html).")
    response = FileResponse(index)
    response.headers["Cache-Control"] = "no-cache"
    return response
