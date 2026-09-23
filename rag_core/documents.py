"""
=============================================================================
GESTION DES DOCUMENTS — isolation par utilisateur
=============================================================================

LE PROBLÈME RÉSOLU
------------------
Avant ce module, TOUS les utilisateurs partageaient le même dossier
`ressources/` et la même collection Chroma. Conséquence : si Alice
uploadait un document confidentiel, Bob le voyait dans sa liste et le
retrieval pouvait le citer dans ses réponses.

C'est inacceptable pour un SaaS. Ce module apporte l'ISOLATION :

    ressources/
        user_1/
            faq.md
            cgv.md
        user_2/
            contrat.pdf

Chaque utilisateur ne voit QUE son dossier, et le retrieval ne cherche
QUE dans ses chunks (via un filtre sur la métadonnée `user_id`).

DEUX SOURCES DE VÉRITÉ, UN SEUL ÉTAT COHÉRENT
---------------------------------------------
    - Le DISQUE (ressources/user_{id}/) contient les octets des fichiers.
    - La BASE (table `documents`) contient les métadonnées.

Les deux doivent rester synchronisés. C'est pourquoi toute opération
d'écriture (ajout, suppression) passe par ce module, qui met à jour
les deux en même temps.

POURQUOI PAS LES FICHIERS DANS LA BASE ?
----------------------------------------
Voir la note dans models.py : la base stocke les métadonnées, le disque
stocke les octets. Mélanger les deux rend la base lourde, lente à
sauvegarder, et empêche le streaming des téléchargements.
=============================================================================
"""

import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from models import Document

# =============================================================================
# 1. CHEMINS
# =============================================================================
def user_dir(user_id: int) -> Path:
    """
    Retourne le dossier de travail d'un utilisateur, en le créant si besoin.

    On nomme les dossiers `user_{id}` (et non par email) : les emails
    contiennent des caractères problématiques pour les systèmes de
    fichiers (@, espaces, majuscules) et peuvent changer. L'id, lui, est
    stable et sûr.
    """
    path = config.RESSOURCES_DIR / f"user_{user_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_document_path(user_id: int, filename: str) -> Path:
    """
    Construit le chemin d'un document en empêchant toute traversée de
    répertoire (`../../etc/passwd`).

    ⚠️ `filename` peut venir de l'URL : on ne lui fait JAMAIS confiance.
    On le réduit à son nom de base, puis on vérifie que le chemin résolu
    reste bien à l'intérieur du dossier de l'utilisateur.
    """
    safe_name = Path(filename).name
    target = user_dir(user_id) / safe_name

    if not target.resolve().is_relative_to(user_dir(user_id).resolve()):
        raise ValueError("Nom de fichier invalide.")

    return target


# =============================================================================
# 2. LECTURE
# =============================================================================
def list_documents(db: Session, user_id: int) -> list[Document]:
    """
    Liste les documents d'un utilisateur, du plus récent au plus ancien.

    ⚠️ On filtre TOUJOURS par user_id : c'est la garantie d'isolation.
    """
    return list(
        db.scalars(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
        ).all()
    )


def get_document(db: Session, user_id: int, filename: str) -> Document | None:
    """
    Récupère un document par son nom, EN VÉRIFIANT qu'il appartient bien
    à l'utilisateur.

    ⚠️ Le filtre sur user_id est indispensable : sans lui, un utilisateur
    pourrait accéder au document d'un autre en devinant son nom.
    """
    return db.scalars(
        select(Document).where(
            Document.user_id == user_id,
            Document.filename == filename,
        )
    ).first()


# =============================================================================
# 3. ÉCRITURE
# =============================================================================
def register_document(
    db: Session,
    user_id: int,
    filename: str,
    source_path: Path,
    chunks_count: int,
) -> Document:
    """
    Enregistre un document : copie le fichier dans le dossier de
    l'utilisateur puis crée la ligne en base.

    Si un document du même nom existe déjà, il est REMPLACÉ (le fichier
    est écrasé et la ligne mise à jour). C'est le comportement attendu :
    ré-uploader un fichier corrigé doit mettre à jour l'existant, pas
    créer un doublon.

    Args:
        db: session SQLAlchemy.
        user_id: propriétaire du document.
        filename: nom affiché (ex: "faq.md").
        source_path: fichier temporaire à copier.
        chunks_count: nombre de chunks indexés pour ce document.

    Returns:
        La ligne `Document` créée ou mise à jour.
    """
    safe_name = Path(filename).name
    destination = user_dir(user_id) / safe_name

    # Copie physique du fichier dans le dossier de l'utilisateur.
    shutil.copyfile(source_path, destination)

    # Chemin relatif à ressources/ : portable d'une machine à l'autre.
    stored_path = str(destination.relative_to(config.RESSOURCES_DIR))

    existing = get_document(db, user_id, safe_name)
    if existing is not None:
        # Remplacement : on met à jour la ligne existante.
        existing.stored_path = stored_path
        existing.size = destination.stat().st_size
        existing.chunks_count = chunks_count
        db.commit()
        db.refresh(existing)
        return existing

    document = Document(
        user_id=user_id,
        filename=safe_name,
        stored_path=stored_path,
        size=destination.stat().st_size,
        chunks_count=chunks_count,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def delete_document(db: Session, user_id: int, filename: str) -> Document | None:
    """
    Supprime un document : le fichier sur disque ET la ligne en base.

    Returns:
        La ligne supprimée, ou None si le document n'existe pas.
    """
    document = get_document(db, user_id, filename)
    if document is None:
        return None

    # Suppression du fichier physique (s'il existe encore).
    target = config.RESSOURCES_DIR / document.stored_path
    if target.is_file():
        target.unlink()

    db.delete(document)
    db.commit()
    return document


def load_user_documents(user_id: int) -> list:
    """
    Charge les fichiers d'un utilisateur depuis le disque, prêts à être
    découpés en chunks.

    ⚠️ On lit le DISQUE et non la base : c'est le contenu réel des fichiers
    qui compte pour l'indexation. La base ne sert qu'à savoir à qui ils
    appartiennent.

    Returns:
        Liste de `langchain_core.documents.Document`.
    """
    from ingestion import load_documents_from_folder

    directory = user_dir(user_id)
    return load_documents_from_folder(directory)
