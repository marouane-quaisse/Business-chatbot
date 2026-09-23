"""
=============================================================================
MIGRATION — attribuer les documents existants à un utilisateur
=============================================================================

CONTEXTE
--------
Avant l'isolation multi-utilisateur, tous les documents vivaient à plat
dans ressources/ et appartenaient à personne. Ce script les attribue à
un utilisateur précis et les déplace dans son dossier.

CE QUE FAIT LE SCRIPT
---------------------
  1. Trouve l'utilisateur cible (par email).
  2. Déplace chaque fichier de ressources/ vers ressources/user_{id}/.
  3. Crée la ligne correspondante dans la table `documents`.
  4. Réindexe les chunks avec la métadonnée `user_id`.

⚠️ Le script est IDEMPOTENT : le relancer ne crée pas de doublons (les
documents déjà enregistrés sont ignorés).

USAGE
-----
    python migrate_documents.py marouane.quaisse@utbm.fr
=============================================================================
"""

import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import select

import config
from database import SessionLocal, init_db
from documents import get_document, register_document, user_dir
from ingestion import SUPPORTED_EXTENSIONS, load_documents_from_folder, split_documents
from models import Document, User
from vectorstore import delete_user_chunks, index_documents


def migrate(email: str) -> None:
    """Attribue tous les documents à plat de ressources/ à l'utilisateur `email`."""
    init_db()
    db = SessionLocal()

    try:
        # --- 1. Trouver l'utilisateur cible ---
        user = db.scalars(select(User).where(User.email == email)).first()
        if user is None:
            print(f"❌ Aucun utilisateur avec l'email « {email} ».")
            print("   Créez d'abord le compte via l'interface (inscription).")
            return

        print(f"✓ Utilisateur trouvé : {user.email} (id={user.id})")

        # --- 2. Lister les fichiers à plat (hors dossiers user_*) ---
        root = config.RESSOURCES_DIR
        candidates = [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not candidates:
            print("ℹ️  Aucun document à plat à migrer.")
        else:
            print(f"→ {len(candidates)} document(s) à migrer vers user_{user.id}/")

        # --- 3. Déplacer chaque fichier + créer la ligne en base ---
        for path in candidates:
            if get_document(db, user.id, path.name) is not None:
                print(f"   • {path.name} : déjà enregistré, ignoré.")
                continue

            # On copie via register_document (qui gère la copie + la base),
            # puis on supprime l'original resté à la racine.
            register_document(db, user.id, path.name, path, chunks_count=0)
            path.unlink()
            print(f"   • {path.name} → user_{user.id}/{path.name}")

        # --- 4. Réindexation de l'espace de l'utilisateur ---
        documents = load_documents_from_folder(user_dir(user.id))
        for doc in documents:
            doc.metadata["user_id"] = user.id

        chunks = split_documents(documents) if documents else []
        delete_user_chunks(user.id)
        if chunks:
            index_documents(chunks)

        # Mise à jour du compteur de chunks par document.
        counts = Counter(chunk.metadata.get("source", "") for chunk in chunks)
        for document in db.scalars(
            select(Document).where(Document.user_id == user.id)
        ).all():
            document.chunks_count = counts.get(document.filename, 0)
        db.commit()

        print(f"\n✅ Migration terminée : {len(documents)} document(s), {len(chunks)} chunk(s).")

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python migrate_documents.py <email>")
        sys.exit(1)
    migrate(sys.argv[1])
