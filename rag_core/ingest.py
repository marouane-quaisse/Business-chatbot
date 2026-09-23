"""
Script d'ingestion : charge les documents du dossier ressources/,
les découpe en chunks, génère les embeddings et les stocke dans Chroma.

RÔLE DE CE SCRIPT
-----------------
C'est le POINT D'ENTRÉE de la partie "écriture" du RAG. Il orchestre les
modules ingestion.py et vectorstore.py dans le bon ordre :

    ressources/  →  chargement  →  chunking  →  embeddings  →  Chroma

On le lance UNE FOIS (ou à chaque fois qu'on ajoute/modifie des documents).
Ensuite, chat.py peut poser des questions sur ce contenu.

Usage :
    python ingest.py            # indexe (ajoute aux données existantes)
    python ingest.py --reset    # vide la base puis réindexe tout
"""

import argparse
import sys
import time

from config import RESSOURCES_DIR
from ingestion import load_documents_from_folder, split_documents
from vectorstore import collection_count, index_documents


def main() -> None:
    # argparse permet d'ajouter l'option --reset en ligne de commande.
    parser = argparse.ArgumentParser(description="Ingestion des ressources dans le vector store.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Supprime la collection existante avant de réindexer.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("📥 INGESTION DES RESSOURCES")
    print("=" * 60)
    print(f"Dossier source : {RESSOURCES_DIR}\n")

    if not RESSOURCES_DIR.exists():
        print(f"❌ Le dossier {RESSOURCES_DIR} n'existe pas.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # ÉTAPE 1 : CHARGEMENT — lire les fichiers et en extraire le texte
    # -------------------------------------------------------------------------
    print("1️⃣  Chargement des documents...")
    documents = load_documents_from_folder(RESSOURCES_DIR)

    if not documents:
        print("\n❌ Aucun document trouvé. Ajoutez des fichiers .txt, .md ou .pdf dans ressources/.")
        sys.exit(1)

    print(f"\n   → {len(documents)} document(s) chargé(s)")

    # -------------------------------------------------------------------------
    # ÉTAPE 2 : CHUNKING — découper chaque document en morceaux
    # -------------------------------------------------------------------------
    print("\n2️⃣  Découpage en chunks...")
    chunks = split_documents(documents)
    print(f"   → {len(chunks)} chunk(s) créé(s)")

    # -------------------------------------------------------------------------
    # ÉTAPE 3 : EMBEDDINGS + STOCKAGE — vectoriser et écrire dans Chroma
    # -------------------------------------------------------------------------
    print("\n3️⃣  Génération des embeddings et stockage (peut prendre du temps)...")
    start = time.time()
    count = index_documents(chunks, reset=args.reset)
    elapsed = time.time() - start

    print(f"   → {count} chunk(s) indexé(s) en {elapsed:.1f}s")

    # -------------------------------------------------------------------------
    # RÉSUMÉ
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"✅ Ingestion terminée. Total dans la base : {collection_count()} chunks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
