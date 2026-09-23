"""
Chat console : reçoit une question, fait le matching RAG et affiche la réponse.

RÔLE DE CE SCRIPT
-----------------
C'est le POINT D'ENTRÉE de la partie "lecture" du RAG. Il ne fait que :
  1. afficher un état de la base (combien de chunks indexés)
  2. lire la question de l'utilisateur
  3. appeler rag.answer() qui fait tout le travail
  4. afficher la réponse + les sources

Toute la logique RAG est dans rag.py : ce fichier n'est qu'une interface.

Usage :
    python chat.py
"""

from rag import answer
from vectorstore import collection_count


def main() -> None:
    print("=" * 60)
    print("💬 CHATBOT RAG (console)")
    print("=" * 60)

    # On prévient l'utilisateur si la base est vide : sans documents,
    # le RAG ne peut rien retrouver.
    count = collection_count()
    if count == 0:
        print("⚠️  La base vectorielle est vide.")
        print("   Lancez d'abord : python ingest.py\n")
    else:
        print(f"📚 {count} chunks indexés et prêts.\n")

    print("Tapez 'quit' ou 'exit' pour quitter.\n")

    # Boucle de conversation
    while True:
        try:
            question = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C ou fin de flux → on sort proprement.
            print("\nAu revoir !")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Au revoir !")
            break

        # --- APPEL AU PIPELINE RAG ---
        # C'est ici que tout se joue : retrieval + génération.
        result = answer(question)

        print(f"\n🤖 Agent : {result['answer']}")
        if result["sources"]:
            print(f"📎 Sources : {', '.join(result['sources'])}")
        print()


if __name__ == "__main__":
    main()
