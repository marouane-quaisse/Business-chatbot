"""
Version LangChain de l'agent IA (équivalent de Agent.py).

Installation :
    ./.venv/Scripts/python.exe -m pip install langchain langchain-groq

Lancement :
    ./.venv/Scripts/python.exe Agent_langchain.py
"""

import datetime
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()


# 1. Configuration du modèle Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# 2. Définition des outils avec le décorateur @tool
#    La docstring sert de description au LLM (comme avec Pydantic-AI).
@tool
def get_current_time() -> str:
    """Retourne la date et l'heure actuelles."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Évalue une expression mathématique."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Erreur de calcul : {e}"


@tool
def save_note(content: str) -> str:
    """Enregistre une note dans un fichier texte local."""
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(content + "\n")
    return "Note sauvegardée avec succès."


@tool
def read_notes() -> str:
    """Lit les notes stockées dans le fichier texte local."""
    try:
        with open("notes.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Aucune note trouvée."


tools = [get_current_time, calculate, save_note, read_notes]


# 3. Création de l'agent (LangChain 1.x utilise create_agent, basé sur LangGraph)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "Vous êtes un assistant IA efficace. Vous pouvez utiliser vos outils "
        "pour gérer des notes, faire des calculs ou donner la date et l'heure."
    ),
)


# 4. Boucle principale
def main():
    print("🤖 Agent IA (LangChain + Groq) prêt ! (Tapez 'quit' ou 'exit' pour quitter)\n")
    messages = []

    while True:
        user_input = input("Vous : ")
        if user_input.lower() in ["quit", "exit"]:
            print("Au revoir !")
            break

        # Ajout du message utilisateur à l'historique
        messages.append(HumanMessage(content=user_input))

        # Exécution de l'agent
        result = agent.invoke({"messages": messages})

        # Récupération de l'historique complet (l'agent ajoute ses réponses)
        messages = result["messages"]

        # Le dernier message est la réponse finale de l'agent
        print(f"\nAgent : {messages[-1].content}\n")


if __name__ == "__main__":
    main()
