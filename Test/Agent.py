import datetime
import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

# --- Configuration DeepSeek (désactivée, gardée pour référence) ---
# from pydantic_ai.models.openai import OpenAIChatModel
# from pydantic_ai.providers.deepseek import DeepSeekProvider
#
# model = OpenAIChatModel(
#     model_name="deepseek-chat",  # ou "deepseek-reasoner" pour le modèle de raisonnement
#     provider=DeepSeekProvider(api_key=os.getenv("DEEPSEEK_API_KEY")),
# )


# 1. Configuration du modèle Groq via son API
# La clé API est lue depuis la variable d'environnement GROQ_API_KEY
load_dotenv()

model = GroqModel(
    model_name="openai/gpt-oss-120b",  # autres options : "openai/gpt-oss-20b", "qwen/qwen3.6-27b"
    provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")),
)


# 2. Définition des outils (fonctions Python que l'agent peut exécuter)
def get_current_time() -> str:
    """Retourne la date et l'heure actuelles."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Évalue une expression mathématique."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Erreur de calcul : {e}"


def save_note(content: str) -> str:
    """Enregistre une note dans un fichier texte local."""
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(content + "\n")
    return "Note sauvegardée avec succès."


def read_notes() -> str:
    """Lit les notes stockées dans le fichier texte local."""
    try:
        with open("notes.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Aucune note trouvée."


# 3. Initialisation de l'agent IA avec son modèle, ses outils et ses instructions
agent = Agent(
    model=model,
    tools=[get_current_time, calculate, save_note, read_notes],
    system_prompt=(
        "Vous êtes un assistant IA local efficace. Vous pouvez utiliser vos outils "
        "pour gérer des notes, faire des calculs ou donner la date et l'heure."
    ),
)


# 4. Boucle principale pour interagir avec l'agent
def main():
    print("🤖 Agent IA local prêt ! (Tapez 'quit' ou 'exit' pour quitter)\n")
    message_history = []

    while True:
        user_input = input("Vous : ")
        if user_input.lower() in ["quit", "exit"]:
            print("Au revoir !")
            break

        # Exécution de l'agent avec conservation de l'historique
        result = agent.run_sync(user_input, message_history=message_history)

        # Mise à jour de l'historique
        message_history = result.all_messages()

        print(f"\nAgent : {result.output}\n")


if __name__ == "__main__":
    main()