"""
=============================================================================
SCRIPT D'ÉVALUATION DE LA GÉNÉRATION (LLM-as-a-Judge)
=============================================================================

POURQUOI CE SCRIPT ?
--------------------
eval_retrieval.py mesure si on trouve les BONS CHUNKS.
Ce script mesure si le LLM rédige une BONNE RÉPONSE à partir de ces chunks.

Un retrieval parfait ne garantit rien : le LLM peut halluciner, refuser à
tort, ou répondre à côté. C'est ce qu'on mesure ici.

LES 3 MÉTIQUES ÉVALUÉES
-----------------------
  1. FAITHFULNESS (fidélité) — 🔴 LA PLUS IMPORTANTE
     Chaque affirmation de la réponse est-elle soutenue par le contexte ?
     → Détecte les hallucinations. Critique pour un chatbot business.

  2. RELEVANCY (pertinence)
     La réponse répond-elle vraiment à la question posée ?
     → Détecte les réponses hors sujet ou incomplètes.

  3. REFUS CORRECT (anti-hallucination)
     Sur les questions "piège" (info absente), le LLM refuse-t-il de répondre ?
     → Test binaire : soit il refuse (✅), soit il invente (❌).

COMMENT ÇA MARCHE : LE "JUGE LLM"
---------------------------------
On utilise un LLM comme évaluateur. Il reçoit (question, contexte, réponse)
et note selon des critères précis.

⚠️ RÈGLE IMPORTANTE : le juge doit être un modèle DIFFÉRENT de celui qui
génère. Sinon il a tendance à préférer ses propres réponses (biais
d'auto-préférence). Ici :
   - génération : GROQ_MODEL (openai/gpt-oss-120b)
   - jugement   : JUDGE_MODEL (openai/gpt-oss-20b)

COÛT
----
1 appel LLM par question évaluée. Sur 28 cas = 28 appels. Acceptable.

Usage :
    python eval_generation.py
=============================================================================
"""

import json
import re
import time

from langchain_groq import ChatGroq

import config
from eval_dataset import get_cases
from rag import answer

# Modèle utilisé comme juge. DOIT être différent du modèle de génération.
JUDGE_MODEL = "openai/gpt-oss-20b"

# Nombre de cas à évaluer (mets None pour tout évaluer).
# Utile pour tester rapidement sans consommer trop d'appels API.
MAX_CASES = None


# =============================================================================
# 1. LE PROMPT DU JUGE
# =============================================================================
JUDGE_PROMPT = """Tu es un évaluateur strict et impartial de réponses de chatbot.

Voici une question, le contexte fourni au chatbot, et la réponse qu'il a produite.

--- QUESTION ---
{question}

--- CONTEXTE FOURNI ---
{context}

--- RÉPONSE DU CHATBOT ---
{answer}

Évalue la réponse selon 2 critères, chacun noté de 0.0 à 1.0 :

1. FAITHFULNESS (fidélité au contexte) :
   - 1.0 = chaque affirmation est explicitement soutenue par le contexte
   - 0.5 = globalement correct mais contient un détail non vérifiable
   - 0.0 = contient des informations inventées absentes du contexte

2. RELEVANCY (pertinence) :
   - 1.0 = répond complètement et directement à la question
   - 0.5 = répond partiellement ou de façon vague
   - 0.0 = ne répond pas à la question posée

Réponds UNIQUEMENT avec un objet JSON valide, sans texte autour :
{{"faithfulness": 0.0, "relevancy": 0.0, "raison": "explication courte"}}
"""


# =============================================================================
# 2. LE JUGE
# =============================================================================
_judge_llm = None


def get_judge() -> ChatGroq:
    """Retourne le LLM juge (créé une seule fois, mis en cache)."""
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatGroq(
            model=JUDGE_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0,  # juge déterministe
        )
    return _judge_llm


def parse_json_response(text: str) -> dict:
    """
    Extrait le JSON de la réponse du juge.

    Le LLM ajoute parfois du texte autour du JSON (markdown, explications).
    On cherche donc le premier objet {...} valide dans la réponse.
    """
    # Tentative directe
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Recherche d'un bloc {...} dans le texte
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Échec : on retourne des scores neutres plutôt que de planter.
    return {"faithfulness": None, "relevancy": None, "raison": "parse error"}


def judge_answer(question: str, context: str, answer_text: str) -> dict:
    """
    Fait juger une réponse par le LLM juge.

    Returns:
        dict avec faithfulness, relevancy (0-1 ou None) et raison.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context[:4000],  # on tronque pour rester dans les limites
        answer=answer_text,
    )
    try:
        response = get_judge().invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        return parse_json_response(text)
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  Juge échoué : {exc}")
        return {"faithfulness": None, "relevancy": None, "raison": str(exc)}


# =============================================================================
# 3. DÉTECTION DU REFUS (pour les cas pièges)
# =============================================================================
# Formules de refus acceptables. Si la réponse contient l'une d'elles,
# on considère que le LLM a correctement refusé de répondre.
REFUSAL_MARKERS = [
    "je ne dispose pas",
    "je n'ai pas",
    "pas d'information",
    "pas cette information",
    "ne figure pas",
    "n'est pas mentionné",
    "n'est pas précisé",
    "aucune information",
    "je ne peux pas répondre",
    "introuvable",
]


def is_refusal(text: str) -> bool:
    """Détecte si la réponse est un refus de répondre."""
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


# =============================================================================
# 4. ÉVALUATION D'UN CAS
# =============================================================================
def evaluate_case(case: dict) -> dict:
    """
    Évalue un cas complet : génération + jugement.

    Returns:
        dict avec la réponse, les scores du juge, et le statut du refus.
    """
    question = case["question"]

    # --- Génération ---
    start = time.perf_counter()
    result = answer(question)
    latency = time.perf_counter() - start

    response_text = result["answer"]
    context = "\n\n".join(d.page_content for d in result["chunks"])

    # --- Cas piège : on teste le refus, pas la fidélité ---
    if case["difficulty"] == "piege":
        refused = is_refusal(response_text)
        return {
            "question": question,
            "difficulty": "piege",
            "answer": response_text,
            "refused": refused,
            "faithfulness": None,
            "relevancy": None,
            "raison": "refus correct" if refused else "A INVENTÉ une réponse",
            "latency": latency,
        }

    # --- Cas normal : on fait juger la réponse ---
    scores = judge_answer(question, context, response_text)

    return {
        "question": question,
        "difficulty": case["difficulty"],
        "answer": response_text,
        "refused": is_refusal(response_text),
        "faithfulness": scores.get("faithfulness"),
        "relevancy": scores.get("relevancy"),
        "raison": scores.get("raison", ""),
        "latency": latency,
    }


# =============================================================================
# 5. ÉVALUATION COMPLÈTE
# =============================================================================
def evaluate_all() -> dict:
    """Lance l'évaluation sur tous les cas (y compris les pièges)."""
    cases = get_cases(include_traps=True)
    if MAX_CASES:
        cases = cases[:MAX_CASES]

    results = []

    print(f"\n{'=' * 78}")
    print(f"🧠 ÉVALUATION DE LA GÉNÉRATION — {len(cases)} cas")
    print(f"   Générateur : {config.GROQ_MODEL}")
    print(f"   Juge       : {JUDGE_MODEL}")
    print(f"{'=' * 78}\n")

    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['question'][:60]}")
        metrics = evaluate_case(case)
        results.append({**case, **metrics})

        # Affichage immédiat du verdict
        if metrics["difficulty"] == "piege":
            icon = "✅" if metrics["refused"] else "❌"
            print(f"   {icon} {metrics['raison']}")
        else:
            f = metrics["faithfulness"]
            r = metrics["relevancy"]
            f_str = f"{f:.2f}" if isinstance(f, (int, float)) else "n/a"
            r_str = f"{r:.2f}" if isinstance(r, (int, float)) else "n/a"
            icon = "✅" if (isinstance(f, (int, float)) and f >= 0.8) else "⚠️"
            print(f"   {icon} faithfulness={f_str}  relevancy={r_str}")
        print()

    return {"details": results}


# =============================================================================
# 6. RAPPORT
# =============================================================================
def print_report(data: dict) -> None:
    """Agrège et affiche les résultats."""
    details = data["details"]

    normal = [d for d in details if d["difficulty"] != "piege"]
    traps = [d for d in details if d["difficulty"] == "piege"]

    # --- Moyennes sur les cas normaux ---
    faith_scores = [d["faithfulness"] for d in normal if isinstance(d["faithfulness"], (int, float))]
    relev_scores = [d["relevancy"] for d in normal if isinstance(d["relevancy"], (int, float))]

    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
    avg_relev = sum(relev_scores) / len(relev_scores) if relev_scores else 0.0

    # --- Taux de refus correct sur les pièges ---
    refused_ok = sum(1 for d in traps if d["refused"])
    refusal_rate = refused_ok / len(traps) if traps else 0.0

    # --- Faux refus : le LLM refuse alors que l'info existe ---
    false_refusals = [d for d in normal if d["refused"]]

    print(f"\n{'=' * 78}")
    print("📈 RÉSULTATS DE LA GÉNÉRATION")
    print(f"{'=' * 78}")
    print(f"  Faithfulness moyenne   : {avg_faith:.1%}   (fidélité au contexte)")
    print(f"  Relevancy moyenne      : {avg_relev:.1%}   (répond à la question)")
    print(f"  Refus correct (pièges) : {refusal_rate:.1%}   ({refused_ok}/{len(traps)})")
    print(f"  Faux refus             : {len(false_refusals)}   (refuse alors que l'info existe)")

    # --- Détail des hallucinations ---
    hallucinations = [
        d for d in normal
        if isinstance(d["faithfulness"], (int, float)) and d["faithfulness"] < 0.8
    ]
    if hallucinations:
        print(f"\n⚠️  {len(hallucinations)} réponse(s) peu fidèle(s) :")
        for h in hallucinations:
            print(f"   • {h['question']}")
            print(f"     faithfulness={h['faithfulness']} — {h['raison'][:80]}")

    # --- Détail des pièges ratés ---
    failed_traps = [d for d in traps if not d["refused"]]
    if failed_traps:
        print(f"\n❌ {len(failed_traps)} hallucination(s) sur question piège :")
        for t in failed_traps:
            print(f"   • {t['question']}")
            print(f"     → {t['answer'][:100]}")

    # --- Faux refus ---
    if false_refusals:
        print(f"\n⚠️  {len(false_refusals)} faux refus (l'info existait) :")
        for f in false_refusals:
            print(f"   • {f['question']}")

    print(f"\n{'=' * 78}\n")


# =============================================================================
# 7. POINT D'ENTRÉE
# =============================================================================
def main() -> None:
    from vectorstore import collection_count

    if collection_count() == 0:
        print("⚠️  Base vide. Lancez d'abord : python ingest.py --reset")
        return

    data = evaluate_all()
    print_report(data)


if __name__ == "__main__":
    main()
