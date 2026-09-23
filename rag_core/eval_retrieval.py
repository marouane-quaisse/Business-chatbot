"""
=============================================================================
SCRIPT D'ÉVALUATION DU RETRIEVAL
=============================================================================

CE QUE FAIT CE SCRIPT
---------------------
Il mesure objectivement la qualité du RETRIEVAL (pas de la génération) en
utilisant le jeu de test de eval_dataset.py.

Il calcule 4 métriques classiques :

  1. HIT RATE @k
     % de questions pour lesquelles AU MOINS UN bon document est dans le top-k.
     → "Est-ce que le bon fichier est quelque part dans les résultats ?"

  2. RECALL @k
     % des documents attendus qui ont été retrouvés.
     (Ici équivalent au Hit Rate car on n'attend qu'1 source par question.)

  3. MRR (Mean Reciprocal Rank)
     Moyenne de 1/rang du PREMIER bon résultat.
     → "À quelle position apparaît le bon document ?"
     → MRR = 1.0 signifie que le bon doc est toujours en 1ère position.
     → MRR = 0.5 signifie qu'il est en moyenne en 2ème position.

  4. PRECISION @k (chunk-level)
     % des chunks récupérés qui viennent du bon fichier.
     → "Est-ce que je ramène du bruit ?"

COMMENT L'UTILISER
------------------
    python eval_retrieval.py

Le script lit les flags ENABLE_MULTI_QUERY / ENABLE_HYDE / ENABLE_RERANK
depuis .env. Pour comparer les configurations, modifie .env et relance.

POURQUOI C'EST IMPORTANT
------------------------
Sans ce script, on ne peut pas savoir si Multi-Query/HyDE/Reranking aident
vraiment. Avec, on obtient un tableau chiffré et reproductible.
=============================================================================
"""

import time

from eval_dataset import get_cases
from rag import retrieve
from vectorstore import collection_count

# Nombre de chunks récupérés pour l'évaluation (doit correspondre à TOP_K).
K = 4


# =============================================================================
# 1. ÉVALUATION D'UN SEUL CAS
# =============================================================================
def evaluate_case(case: dict, k: int = K) -> dict:
    """
    Évalue un cas de test et retourne les métriques intermédiaires.

    Returns:
        dict avec :
          - hit          : bool, le bon doc est-il dans le top-k ?
          - rank         : position du premier bon doc (1-indexé, None si absent)
          - precision    : proportion de chunks venant du bon fichier
          - keyword_hit  : bool, un mot-clé attendu est-il présent ?
          - latency      : temps de retrieval en secondes
    """
    start = time.perf_counter()
    docs = retrieve(case["question"], k=k)
    latency = time.perf_counter() - start

    expected = case["expected"]
    sources = [d.metadata.get("source", "") for d in docs]

    # --- Hit & Rank ---
    rank = None
    for i, src in enumerate(sources, start=1):
        if expected in src:
            rank = i
            break

    hit = rank is not None

    # --- Precision (chunk-level) ---
    relevant = sum(1 for src in sources if expected in src)
    precision = relevant / len(sources) if sources else 0.0

    # --- Vérification des mots-clés (qualité du PASSAGE, pas juste du fichier) ---
    # Un PDF de 140 chunks peut contenir le bon fichier mais le mauvais passage.
    # Les mots-clés permettent de détecter ce cas.
    keywords = case.get("keywords", [])
    if keywords:
        combined = " ".join(d.page_content.lower() for d in docs)
        keyword_hit = any(kw.lower() in combined for kw in keywords)
    else:
        keyword_hit = True  # pas de mot-clé défini → non testé

    return {
        "hit": hit,
        "rank": rank,
        "precision": precision,
        "keyword_hit": keyword_hit,
        "latency": latency,
    }


# =============================================================================
# 2. ÉVALUATION DE TOUT LE JEU DE TEST
# =============================================================================
def evaluate_all(k: int = K) -> dict:
    """
    Lance l'évaluation sur tous les cas et agrège les métriques.

    Returns:
        dict avec hit_rate, recall, mrr, precision, keyword_rate, avg_latency
        et le détail par cas.
    """
    cases = get_cases(include_traps=False)
    results = []

    print(f"\n{'=' * 78}")
    print(f"📊 ÉVALUATION DU RETRIEVAL — {len(cases)} cas de test, k={k}")
    print(f"{'=' * 78}\n")

    for case in cases:
        metrics = evaluate_case(case, k=k)
        results.append({**case, **metrics})

        # Affichage ligne par ligne pour suivre la progression.
        status = "✅" if metrics["hit"] else "❌"
        rank_str = f"rang {metrics['rank']}" if metrics["rank"] else "absent"
        kw_str = "" if metrics["keyword_hit"] else "  ⚠️ mot-clé manquant"
        diff = case["difficulty"][:4]
        print(f"{status} [{diff}] {case['question'][:52]:<52} → {rank_str}{kw_str}")

    # --- Agrégation ---
    n = len(results)
    hit_rate = sum(r["hit"] for r in results) / n
    recall = hit_rate  # 1 seule source attendue par question
    mrr = sum(1.0 / r["rank"] for r in results if r["rank"]) / n
    precision = sum(r["precision"] for r in results) / n
    keyword_rate = sum(r["keyword_hit"] for r in results) / n
    avg_latency = sum(r["latency"] for r in results) / n

    return {
        "hit_rate": hit_rate,
        "recall": recall,
        "mrr": mrr,
        "precision": precision,
        "keyword_rate": keyword_rate,
        "avg_latency": avg_latency,
        "details": results,
    }


# =============================================================================
# 3. AFFICHAGE DU RAPPORT
# =============================================================================
def print_report(metrics: dict) -> None:
    """Affiche le rapport final formaté."""
    print(f"\n{'=' * 78}")
    print("📈 RÉSULTATS")
    print(f"{'=' * 78}")
    print(f"  Hit Rate @k      : {metrics['hit_rate']:.1%}   (bon doc trouvé)")
    print(f"  Recall @k        : {metrics['recall']:.1%}")
    print(f"  MRR              : {metrics['mrr']:.3f}  (1.0 = toujours en 1ère position)")
    print(f"  Precision @k     : {metrics['precision']:.1%}   (chunks pertinents)")
    print(f"  Mots-clés OK     : {metrics['keyword_rate']:.1%}   (bon passage, pas juste bon fichier)")
    print(f"  Latence moyenne  : {metrics['avg_latency']:.2f}s")

    # --- Détail des échecs : c'est là qu'on apprend le plus ---
    failures = [r for r in metrics["details"] if not r["hit"]]
    if failures:
        print(f"\n❌ {len(failures)} échec(s) :")
        for f in failures:
            print(f"   • [{f['difficulty']}] {f['question']}")
            print(f"     attendu : {f['expected']}")

    # --- Cas où le fichier est bon mais le passage est mauvais ---
    bad_passages = [
        r for r in metrics["details"] if r["hit"] and not r["keyword_hit"]
    ]
    if bad_passages:
        print(f"\n⚠️  {len(bad_passages)} cas avec le bon fichier mais le mauvais passage :")
        for b in bad_passages:
            print(f"   • {b['question']}")

    print(f"\n{'=' * 78}\n")


# =============================================================================
# 4. POINT D'ENTRÉE
# =============================================================================
def main() -> None:
    count = collection_count()
    if count == 0:
        print("⚠️  Base vide. Lancez d'abord : python ingest.py --reset")
        return

    print(f"📚 {count} chunks indexés dans la base.")

    # On affiche la configuration active pour savoir ce qu'on mesure.
    import config

    print("\n⚙️  Configuration active :")
    print(f"   ENABLE_MULTI_QUERY   = {config.ENABLE_MULTI_QUERY}")
    print(f"   ENABLE_HYDE          = {config.ENABLE_HYDE}")
    print(f"   ENABLE_RERANK        = {config.ENABLE_RERANK}")
    print(f"   MULTI_QUERY_COUNT    = {config.MULTI_QUERY_COUNT}")
    print(f"   RETRIEVAL_CANDIDATES = {config.RETRIEVAL_CANDIDATES}")

    metrics = evaluate_all(k=K)
    print_report(metrics)


if __name__ == "__main__":
    main()
