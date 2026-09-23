"""
=============================================================================
COMPARAISON DES CONFIGURATIONS — RETRIEVAL + GÉNÉRATION
=============================================================================

CE QUE FAIT CE SCRIPT
---------------------
Il teste TOUTES les combinaisons des 3 flags (Multi-Query, HyDE, Rerank)
et produit un tableau comparatif.

  2^3 = 8 combinaisons possibles

Pour chaque combinaison, il mesure :
  - les métriques de RETRIEVAL (Hit Rate, MRR, Precision, Mots-clés)
  - les métriques de GÉNÉRATION (Faithfulness, Relevancy, Refus correct)
  - la latence

POURQUOI C'EST UTILE
--------------------
Sans ce script, on ne peut pas savoir si les optimisations servent à quelque
chose. On obtient un tableau chiffré qui permet de DÉCIDER objectivement.

⚠️ COÛT API
-----------
Chaque cas évalué consomme :
  - 1 appel LLM de génération
  - 1 appel LLM de jugement
  - (+ 1-2 appels si Multi-Query/HyDE activés)

Avec 28 cas × 8 configs × ~3 appels = ~670 appels LLM.
C'est long (10-20 min) et proche des rate limits Groq.

→ Utilise --quick pour tester sur un sous-ensemble (recommandé pour itérer).
→ Utilise --retrieval-only pour ne tester que le retrieval (rapide, gratuit).

USAGE
-----
    python eval_compare.py                    # tout, toutes les configs
    python eval_compare.py --quick            # 8 cas seulement
    python eval_compare.py --retrieval-only   # pas d'appel LLM de génération
    python eval_compare.py --configs 0,7      # seulement baseline et full
=============================================================================
"""

import argparse
import itertools
import time

import config
from eval_dataset import get_cases


# =============================================================================
# 1. GÉNÉRATION DES COMBINAISONS
# =============================================================================
def build_configs() -> list[dict]:
    """
    Génère les 8 combinaisons des 3 flags.

    Returns:
        Liste de dicts : {"name", "multi_query", "hyde", "rerank"}
    """
    configs = []
    for mq, hyde, rr in itertools.product([False, True], repeat=3):
        # Nom lisible : "MQ+HyDE+Rerank" ou "Baseline"
        parts = []
        if mq:
            parts.append("MQ")
        if hyde:
            parts.append("HyDE")
        if rr:
            parts.append("Rerank")
        name = "+".join(parts) if parts else "Baseline"

        configs.append({
            "name": name,
            "multi_query": mq,
            "hyde": hyde,
            "rerank": rr,
        })
    return configs


# =============================================================================
# 2. APPLICATION D'UNE CONFIGURATION
# =============================================================================
def apply_config(cfg: dict) -> None:
    """
    Force les flags dans le module config.

    ⚠️ On modifie config en mémoire (pas le .env). Les modules rag.py et
    query_transform.py lisent config.ENABLE_* à chaque appel, donc le
    changement prend effet immédiatement.
    """
    config.ENABLE_MULTI_QUERY = cfg["multi_query"]
    config.ENABLE_HYDE = cfg["hyde"]
    config.ENABLE_RERANK = cfg["rerank"]


# =============================================================================
# 3. ÉVALUATION D'UNE CONFIGURATION
# =============================================================================
def evaluate_config(cfg: dict, cases: list[dict], with_generation: bool) -> dict:
    """
    Évalue une configuration sur tous les cas.

    Args:
        cfg: la configuration à tester.
        cases: les cas de test.
        with_generation: si True, évalue aussi la génération (coûteux).

    Returns:
        dict avec les métriques agrégées.
    """
    # Import local : les modules lisent config au moment de l'appel.
    from rag import retrieve

    apply_config(cfg)

    # --- Accumulateurs ---
    hits = 0
    ranks = []
    precisions = []
    keyword_hits = 0
    latencies = []

    faith_scores = []
    relev_scores = []
    traps_ok = 0
    traps_total = 0

    for case in cases:
        # ---------- RETRIEVAL ----------
        start = time.perf_counter()
        docs = retrieve(case["question"], k=4)
        latencies.append(time.perf_counter() - start)

        expected = case["expected"]
        sources = [d.metadata.get("source", "") for d in docs]

        # Hit & rank (uniquement pour les cas avec source attendue)
        if expected:
            rank = None
            for i, src in enumerate(sources, start=1):
                if expected in src:
                    rank = i
                    break
            if rank:
                hits += 1
                ranks.append(rank)

            relevant = sum(1 for s in sources if expected in s)
            precisions.append(relevant / len(sources) if sources else 0.0)

            keywords = case.get("keywords", [])
            if keywords:
                combined = " ".join(d.page_content.lower() for d in docs)
                if any(kw.lower() in combined for kw in keywords):
                    keyword_hits += 1

        # ---------- GÉNÉRATION ----------
        if with_generation:
            from eval_generation import evaluate_case as eval_gen_case

            gen = eval_gen_case(case)

            if case["difficulty"] == "piege":
                traps_total += 1
                if gen["refused"]:
                    traps_ok += 1
            else:
                if isinstance(gen["faithfulness"], (int, float)):
                    faith_scores.append(gen["faithfulness"])
                if isinstance(gen["relevancy"], (int, float)):
                    relev_scores.append(gen["relevancy"])

    # --- Agrégation ---
    n_with_source = sum(1 for c in cases if c["expected"])
    n_with_keywords = sum(1 for c in cases if c.get("keywords"))

    return {
        "name": cfg["name"],
        "hit_rate": hits / n_with_source if n_with_source else 0.0,
        "mrr": sum(1.0 / r for r in ranks) / n_with_source if n_with_source else 0.0,
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "keyword_rate": keyword_hits / n_with_keywords if n_with_keywords else 0.0,
        "faithfulness": sum(faith_scores) / len(faith_scores) if faith_scores else None,
        "relevancy": sum(relev_scores) / len(relev_scores) if relev_scores else None,
        "refusal_rate": traps_ok / traps_total if traps_total else None,
        "latency": sum(latencies) / len(latencies) if latencies else 0.0,
    }


# =============================================================================
# 4. AFFICHAGE DU TABLEAU COMPARATIF
# =============================================================================
def print_table(results: list[dict], with_generation: bool) -> None:
    """Affiche le tableau comparatif final."""
    print(f"\n{'=' * 100}")
    print("📊 TABLEAU COMPARATIF DES CONFIGURATIONS")
    print(f"{'=' * 100}\n")

    # En-tête
    header = f"{'Configuration':<22} {'Hit':>6} {'MRR':>6} {'Prec':>6} {'Mots-clés':>10}"
    if with_generation:
        header += f" {'Faith':>7} {'Relev':>7} {'Refus':>7}"
    header += f" {'Latence':>9}"
    print(header)
    print("-" * len(header))

    # Lignes
    for r in results:
        line = (
            f"{r['name']:<22} "
            f"{r['hit_rate']:>5.0%} "
            f"{r['mrr']:>6.3f} "
            f"{r['precision']:>5.0%} "
            f"{r['keyword_rate']:>9.0%}"
        )
        if with_generation:
            f = f"{r['faithfulness']:.0%}" if r["faithfulness"] is not None else "n/a"
            rv = f"{r['relevancy']:.0%}" if r["relevancy"] is not None else "n/a"
            rf = f"{r['refusal_rate']:.0%}" if r["refusal_rate"] is not None else "n/a"
            line += f" {f:>7} {rv:>7} {rf:>7}"
        line += f" {r['latency']:>8.2f}s"
        print(line)

    print("-" * len(header))

    # --- Identification du meilleur ---
    print("\n🏆 MEILLEURES CONFIGURATIONS :")

    best_quality = max(results, key=lambda r: (r["keyword_rate"], r["mrr"]))
    print(f"   Qualité retrieval : {best_quality['name']} "
          f"(mots-clés {best_quality['keyword_rate']:.0%}, MRR {best_quality['mrr']:.3f})")

    best_speed = min(results, key=lambda r: r["latency"])
    print(f"   Rapidité          : {best_speed['name']} ({best_speed['latency']:.2f}s)")

    if with_generation:
        scored = [r for r in results if r["faithfulness"] is not None]
        if scored:
            best_faith = max(scored, key=lambda r: r["faithfulness"])
            print(f"   Fidélité          : {best_faith['name']} "
                  f"({best_faith['faithfulness']:.0%})")

    # --- Recommandation ---
    print("\n💡 RECOMMANDATION :")
    # On cherche la config la plus rapide qui garde la meilleure qualité.
    best_kw = max(r["keyword_rate"] for r in results)
    candidates = [r for r in results if r["keyword_rate"] >= best_kw - 0.02]
    recommended = min(candidates, key=lambda r: r["latency"])
    print(f"   → {recommended['name']}")
    print(f"     (qualité quasi-optimale avec la latence la plus basse : "
          f"{recommended['latency']:.2f}s)")
    print(f"\n{'=' * 100}\n")


# =============================================================================
# 5. POINT D'ENTRÉE
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare les configurations RAG.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Teste sur un sous-ensemble de cas (rapide).",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="N'évalue que le retrieval (pas d'appel LLM de génération).",
    )
    parser.add_argument(
        "--configs", type=str, default=None,
        help="Indices des configs à tester, ex: '0,7' (0=Baseline, 7=tout).",
    )
    parser.add_argument(
        "--limit", type=int, default=8,
        help="Nombre de cas en mode --quick (défaut: 8).",
    )
    args = parser.parse_args()

    from vectorstore import collection_count

    if collection_count() == 0:
        print("⚠️  Base vide. Lancez d'abord : python ingest.py --reset")
        return

    # --- Sélection des cas ---
    cases = get_cases(include_traps=True)
    if args.quick:
        # On garde un mélange : quelques faciles, difficiles et pièges.
        normal = [c for c in cases if c["difficulty"] != "piege"]
        traps = [c for c in cases if c["difficulty"] == "piege"]
        cases = normal[: args.limit - 1] + traps[:1]
        print(f"⚡ Mode rapide : {len(cases)} cas")

    # --- Sélection des configs ---
    all_configs = build_configs()
    if args.configs:
        indices = [int(i) for i in args.configs.split(",")]
        all_configs = [all_configs[i] for i in indices]

    with_generation = not args.retrieval_only

    # --- Estimation du coût ---
    n_calls = len(cases) * len(all_configs)
    if with_generation:
        n_calls *= 2  # génération + jugement
    print(f"\n⚙️  {len(all_configs)} configuration(s) × {len(cases)} cas")
    print(f"   Mode : {'retrieval + génération' if with_generation else 'retrieval seul'}")
    print(f"   Appels LLM estimés : ~{n_calls}")

    # --- Boucle principale ---
    results = []
    for i, cfg in enumerate(all_configs, start=1):
        print(f"\n{'─' * 78}")
        print(f"[{i}/{len(all_configs)}] Configuration : {cfg['name']}")
        print(f"   Multi-Query={cfg['multi_query']}  HyDE={cfg['hyde']}  Rerank={cfg['rerank']}")
        print(f"{'─' * 78}")

        start = time.perf_counter()
        metrics = evaluate_config(cfg, cases, with_generation)
        elapsed = time.perf_counter() - start

        results.append(metrics)
        print(f"   ✅ Terminé en {elapsed:.1f}s — "
              f"mots-clés {metrics['keyword_rate']:.0%}, "
              f"MRR {metrics['mrr']:.3f}")

    # --- Rapport final ---
    print_table(results, with_generation)


if __name__ == "__main__":
    main()
