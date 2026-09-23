"""
=============================================================================
JEU DE TEST POUR L'ÉVALUATION DU RETRIEVAL
=============================================================================

À QUOI ÇA SERT ?
----------------
Sans jeu de test, impossible de savoir si Multi-Query / HyDE / Reranking
améliorent réellement les résultats. On ne peut que "sentir" que c'est mieux.

Ce fichier contient une liste de cas de test. Chaque cas associe :
  - une QUESTION (formulée comme un vrai utilisateur)
  - la SOURCE attendue (le fichier qui devrait contenir la réponse)
  - éventuellement un MOT-CLÉ attendu (pour vérifier la précision du chunk)

COMMENT L'ENRICHIR ?
--------------------
Ajoute des questions réelles posées par des utilisateurs. Plus le jeu est
représentatif, plus l'évaluation est fiable. 20-50 cas est un bon équilibre.

TYPES DE QUESTIONS (important pour tester la robustesse)
--------------------------------------------------------
  - "facile"    : reprend le vocabulaire du document → le retrieval de base
                  devrait réussir.
  - "difficile" : utilise des synonymes / une formulation différente → c'est
                  LÀ que Multi-Query et HyDE doivent faire la différence.
  - "piège"     : la réponse n'existe PAS dans les documents → le système
                  doit dire "je ne sais pas" (test anti-hallucination).
=============================================================================
"""

# =============================================================================
# CAS DE TEST
# =============================================================================
# difficulty : "facile" | "difficile" | "piege"
# expected   : nom du fichier source attendu (None si la réponse n'existe pas)
# keywords   : mots qui DEVRAIENT apparaître dans le chunk récupéré
#              (permet de vérifier qu'on a le bon PASSAGE, pas juste le bon
#               fichier — utile car un PDF de 140 chunks contient beaucoup
#               de passages non pertinents)
# reference  : réponse de référence courte, utilisée par eval_generation.py
#              pour juger la qualité de la réponse générée.
TEST_CASES: list[dict] = [
    # -------------------------------------------------------------------------
    # POLITIQUE DE RETOUR (politique_retour.md)
    # -------------------------------------------------------------------------
    {
        "question": "Quel est le délai de retour d'un article ?",
        "expected": "politique_retour.md",
        "keywords": ["30 jours"],
        "reference": "Le client dispose de 30 jours calendaires à compter de la réception du colis.",
        "difficulty": "facile",
    },
    {
        "question": "Combien de temps j'ai pour renvoyer un produit ?",
        "expected": "politique_retour.md",
        "keywords": ["30 jours"],
        "reference": "30 jours calendaires à partir de la réception.",
        "difficulty": "difficile",  # synonymes : renvoyer / produit
    },
    {
        "question": "Quels sont les frais si je change d'avis ?",
        "expected": "politique_retour.md",
        "keywords": ["4,90"],
        "reference": "Des frais de retour forfaitaires de 4,90 € sont déduits du remboursement.",
        "difficulty": "difficile",
    },
    {
        "question": "Quels produits ne peuvent pas être remboursés ?",
        "expected": "politique_retour.md",
        "keywords": ["exclus", "hygiène"],
        "reference": "Les produits d'hygiène ouverts, sous-vêtements, cartes cadeaux, produits personnalisés et denrées périssables.",
        "difficulty": "facile",
    },
    {
        "question": "Sous quel délai suis-je remboursé après un retour ?",
        "expected": "politique_retour.md",
        "keywords": ["5 à 10 jours"],
        "reference": "Le remboursement est effectué sous 5 à 10 jours ouvrés après validation du retour.",
        "difficulty": "facile",
    },
    {
        "question": "Est-ce que je peux échanger un article directement ?",
        "expected": "politique_retour.md",
        "keywords": ["échange"],
        "reference": "Non, l'échange direct n'est pas proposé : il faut faire un retour puis une nouvelle commande.",
        "difficulty": "facile",
    },
    {
        "question": "Que se passe-t-il si je renvoie le colis en retard ?",
        "expected": "politique_retour.md",
        "keywords": ["refusé", "délai"],
        "reference": "Un retour hors délai est refusé automatiquement par le système.",
        "difficulty": "difficile",
    },

    # -------------------------------------------------------------------------
    # HORAIRES / CONTACT / LIVRAISON (horaires_contact.md)
    # -------------------------------------------------------------------------
    {
        "question": "Quels sont les horaires du service client ?",
        "expected": "horaires_contact.md",
        "keywords": ["9h00", "18h00"],
        "reference": "Du lundi au vendredi de 9h00 à 18h00, et le samedi de 10h00 à 13h00.",
        "difficulty": "facile",
    },
    {
        "question": "Le support est-il ouvert le dimanche ?",
        "expected": "horaires_contact.md",
        "keywords": ["Dimanche", "Fermé"],
        "reference": "Non, le service est fermé le dimanche et les jours fériés.",
        "difficulty": "facile",
    },
    {
        "question": "Quel est le numéro de téléphone du support ?",
        "expected": "horaires_contact.md",
        "keywords": ["01 23 45 67 89"],
        "reference": "Le numéro est le 01 23 45 67 89.",
        "difficulty": "facile",
    },
    {
        "question": "Combien de temps prend la livraison en France ?",
        "expected": "horaires_contact.md",
        "keywords": ["3 à 5 jours"],
        "reference": "3 à 5 jours ouvrés en standard, 24 heures en express.",
        "difficulty": "facile",
    },
    {
        "question": "La livraison est-elle gratuite ?",
        "expected": "horaires_contact.md",
        "keywords": ["50", "offerte"],
        "reference": "Elle est offerte en France métropolitaine à partir de 50 €, sinon 5,90 €.",
        "difficulty": "difficile",  # la réponse dépend du montant
    },
    {
        "question": "Que se passe-t-il si je suis absent à la livraison ?",
        "expected": "horaires_contact.md",
        "keywords": ["avis de passage", "10 jours"],
        "reference": "Un avis de passage est laissé et le colis est conservé 10 jours en point relais.",
        "difficulty": "difficile",
    },
    {
        "question": "Quel est le délai de livraison pour les DOM-TOM ?",
        "expected": "horaires_contact.md",
        "keywords": ["7 à 12 jours"],
        "reference": "7 à 12 jours ouvrés, sans option express.",
        "difficulty": "facile",
    },
    {
        "question": "Comment suivre mon colis ?",
        "expected": "horaires_contact.md",
        "keywords": ["suivi", "numéro"],
        "reference": "Un numéro de suivi est envoyé par e-mail et SMS, consultable dans l'espace personnel.",
        "difficulty": "facile",
    },

    # -------------------------------------------------------------------------
    # RAPPORT PROJET MÉTIER — Orbyte (Rapport Projet Metier.pdf)
    # -------------------------------------------------------------------------
    {
        "question": "Quelles sont les entités du projet ?",
        "expected": "Rapport Projet Metier.pdf",
        "keywords": ["Workspace", "Sprint", "Tâche"],
        "reference": "Les entités sont Workspace, Sprint, Tâche et Space.",
        "difficulty": "facile",
    },
    {
        "question": "Quelle technologie est utilisée pour le backend ?",
        "expected": "Rapport Projet Metier.pdf",
        "keywords": ["Spring Boot"],
        "reference": "Le backend utilise Spring Boot.",
        "difficulty": "facile",
    },
    {
        "question": "Quel framework est utilisé côté frontend ?",
        "expected": "Rapport Projet Metier.pdf",
        "keywords": ["React", "TypeScript"],
        "reference": "Le frontend utilise React et TypeScript.",
        "difficulty": "facile",
    },
    {
        "question": "Sur quel cloud l'application est-elle déployée ?",
        "expected": "Rapport Projet Metier.pdf",
        "keywords": ["Hetzner"],
        "reference": "L'application est déployée sur Hetzner Cloud.",
        "difficulty": "facile",
    },
    {
        "question": "Quels outils d'infrastructure as code sont utilisés ?",
        "expected": "Rapport Projet Metier.pdf",
        "keywords": ["Terraform", "Ansible"],
        "reference": "Terraform et Ansible orchestrent l'infrastructure.",
        "difficulty": "difficile",  # formulation technique indirecte
    },
    {
        "question": "Quel est le nom de la plateforme développée ?",
        "expected": "Rapport Projet Metier.pdf",
        "keywords": ["Orbyte"],
        "reference": "La plateforme s'appelle Orbyte.",
        "difficulty": "facile",
    },
    {
        "question": "Qui a encadré ce projet ?",
        "expected": "Rapport Projet Metier.pdf",
        "keywords": ["JADLI"],
        "reference": "Le projet a été encadré par le Professeur JADLI Aissam.",
        "difficulty": "facile",
    },
    {
        "question": "Quel service gère l'intelligence artificielle ?",
        "expected": "Rapport Projet Metier.pdf",
        "keywords": ["FastAPI"],
        "reference": "Un service IA spécialisé sous FastAPI.",
        "difficulty": "difficile",
    },
    {
        "question": "Quelles méthodologies agiles sont supportées ?",
        "expected": "Rapport Projet Metier.pdf",
        "keywords": ["Scrum", "Kanban"],
        "reference": "Les méthodologies Scrum et Kanban sont supportées.",
        "difficulty": "facile",
    },
    {
        "question": "Quelle est l'architecture logicielle du système ?",
        "expected": "Rapport Projet Metier.pdf",
        "keywords": ["microservices"],
        "reference": "L'architecture repose sur une approche microservices.",
        "difficulty": "facile",
    },

    # -------------------------------------------------------------------------
    # CAS PIÈGES — la réponse N'EXISTE PAS dans les documents
    # -------------------------------------------------------------------------
    # Ces cas testent l'anti-hallucination. Le retrieval peut ramener des
    # chunks (c'est normal), mais la GÉNÉRATION doit refuser de répondre.
    # Pour l'évaluation du retrieval, on les marque "piege" et on les exclut
    # du calcul de Recall (ils n'ont pas de source attendue).
    {
        "question": "Quel est le chiffre d'affaires de l'entreprise en 2025 ?",
        "expected": None,
        "keywords": [],
        "reference": "REFUS ATTENDU : l'information n'existe pas dans les documents.",
        "difficulty": "piege",
    },
    {
        "question": "Combien d'employés travaillent dans l'entreprise ?",
        "expected": None,
        "keywords": [],
        "reference": "REFUS ATTENDU : l'information n'existe pas dans les documents.",
        "difficulty": "piege",
    },
    {
        "question": "Quelle est la recette du couscous royal ?",
        "expected": None,
        "keywords": [],
        "reference": "REFUS ATTENDU : l'information n'existe pas dans les documents.",
        "difficulty": "piege",
    },
]


def get_cases(include_traps: bool = False) -> list[dict]:
    """
    Retourne les cas de test.

    Args:
        include_traps: si False (défaut), exclut les cas "piège" qui n'ont
                       pas de source attendue et fausseraient le Recall.
    """
    if include_traps:
        return TEST_CASES
    return [c for c in TEST_CASES if c["expected"] is not None]
