from psycopg.types.json import Jsonb


def enregistrer_entree(connexion, salarie_id, donnees):
    """Enregistre les caractéristiques utilisées pour la prédiction."""

    with connexion.cursor() as curseur:
        curseur.execute(
            """
            INSERT INTO predictions_entrees
                (salarie_test_id, donnees)
            VALUES (%s, %s)
            RETURNING id
            """,
            (salarie_id, Jsonb(donnees))
        )

        return curseur.fetchone()[0]


def enregistrer_sortie(connexion, entree_id, score, prediction, seuil):
    """Enregistre le résultat de la prédiction."""

    with connexion.cursor() as curseur:
        curseur.execute(
            """
            INSERT INTO predictions_sorties
                (entree_id, score_depart, prediction, seuil_decision)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (entree_id, score, prediction, seuil)
        )

        return curseur.fetchone()[0]
    
def obtenir_entree_par_id(connexion, entree_id: int):
    """Relit les caractéristiques enregistrées dans PostgreSQL."""

    with connexion.cursor() as curseur:
        curseur.execute(
            """
            SELECT donnees
            FROM predictions_entrees
            WHERE id = %s
            """,
            (entree_id,)
        )

        ligne = curseur.fetchone()

    if ligne is None:
        raise LookupError(
            f"Entrée de prédiction introuvable : {entree_id}"
        )

    return ligne[0]

