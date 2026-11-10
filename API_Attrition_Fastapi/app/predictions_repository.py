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
