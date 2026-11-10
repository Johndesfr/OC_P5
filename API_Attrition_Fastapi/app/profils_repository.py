
from app.database import ouvrir_connexion


def lister_profils():
    """Récupère les profils enregistrés dans PostgreSQL."""

    with ouvrir_connexion() as connexion:
        with connexion.cursor() as curseur:

            curseur.execute("""
                SELECT
                    id,
                    nom_profil,
                    donnees->>'poste' AS poste
                FROM salaries_test
                ORDER BY id
            """)

            lignes = curseur.fetchall()

    return [
        {
            "id": ligne[0],
            "nom_profil": ligne[1],
            "poste": ligne[2]
        }
        for ligne in lignes
    ]

def obtenir_profil_par_id(connexion, salarie_id: int):
    """Récupère les caractéristiques d'un salarié par son ID."""

    with connexion.cursor() as curseur:
        curseur.execute(
            """
            SELECT donnees
            FROM salaries_test
            WHERE id = %s
            """,
            (salarie_id,)
        )

        ligne = curseur.fetchone()

    return ligne[0] if ligne else None
