
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
