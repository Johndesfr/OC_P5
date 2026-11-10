
from app.database import ouvrir_connexion


def main():

    with ouvrir_connexion() as connexion:

        with connexion.cursor() as curseur:

            # Vérifier la base utilisée
            curseur.execute(
                "SELECT current_database(), current_user"
            )

            base, utilisateur = curseur.fetchone()

            print("Base connectée :", base)
            print("Utilisateur :", utilisateur)

            # Vérifier les tables présentes
            curseur.execute("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
            """)

            tables = curseur.fetchall()

            print("\nTables PostgreSQL :")

            for table in tables:
                print("-", table[0])


if __name__ == "__main__":
    main()
