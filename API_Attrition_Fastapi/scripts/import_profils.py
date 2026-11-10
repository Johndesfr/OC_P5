
import json
from pathlib import Path

from psycopg.types.json import Jsonb

from app.database import ouvrir_connexion
from app.schemas import SalarieEntree


# Dossier contenant nos 10 profils JSON
BASE_DIR = Path(__file__).resolve().parents[1]
DOSSIER_PROFILS = BASE_DIR / "tests" / "fixtures"


def importer_profils():

    fichiers = sorted(DOSSIER_PROFILS.glob("*.json"))

    print(f"Profils trouvés : {len(fichiers)}")

    if len(fichiers) != 10:
        raise ValueError("10 fichiers JSON sont attendus.")

    nb_insertions = 0

    with ouvrir_connexion() as connexion:
        with connexion.cursor() as curseur:

            for fichier in fichiers:

                # 1. Lire le JSON
                with fichier.open(
                    encoding="utf-8-sig"
                ) as f:
                    donnees = json.load(f)

                # 2. Valider avec Pydantic
                salarie = SalarieEntree.model_validate(donnees)

                # 3. Insérer dans PostgreSQL
                curseur.execute(
                    """
                    INSERT INTO salaries_test
                        (nom_profil, donnees)
                    VALUES (%s, %s)
                    ON CONFLICT (nom_profil) DO NOTHING
                    RETURNING id
                    """,
                    (
                        fichier.stem,
                        Jsonb(salarie.model_dump(mode="json"))
                    )
                )

                resultat = curseur.fetchone()

                if resultat is not None:
                    nb_insertions += 1
                    print(f"Inséré : {fichier.stem}")
                else:
                    print(f"Déjà présent : {fichier.stem}")

    print(f"\nNouvelles insertions : {nb_insertions}")


if __name__ == "__main__":
    importer_profils()
