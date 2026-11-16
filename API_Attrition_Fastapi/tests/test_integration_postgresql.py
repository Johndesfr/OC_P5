
import json
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
import pytest
from psycopg.types.json import Jsonb
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

FICHIER_PROFIL = sorted(
    (Path(__file__).parent / "fixtures").glob("*.json")
)[0]


@pytest.fixture
def connexion_test(monkeypatch):
    """Connexion à la base de test, sans validation définitive."""

    connexion = psycopg.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname="attrition_rh_test",
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )

    @contextmanager
    def ouvrir_connexion_isolee():
        yield connexion

    # L'API utilisera notre connexion de test
    monkeypatch.setattr(
        "app.main.ouvrir_connexion",
        ouvrir_connexion_isolee
    )

    try:
        yield connexion

    finally:
        # Annuler toutes les insertions du test
        connexion.rollback()
        connexion.close()


def charger_profil():
    with FICHIER_PROFIL.open(encoding="utf-8-sig") as fichier:
        return json.load(fichier)


def test_prediction_json_persistante(connexion_test):
    """Vérifie la persistance d'une prédiction JSON."""

    donnees = charger_profil()

    response = client.post("/predict", json=donnees)

    assert response.status_code == 200

    with connexion_test.cursor() as curseur:
        curseur.execute("""
            SELECT e.donnees, o.score_depart, o.prediction
            FROM predictions_entrees e
            JOIN predictions_sorties o ON o.entree_id = e.id
            ORDER BY e.id DESC
            LIMIT 1
        """)

        ligne = curseur.fetchone()

    assert ligne is not None

    donnees_stockees, score, prediction = ligne

    assert donnees_stockees == donnees
    assert score == pytest.approx(
        response.json()["score_depart"]
    )
    assert prediction == response.json()["prediction"]


def test_prediction_par_id_persistante(connexion_test):
    """Vérifie la prédiction et la traçabilité par ID."""

    donnees = charger_profil()

    # Créer un salarié temporaire en base de test
    with connexion_test.cursor() as curseur:
        curseur.execute(
            """
            INSERT INTO salaries_test (nom_profil, donnees)
            VALUES (%s, %s)
            RETURNING id
            """,
            ("profil_pytest", Jsonb(donnees))
        )

        salarie_id = curseur.fetchone()[0]

    # Calculer sa prédiction
    response = client.post(f"/predict/{salarie_id}")

    assert response.status_code == 200
    assert response.json()["salarie_id"] == salarie_id

    # Vérifier le lien entre salarié, entrée et sortie
    with connexion_test.cursor() as curseur:
        curseur.execute(
            """
            SELECT o.score_depart, o.prediction
            FROM predictions_entrees e
            JOIN predictions_sorties o ON o.entree_id = e.id
            WHERE e.salarie_test_id = %s
            """,
            (salarie_id,)
        )

        ligne = curseur.fetchone()

    assert ligne is not None

    score, prediction = ligne

    assert score == pytest.approx(
        response.json()["score_depart"]
    )
    assert prediction == response.json()["prediction"]
