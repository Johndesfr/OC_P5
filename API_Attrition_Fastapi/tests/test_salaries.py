from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_lister_salaries():

    profils_fictifs = [
        {
            "id": 1,
            "nom_profil": "profil_test",
            "poste": "Consultant"
        }
    ]

    with patch(
        "app.main.lister_profils",
        return_value=profils_fictifs
    ):
        response = client.get("/salaries")

    assert response.status_code == 200
    assert response.json() == profils_fictifs
