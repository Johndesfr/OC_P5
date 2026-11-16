import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_prediction_json_enregistree():

    dossier = Path(__file__).parent / "fixtures"
    fichier = sorted(dossier.glob("*.json"))[0]

    donnees = json.loads(
        fichier.read_text(encoding="utf-8-sig")
    )

    with (
        patch("app.main.ouvrir_connexion"),
        patch(
            "app.main.enregistrer_entree",
            return_value=501
        ) as enregistrer,
        patch(
            "app.main.obtenir_entree_par_id",
            return_value=donnees
        ) as relire,
        patch(
            "app.main.enregistrer_sortie",
            return_value=601
        ) as enregistrer_sortie_mock
    ):
        response = client.post("/predict", json=donnees)

    assert response.status_code == 200
    assert "score_depart" in response.json()

    enregistrer.assert_called_once()
    assert enregistrer.call_args.args[1] is None

    relire.assert_called_once()
    enregistrer_sortie_mock.assert_called_once()
