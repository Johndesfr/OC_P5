
from fastapi.testclient import TestClient

from app.main import app


# Créer un client HTTP de test
client = TestClient(app)


# Test 1 : vérifier que l'API répond
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Test 2 : vérifier les informations du modèle
def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    donnees = response.json()

    assert donnees["charge"] is True
    assert donnees["modele"] == "LogisticRegression"
    assert donnees["nombre_features"] == 34
    assert donnees["seuil_decision"] == 0.4
