
from pathlib import Path

import joblib
from fastapi import FastAPI

from .schemas import SalarieEntree, PredictionSortie
from .features import construire_features


# Initialisation de FastAPI
app = FastAPI(
    title="API Attrition RH",
    description="API de prédiction du départ des salariés",
    version="1.0.0"
)


# Chargement du modèle
BASE_DIR = Path(__file__).resolve().parent.parent
CHEMIN_MODELE = BASE_DIR / "models" / "turnover_logistic.joblib"

artefact = joblib.load(CHEMIN_MODELE)

pipeline = artefact["pipeline"]
seuil = artefact["seuil"]
features = artefact["features"]


# Endpoint 1 : état de l'API
@app.get("/health")
def health():
    return {"status": "ok"}


# Endpoint 2 : informations sur le modèle
@app.get("/model-info")
def model_info():
    return {
        "modele": "LogisticRegression",
        "charge": True,
        "nombre_features": len(features),
        "seuil_decision": seuil
    }


# Endpoint 3 : prédiction
@app.post("/predict", response_model=PredictionSortie)
def predire_depart(salarie: SalarieEntree):

    # Transformation des données
    X = construire_features(salarie, features)

    # Score de la classe 1 : départ potentiel
    score = float(
        pipeline.predict_proba(X)[0, 1]
    )

    # Application du seuil métier
    prediction = int(score >= seuil)

    # Interprétation
    interpretation = (
        "Départ potentiel détecté"
        if prediction == 1
        else "Départ non détecté"
    )

    return PredictionSortie(
        score_depart=score,
        prediction=prediction,
        interpretation=interpretation,
        seuil_decision=seuil
    )
