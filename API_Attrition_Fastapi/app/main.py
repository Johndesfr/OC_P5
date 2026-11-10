
from pathlib import Path

import joblib
from fastapi import FastAPI

from .schemas import SalarieEntree, PredictionSortie
from .features import construire_features
from app.profils_repository import lister_profils
from fastapi import HTTPException

from app.database import ouvrir_connexion
from app.profils_repository import obtenir_profil_par_id
from app.predictions_repository import (
    enregistrer_entree,
    enregistrer_sortie,
    obtenir_entree_par_id,
)

# Initialisation de FastAPI
app = FastAPI(
    title="API Attrition RH",
    description="API de prédiction du départ des salariés",
    version="1.0.0"
)

@app.get("/")
def accueil():
    return {
        "message": "API de prédiction du départ des salariés",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "prediction": "/predict",
            "documentation": "/docs"
        }
    }

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


#
@app.post("/predict", response_model=PredictionSortie)
def predire(salarie: SalarieEntree):
    """Prédit l'attrition et trace l'opération dans PostgreSQL."""

    with ouvrir_connexion() as connexion:

        # 1. Enregistrer les caractéristiques reçues
        entree_id = enregistrer_entree(
            connexion,
            None,
            salarie.model_dump(mode="json")
        )

        # 2. Relire les caractéristiques depuis PostgreSQL
        donnees = obtenir_entree_par_id(
            connexion,
            entree_id
        )

        # 3. Reconstruire les 34 features
        salarie_bdd = SalarieEntree.model_validate(donnees)

        X = construire_features(
            salarie_bdd,
            features
        )

        # 4. Calculer la prédiction
        score = float(pipeline.predict_proba(X)[0, 1])
        prediction = int(score >= seuil)

        # 5. Enregistrer le résultat
        enregistrer_sortie(
            connexion,
            entree_id,
            score,
            prediction,
            seuil
        )

    return {
        "score_depart": score,
        "prediction": prediction,
        "interpretation": (
            "Départ potentiel détecté"
            if prediction == 1
            else "Départ non détecté"
        ),
        "seuil_decision": seuil
    }


@app.get("/salaries")
def get_salaries():
    """Retourne les profils de salariés présents en base."""
    return lister_profils()


@app.post("/predict/{salarie_id}")
def predire_salarie_par_id(salarie_id: int):
    """Calcule et enregistre la prédiction d'un salarié."""

    with ouvrir_connexion() as connexion:

        # 1. Lire le salarié depuis PostgreSQL
        donnees = obtenir_profil_par_id(
            connexion,
            salarie_id
        )

        if donnees is None:
            raise HTTPException(
                status_code=404,
                detail="Salarié introuvable"
            )

        # 2. Valider les caractéristiques
        salarie = SalarieEntree.model_validate(donnees)

        # 3. Enregistrer l'entrée de prédiction
        entree_id = enregistrer_entree(
            connexion,
            salarie_id,
            salarie.model_dump(mode="json")
        )

        # 4. Préparer les features avec la fonction existante
        X = construire_features(salarie, features)

        # 5. Calculer la probabilité de départ
        score = float(pipeline.predict_proba(X)[0, 1])
        prediction = int(score >= seuil)

        # 6. Enregistrer le résultat
        sortie_id = enregistrer_sortie(
            connexion,
            entree_id,
            score,
            prediction,
            seuil
        )

    # 7. Retourner la prédiction
    return {
        "salarie_id": salarie_id,
        "entree_id": entree_id,
        "sortie_id": sortie_id,
        "score_depart": score,
        "prediction": prediction,
        "interpretation": (
            "Départ potentiel détecté"
            if prediction == 1
            else "Départ non détecté"
        ),
        "seuil_decision": seuil
    }

