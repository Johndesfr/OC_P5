
from pathlib import Path
import joblib
from fastapi import FastAPI
from .features import construire_features
from .schemas import SalarieEntree, PredictionSortie


# Création de l'application
app = FastAPI(
    title="API Attrition RH",
    description="API de prédiction du départ des salariés",
    version="0.1.0"
)

# Chemin absolu vers notre modèle
BASE_DIR = Path(__file__).resolve().parent.parent
CHEMIN_MODELE = BASE_DIR / "models" / "turnover_logistic.joblib"

# Chargement de l'artefact sauvegardé
artefact = joblib.load(CHEMIN_MODELE)

pipeline = artefact["pipeline"]
seuil = artefact["seuil"]
features = artefact["features"]

# Endpoint de vérification
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint d'information sur notre modèle
@app.get("/model-info")
def model_info():
    return {
        "modele": "LogisticRegression",
        "charge": True,
        "nombre_features": len(features),
        "seuil_decision": seuil
    }

@app.post("/validate")
def valider_salarie(salarie: SalarieEntree):
    return {
        "validation": "ok",
        "message": "Les données ont été validées par Pydantic."
    }

@app.post("/prepare")
def preparer_donnees(salarie: SalarieEntree):

    # Calcul des features métier
    X = construire_features(salarie, features)

    # Variables créées dans le projet 4
    features_creees = [
        "augmentation_salaire_precedente_pct",
        "ratio_anciennete_experience",
        "ratio_temps_poste_anciennete",
        "ratio_promotion_anciennete",
        "revenu_par_annee_experience"
    ]

    return {
        "nombre_features": X.shape[1],
        "ordre_conforme": list(X.columns) == list(features),
        "features_calculees": {
            nom: float(X.iloc[0][nom])
            for nom in features_creees
        }
    }

@app.post("/predict", response_model=PredictionSortie)
def predire_depart(salarie: SalarieEntree):
    """
    Prédit le départ potentiel d'un salarié.
    Applique le seuil métier défini dans le projet 4.
    """

    # 1. Construire les 34 features
    X = construire_features(salarie, features)

    # 2. Calculer le score pour la classe 1 (départ)
    score = float(
        pipeline.predict_proba(X)[0, 1]
    )

    # 3. Appliquer le seuil métier de 0.40
    prediction = int(score >= seuil)

    # 4. Interpréter la classe prédite
    interpretation = (
        "Départ potentiel détecté"
        if prediction == 1
        else "Départ non détecté"
    )

    # 5. Retourner la réponse
    return PredictionSortie(
        score_depart=score,
        prediction=prediction,
        interpretation=interpretation,
        seuil_decision=seuil
    )
