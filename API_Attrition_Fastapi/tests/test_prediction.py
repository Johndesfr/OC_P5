import pytest

from fastapi.testclient import TestClient

from app.main import app, pipeline, seuil, features
from app.schemas import SalarieEntree
from app.features import construire_features


client = TestClient(app)


# Salarié de référence pour les tests
SALARIE_TEST = {
    "age": 41,
    "genre": "F",
    "revenu_mensuel": 5993,
    "statut_marital": "Célibataire",
    "departement": "Commercial",
    "poste": "Cadre Commercial",
    "nombre_experiences_precedentes": 8,
    "nombre_heures_travailless": 80,
    "annee_experience_totale": 8,
    "annees_dans_l_entreprise": 6,
    "annees_dans_le_poste_actuel": 4,
    "satisfaction_employee_environnement": 2,
    "note_evaluation_precedente": 3,
    "niveau_hierarchique_poste": 2,
    "satisfaction_employee_nature_travail": 4,
    "satisfaction_employee_equipe": 1,
    "satisfaction_employee_equilibre_pro_perso": 1,
    "note_evaluation_actuelle": 3,
    "heure_supplementaires": "Oui",
    "augementation_salaire_precedente": "11 %",
    "nombre_participation_pee": 0,
    "nb_formations_suivies": 0,
    "nombre_employee_sous_responsabilite": 1,
    "distance_domicile_travail": 1,
    "niveau_education": 2,
    "domaine_etude": "Infra & Cloud",
    "ayant_enfants": "Y",
    "frequence_deplacement": "Occasionnel",
    "annees_depuis_la_derniere_promotion": 0,
    "annes_sous_responsable_actuel": 5
}


def test_validation_salarie():

    response = client.post(
        "/predict",
        json=SALARIE_TEST
    )

    assert response.status_code == 200

    resultat = response.json()

    assert "score_depart" in resultat
    assert "prediction" in resultat
    assert "seuil_decision" in resultat
    assert "interpretation" in resultat

    assert resultat["prediction"] in (0, 1)




def test_preparation_features():

    # Valider les données avec Pydantic
    salarie = SalarieEntree(**SALARIE_TEST)

    # Appeler directement notre fonction Python
    X = construire_features(salarie, features)

    # Vérifier les dimensions et l'ordre
    assert X.shape == (1, 34)
    assert list(X.columns) == list(features)

    # Vérifier les calculs métier
    ligne = X.iloc[0]

    assert ligne["augmentation_salaire_precedente_pct"] == pytest.approx(11)

    assert ligne["ratio_anciennete_experience"] == pytest.approx(0.75)

    assert ligne["ratio_temps_poste_anciennete"] == pytest.approx(4 / 6)

    assert ligne["ratio_promotion_anciennete"] == pytest.approx(0)

    assert ligne["revenu_par_annee_experience"] == pytest.approx(5993 / 8)

    
def test_prediction_equivalente_au_pipeline():

    # 1. Prédiction obtenue par l'API
    response = client.post(
        "/predict",
        json=SALARIE_TEST
    )

    assert response.status_code == 200

    resultat_api = response.json()

    # 2. Prédiction directe avec Scikit-learn
    salarie = SalarieEntree(**SALARIE_TEST)

    X = construire_features(salarie, features)

    indice_classe_1 = list(pipeline.classes_).index(1)

    score_direct = float(
        pipeline.predict_proba(X)[0, indice_classe_1]
    )

    prediction_directe = int(score_direct >= seuil)

    # 3. Comparaison
    assert resultat_api["score_depart"] == pytest.approx(score_direct)
    assert resultat_api["prediction"] == prediction_directe
    assert resultat_api["seuil_decision"] == pytest.approx(0.4)
    assert 0 <= resultat_api["score_depart"] <= 1

@pytest.mark.parametrize(
    "modifications",
    [
        {"age": -5},
        {"genre": 42},
        {"champ_inconnu": "valeur"}
    ]
)
def test_rejet_donnees_invalides(modifications):

    donnees = SALARIE_TEST.copy()
    donnees.update(modifications)

    response = client.post(
        "/predict",
        json=donnees
    )

    assert response.status_code == 422
    assert "detail" in response.json()

    
@pytest.mark.parametrize(
    "modifications",
    [
        # Ancienneté entreprise > expérience totale
        {"annees_dans_l_entreprise": 10},

        # Ancienneté poste > ancienneté entreprise
        {"annees_dans_le_poste_actuel": 7}
    ]
)
def test_coherence_anciennete(modifications):

    donnees = SALARIE_TEST.copy()
    donnees.update(modifications)

    response = client.post(
        "/predict",
        json=donnees
    )

    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "modifications",
    [
        # Promotion plus ancienne que l'entrée dans l'entreprise
        {"annees_depuis_la_derniere_promotion": 10},

        # Expérience nulle, anciennetés nulles,
        # mais revenu mensuel positif
        {
            "annee_experience_totale": 0,
            "annees_dans_l_entreprise": 0,
            "annees_dans_le_poste_actuel": 0,
            "annees_depuis_la_derniere_promotion": 0
        }
    ]
)
def test_ratios_metier_invalides(modifications):

    donnees = SALARIE_TEST.copy()
    donnees.update(modifications)

    response = client.post(
        "/predict",
        json=donnees
    )

    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "champ,valeur_invalide",
    [
        ("genre", "X"),
        ("statut_marital", "Inconnu"),
        ("departement", "Informatique"),
        ("poste", "Développeur"),
        ("heure_supplementaires", "Peut-être"),
        ("domaine_etude", "Physique"),
        ("ayant_enfants", "N"),
        ("frequence_deplacement", "Jamais")
    ]
)
def test_categories_inconnues(
    champ,
    valeur_invalide
):
    # Copier le salarié valide
    donnees = SALARIE_TEST.copy()

    # Introduire une catégorie inconnue
    donnees[champ] = valeur_invalide

    # Interroger l'API
    response = client.post(
        "/predict",
        json=donnees
    )

    # Vérifier le rejet par Pydantic
    assert response.status_code == 422

    # Vérifier que l'erreur concerne le bon champ
    erreurs = response.json()["detail"]

    assert any(
        champ in erreur["loc"]
        for erreur in erreurs
    )

