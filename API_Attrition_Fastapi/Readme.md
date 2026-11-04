# API Attrition RH — Projet 5

## Présentation

Cette application REST développée avec **FastAPI** expose un modèle de Machine Learning destiné à identifier les salariés présentant un risque potentiel de départ d'une entreprise.

Le modèle provient du projet 4. Il utilise une **régression logistique Scikit-learn**, optimisée par GridSearchCV, avec une régularisation L2 et une pondération équilibrée des classes.

L'objectif métier est de privilégier le rappel afin de limiter les départs non détectés. Le seuil de décision retenu lors du projet 4 est de **0,40**.

L'API permet de soumettre les caractéristiques d'un salarié, de vérifier leur validité avec Pydantic, puis d'obtenir un score de classification et une prédiction binaire.

## 1. Technologies

| Technologie | Utilisation |
|---|---|
| Python | Langage de programmation |
| FastAPI | Framework de l'API REST |
| Pydantic | Validation des données entrantes et sortantes |
| Scikit-learn | Prétraitement et régression logistique |
| Pandas / NumPy | Manipulation et transformation des données |
| Joblib | Chargement du pipeline entraîné |
| Poetry | Gestion des dépendances |
| Uvicorn | Serveur ASGI |
| Pytest / HTTPX | Tests automatisés |

## 2. Organisation du projet

```text
projet5/
├── pyproject.toml
├── poetry.lock
├── API_Attrition_Fastapi/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── features.py
│   ├── models/
│   │   └── turnover_logistic.joblib
│   ├── notebooks/
│   │   └── notebook_corrige_features.ipynb
│   ├── tests/
│   │   ├── test_main.py
│   │   └── test_prediction.py
│   └── README.md
```

## 3. Installation

Prérequis : Python et Poetry, dans des versions compatibles avec celles verrouillées dans `poetry.lock`.

Depuis la racine du projet, exécuter :

```powershell
cd I:\OC\P5\projet5
poetry install
```

L'environnement Poetry installe les dépendances définies dans `pyproject.toml` et `poetry.lock`.

Le fichier `models/turnover_logistic.joblib` doit être présent pour permettre le démarrage de l'API.

## 4. Démarrage de FastAPI

Depuis la racine du projet :

```powershell
poetry run uvicorn app.main:app --app-dir API_Attrition_Fastapi --reload
```

L'application est accessible localement à l'adresse :

`http://127.0.0.1:8000`

Documentation interactive Swagger :

`http://127.0.0.1:8000/docs`

Schéma OpenAPI :

`http://127.0.0.1:8000/openapi.json`

L'option `--reload` est destinée au développement local.

## 5. Endpoints

### GET /health

Vérifie que l'application répond.

Exemple de réponse HTTP 200 :

```json
{
  "status": "ok"
}
```

### GET /model-info

Retourne les informations principales du modèle chargé.

```json
{
  "modele": "LogisticRegression",
  "charge": true,
  "nombre_features": 34,
  "seuil_decision": 0.4
}
```

### POST /predict

Reçoit les caractéristiques d'un salarié au format JSON et retourne une prédiction.

Exemple de requête :

```json
{
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
```

Exemple illustratif de réponse HTTP 200 (score fictif) :

```json
{
  "score_depart": 0.67,
  "prediction": 1,
  "interpretation": "Départ potentiel détecté",
  "seuil_decision": 0.4
}
```

Interprétation des classes :

- `0` : aucun départ potentiel détecté par le modèle.
- `1` : départ potentiel détecté par le modèle.

La classe est déterminée en comparant `score_depart` au seuil de décision de 0,40.

Le score est une sortie de `predict_proba()`. Il ne doit pas être interprété comme une probabilité réelle de démission parfaitement calibrée.

## 6. Validation des données

Pydantic contrôle les 30 champs d'entrée, leurs types, les catégories autorisées et certaines règles de cohérence métier.

Il vérifie notamment la cohérence entre expérience professionnelle, ancienneté dans l'entreprise, ancienneté au poste et temps écoulé depuis une promotion.

Les catégories sont limitées aux valeurs reconnues lors de l'entraînement du modèle.

Une requête non conforme retourne le code **HTTP 422**, accompagné d'informations sur les erreurs de validation.

## 7. Transformation des données

L'application convertit les 30 caractéristiques initiales en 34 features.

La fonction `construire_features()` reproduit les transformations du projet 4, notamment :

- La conversion de l'augmentation salariale en pourcentage numérique.
- Le ratio ancienneté / expérience professionnelle.
- Le ratio ancienneté dans le poste / ancienneté dans l'entreprise.
- Le ratio temps depuis promotion / ancienneté dans l'entreprise.
- Le ratio revenu mensuel / expérience professionnelle.

Les colonnes sont ensuite réordonnées pour correspondre exactement aux entrées attendues par le pipeline entraîné.

## 8. Tests automatisés

Le projet utilise Pytest et `TestClient` de FastAPI.

Depuis le dossier `API_Attrition_Fastapi` :

```powershell
poetry run python -m pytest -v tests
```

La suite comprend 20 cas de test validés lors du développement.

Ils couvrent le fonctionnement des endpoints, la validation des requêtes, la préparation des features, la cohérence entre l'API et le pipeline Scikit-learn ainsi que plusieurs cas d'erreur.

## 9. Limites connues

### Expérience professionnelle nulle

Le modèle du projet 4 utilise le ratio :

`revenu_mensuel / (annee_experience_totale + epsilon)`

Lorsque l'expérience totale est nulle, cette formule peut produire une valeur extrêmement élevée.

L'analyse du jeu d'entraînement a identifié neuf observations présentant cette situation. Ces valeurs influencent fortement les paramètres de standardisation de cette feature.

La version actuelle de l'API refuse donc provisoirement les requêtes associant une expérience professionnelle totale nulle et un revenu mensuel positif.

Cette restriction concerne la version actuelle du modèle et ne signifie pas qu'un salarié débutant constitue une donnée invalide.

Une prise en charge de ces profils nécessiterait de revoir le feature engineering, puis de réentraîner et réévaluer le modèle.

### Variable `ayant_enfants`

L'encodeur du modèle n'a appris qu'une seule modalité pour cette variable : `"Y"`.

L'API limite donc les valeurs acceptées à cette modalité. Cette restriction reflète une limitation des données d'entraînement et ne constitue pas une règle métier générale.

### Généralisation des prédictions

Les performances observées dans le projet 4 ne garantissent pas des performances identiques sur d'autres populations de salariés.

Les scores peuvent être affectés par les évolutions des caractéristiques des salariés, les différences entre entreprises et les changements de distribution des données.

### Confidentialité et utilisation RH

Le modèle est destiné à une démonstration pédagogique et technique. Il ne doit pas être utilisé seul pour prendre des décisions individuelles concernant les salariés.

Les données RH peuvent être personnelles ou sensibles. Un déploiement réel nécessiterait notamment une authentification, un contrôle des accès, une sécurisation des communications, une analyse des biais et une vérification des obligations applicables en matière de protection des données et de décisions automatisées.

L'API actuelle n'est pas présentée comme un service de production sécurisé.

## 10. Reproductibilité

Les dépendances sont gérées par Poetry.

Le fichier `poetry.lock` permet de conserver les versions résolues des bibliothèques. Le modèle sérialisé est chargé depuis `models/turnover_logistic.joblib`.

La compatibilité des versions de Scikit-learn et des autres bibliothèques doit être préservée lors d'un changement d'environnement.

Le chargement Joblib doit être réservé à des artefacts provenant de sources fiables.

## 11. Version

**API Attrition RH — version 1.0.0**

Modèle issu du projet 4, réutilisé sans modification de ses coefficients et avec le seuil métier de 0,40.