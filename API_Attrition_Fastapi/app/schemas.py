
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class SalarieEntree(BaseModel):
    """Caractéristiques initiales d'un salarié."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True
    )

    # Données RH
    age: int = Field(ge=16, le=100)
    genre: str = Field(min_length=1)
    revenu_mensuel: int = Field(ge=0)
    statut_marital: str = Field(min_length=1)
    departement: str = Field(min_length=1)
    poste: str = Field(min_length=1)
    nombre_experiences_precedentes: int = Field(ge=0)
    nombre_heures_travailless: int = Field(ge=0)
    annee_experience_totale: int = Field(ge=0)
    annees_dans_l_entreprise: int = Field(ge=0)
    annees_dans_le_poste_actuel: int = Field(ge=0)

    # Évaluations professionnelles
    satisfaction_employee_environnement: int = Field(ge=1, le=4)
    note_evaluation_precedente: int = Field(ge=1, le=4)
    niveau_hierarchique_poste: int = Field(ge=1, le=5)
    satisfaction_employee_nature_travail: int = Field(ge=1, le=4)
    satisfaction_employee_equipe: int = Field(ge=1, le=4)
    satisfaction_employee_equilibre_pro_perso: int = Field(ge=1, le=4)
    note_evaluation_actuelle: int = Field(ge=1, le=4)
    heure_supplementaires: str = Field(min_length=1)
    augementation_salaire_precedente: str = Field(
        pattern=r"^\d+(?:\.\d+)?\s*%$"
    )

    # Données complémentaires
    nombre_participation_pee: int = Field(ge=0)
    nb_formations_suivies: int = Field(ge=0)
    nombre_employee_sous_responsabilite: int = Field(ge=0)
    distance_domicile_travail: int = Field(ge=0)
    niveau_education: int = Field(ge=1, le=5)
    domaine_etude: str = Field(min_length=1)
    ayant_enfants: str = Field(min_length=1)
    frequence_deplacement: str = Field(min_length=1)
    annees_depuis_la_derniere_promotion: int = Field(ge=0)
    annes_sous_responsable_actuel: int = Field(ge=0)

    
class PredictionSortie(BaseModel):
    """Structure de la réponse de prédiction."""

    score_depart: float = Field(ge=0, le=1)
    prediction: Literal[0, 1]
    interpretation: str
    seuil_decision: float = Field(ge=0, le=1)

