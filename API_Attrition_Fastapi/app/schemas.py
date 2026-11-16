
from pydantic import BaseModel, Field, ConfigDict, model_validator
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
    genre: Literal["F", "M"]
    revenu_mensuel: int = Field(ge=0)
    statut_marital: Literal[
        "Célibataire",
        "Divorcé(e)",
        "Marié(e)"
    ]
    departement: Literal[
    "Commercial",
    "Consulting",
    "Ressources Humaines"
    ]
    poste: Literal[
        "Assistant de Direction",
        "Cadre Commercial",
        "Consultant",
        "Directeur Technique",
        "Manager",
        "Représentant Commercial",
        "Ressources Humaines",
        "Senior Manager",
        "Tech Lead"
    ]
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
    heure_supplementaires: Literal[
        "Non",
        "Oui"
    ]
    augementation_salaire_precedente: str = Field(
        pattern=r"^\d+(?:\.\d+)?\s*%$"
    )

    # Données complémentaires
    nombre_participation_pee: int = Field(ge=0)
    nb_formations_suivies: int = Field(ge=0)
    nombre_employee_sous_responsabilite: int = Field(ge=0)
    distance_domicile_travail: int = Field(ge=0)
    niveau_education: int = Field(ge=1, le=5)
    domaine_etude: Literal[
        "Autre",
        "Entrepreunariat",
        "Infra & Cloud",
        "Marketing",
        "Ressources Humaines",
        "Transformation Digitale"
    ]
    ayant_enfants: Literal["Y"]
    frequence_deplacement: Literal[
        "Aucun",
        "Frequent",
        "Occasionnel"
    ]
    annees_depuis_la_derniere_promotion: int = Field(ge=0)
    annes_sous_responsable_actuel: int = Field(ge=0)

    
    @model_validator(mode="after")
    def verifier_coherence_anciennete(self):

        # Règle 1 : l'ancienneté dans l'entreprise
        # ne dépasse pas l'expérience totale.
        if (
            self.annees_dans_l_entreprise
            > self.annee_experience_totale
        ):
            raise ValueError(
                "L'ancienneté dans l'entreprise ne peut pas "
                "dépasser l'expérience professionnelle totale."
            )

        # Règle 2 : l'ancienneté dans le poste actuel
        # ne dépasse pas l'ancienneté dans l'entreprise.
        if (
            self.annees_dans_le_poste_actuel
            > self.annees_dans_l_entreprise
        ):
            raise ValueError(
                "L'ancienneté dans le poste actuel ne peut pas "
                "dépasser l'ancienneté dans l'entreprise."
            )
        # Règle 3 : cohérence des promotions
        if (
            self.annees_depuis_la_derniere_promotion
            > self.annees_dans_l_entreprise
        ):
            raise ValueError(
                "Les années depuis la dernière promotion "
                "ne peuvent pas dépasser l'ancienneté "
                "dans l'entreprise."
            )

        # Règle 4 : éviter un ratio revenu/expérience extrême
        if (
            self.annee_experience_totale == 0
            and self.revenu_mensuel > 0
        ):
            raise ValueError(
                "Le modèle actuel ne prend pas en charge "
                "un revenu positif avec zéro année "
                "d'expérience professionnelle complète."
            )    
        return self


    
class PredictionSortie(BaseModel):
    """Structure de la réponse de prédiction."""

    score_depart: float = Field(ge=0, le=1)
    prediction: Literal[0, 1]
    interpretation: str
    seuil_decision: float = Field(ge=0, le=1)

