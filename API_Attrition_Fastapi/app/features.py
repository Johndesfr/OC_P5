
import numpy as np
import pandas as pd

from .schemas import SalarieEntree


def construire_features(
    salarie: SalarieEntree,
    features_attendues: list[str]
) -> pd.DataFrame:
    """
    Transforme les 30 données d'entrée en 34 features,
    dans l'ordre exact attendu par le modèle.
    """

    # Conversion Pydantic -> dictionnaire -> DataFrame
    df = pd.DataFrame([salarie.model_dump()])

    # Même epsilon que dans le notebook P4
    eps = np.finfo(float).eps

    # 1. Augmentation salariale
    df["augmentation_salaire_precedente_pct"] = (
        df["augementation_salaire_precedente"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .astype(float)
    )

    # 2. Ancienneté / expérience
    df["ratio_anciennete_experience"] = (
        df["annees_dans_l_entreprise"]
        / (df["annee_experience_totale"] + eps)
    )

    # 3. Temps dans le poste / ancienneté
    df["ratio_temps_poste_anciennete"] = (
        df["annees_dans_le_poste_actuel"]
        / (df["annees_dans_l_entreprise"] + eps)
    )

    # 4. Temps depuis promotion / ancienneté
    df["ratio_promotion_anciennete"] = (
        df["annees_depuis_la_derniere_promotion"]
        / (df["annees_dans_l_entreprise"] + eps)
    )

    # 5. Revenu / expérience
    df["revenu_par_annee_experience"] = (
        df["revenu_mensuel"]
        / (df["annee_experience_totale"] + eps)
    )

    # Supprimer la version textuelle de l'augmentation
    df = df.drop(
        columns=["augementation_salaire_precedente"]
    )

    # Vérifier les colonnes avant prédiction
    absentes = set(features_attendues) - set(df.columns)
    inattendues = set(df.columns) - set(features_attendues)

    if absentes or inattendues:
        raise ValueError(
            f"Features absentes : {sorted(absentes)} ; "
            f"features inattendues : {sorted(inattendues)}"
        )

    # Garantir l'ordre exact des 34 features
    return df.loc[:, features_attendues].copy()
