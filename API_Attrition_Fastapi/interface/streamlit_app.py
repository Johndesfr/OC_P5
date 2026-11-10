
import os

import requests
import streamlit as st


API_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


# ============================================================
# Configuration de l'interface
# ============================================================

st.set_page_config(
    page_title="Attrition RH",
    layout="centered"
)

st.title("Prédiction de l'attrition RH")

st.caption(
    "Application de prédiction avec FastAPI, "
    "PostgreSQL et régression logistique."
)

st.divider()


# ============================================================
# 1. Récupération des salariés depuis FastAPI
# ============================================================

try:
    response = requests.get(
        f"{API_URL}/salaries",
        timeout=5
    )

    response.raise_for_status()
    profils = response.json()

except (requests.RequestException, ValueError) as erreur:
    st.error(
        "Impossible de récupérer les salariés depuis FastAPI."
    )
    st.code(str(erreur))
    st.stop()


if not profils:
    st.warning("Aucun salarié disponible dans PostgreSQL.")
    st.stop()


# ============================================================
# 2. Sélection d'un salarié
# ============================================================

profils_par_id = {
    int(profil["id"]): profil
    for profil in profils
}

identifiants = sorted(profils_par_id)


salarie_id = st.selectbox(
    "Sélectionner un salarié",
    options=identifiants,
    format_func=lambda identifiant: (
        f"ID {identifiant} — "
        f"{profils_par_id[identifiant]['nom_profil']}"
    )
)

profil = profils_par_id[salarie_id]

st.write("**Poste :**", profil["poste"])
st.write("**Identifiant PostgreSQL :**", salarie_id)

st.divider()


# ============================================================
# 3. Calcul de la prédiction
# ============================================================

if st.button(
    "Calculer la prédiction",
    type="primary",
    use_container_width=True
):

    st.session_state.pop("derniere_prediction", None)

    with st.spinner("Calcul de la prédiction..."):

        try:
            response = requests.post(
                f"{API_URL}/predict/{salarie_id}",
                timeout=30
            )

            response.raise_for_status()
            resultat = response.json()

        except (requests.RequestException, ValueError) as erreur:
            st.error("La prédiction a échoué.")
            st.code(str(erreur))

        else:
            st.session_state["derniere_prediction"] = {
                "salarie_id": salarie_id,
                "resultat": resultat
            }


# ============================================================
# 4. Affichage du résultat
# ============================================================

derniere = st.session_state.get(
    "derniere_prediction"
)

if derniere and derniere["salarie_id"] == salarie_id:

    resultat = derniere["resultat"]

    score = float(resultat["score_depart"])
    prediction = int(resultat["prediction"])
    seuil = float(resultat["seuil_decision"])

    st.subheader("Résultat de la prédiction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Probabilité de départ estimée",
            f"{score:.1%}"
        )

    with col2:
        st.metric(
            "Seuil de décision",
            f"{seuil:.0%}"
        )

    st.progress(score)

    if prediction == 1:
        st.warning("Départ potentiel détecté")
    else:
        st.success("Départ non détecté")

    st.caption(
        "Cette estimation statistique n'est pas "
        "une certitude de départ."
    )

    st.divider()

    st.subheader("Traçabilité PostgreSQL")

    st.write(
        "**ID de l'entrée :**",
        resultat["entree_id"]
    )

    st.write(
        "**ID du résultat :**",
        resultat["sortie_id"]
    )

    with st.expander("Afficher la réponse JSON"):
        st.json(resultat)
