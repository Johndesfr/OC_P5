
-- ===================================================
-- Projet P5 : API Attrition RH
-- Base de données : attrition_rh
-- ===================================================

-- 1. Profils de salariés pour les tests
CREATE TABLE IF NOT EXISTS salaries_test (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nom_profil VARCHAR(100) NOT NULL UNIQUE,
    donnees JSONB NOT NULL,
    date_creation TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT salaries_donnees_objet
        CHECK (jsonb_typeof(donnees) = 'object')
);


-- 2. Entrées transmises au modèle
CREATE TABLE IF NOT EXISTS predictions_entrees (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    salarie_test_id BIGINT REFERENCES salaries_test(id),
    donnees JSONB NOT NULL,
    date_requete TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT entrees_donnees_objet
        CHECK (jsonb_typeof(donnees) = 'object')
);


-- 3. Résultats produits par le modèle
CREATE TABLE IF NOT EXISTS predictions_sorties (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    entree_id BIGINT NOT NULL UNIQUE
        REFERENCES predictions_entrees(id),

    score_depart DOUBLE PRECISION NOT NULL
        CHECK (score_depart BETWEEN 0 AND 1),

    prediction SMALLINT NOT NULL
        CHECK (prediction IN (0, 1)),

    seuil_decision DOUBLE PRECISION NOT NULL
        CHECK (seuil_decision BETWEEN 0 AND 1),

    date_prediction TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
