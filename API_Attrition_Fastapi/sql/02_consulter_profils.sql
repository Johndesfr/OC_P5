SELECT
    id,
    nom_profil,
    donnees->>'age' AS age,
    donnees->>'poste' AS poste,
    donnees->>'revenu_mensuel' AS revenu
FROM salaries_test
ORDER BY nom_profil;