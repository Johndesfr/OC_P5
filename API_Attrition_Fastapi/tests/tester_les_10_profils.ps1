# Depuis I:\OC\P5\projet5\API_Attrition_Fastapi\tests\fixtures
# Démarrer l'API dans un autre terminal avant de lancer ce script.
$endpoint = "http://127.0.0.1:8000/predict"
$racine = Join-Path $PSScriptRoot "fixtures"
$fichiers = @(Get-ChildItem -Path $racine -Filter "*.json")
Write-Host "Nombre de profils JSON trouvés : $($fichiers.Count)"
Get-ChildItem -Path $racine -Filter "*.json" | Sort-Object Name | ForEach-Object {
    $attendu = if ($_.Name -like "*depart*") { 1 } else { 0 }
    $payload = Get-Content -Path $_.FullName -Raw -Encoding UTF8
    try {
        $response = Invoke-RestMethod -Uri $endpoint -Method Post -ContentType "application/json; charset=utf-8" -Body $payload
        [PSCustomObject]@{
            Profil = $_.BaseName
            Hypothese = $attendu
            Prediction = $response.prediction
            Score = [math]::Round([double]$response.score_depart, 4)
            Concordance = ([int]$response.prediction -eq $attendu)
        }
    }
    catch {
        Write-Warning ("Echec pour {0}: {1}" -f $_.Name, $_.Exception.Message)
    }
} | Format-Table -AutoSize
