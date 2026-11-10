import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


# Chemin vers le fichier .env à la racine du projet P5
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

load_dotenv(ENV_PATH)


def ouvrir_connexion():
    """Ouvre une connexion à PostgreSQL."""

    return psycopg.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "attrition_rh"),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        connect_timeout=5
    )
