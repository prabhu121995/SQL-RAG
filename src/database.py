import os
import logging
import requests
from langchain_community.utilities import SQLDatabase

from src.config import settings

logger = logging.getLogger(__name__)


def ensure_db_downloaded() -> str:
    """Download tickets.db from GitHub if it's not already on disk."""
    path = settings.tickets_db_path

    if os.path.exists(path) and os.path.getsize(path) > 0:
        logger.info("Using existing DB at %s", path)
        return path

    logger.info("Downloading tickets.db from %s", settings.tickets_url)
    response = requests.get(settings.tickets_url, timeout=60)
    response.raise_for_status()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(response.content)

    logger.info("Saved tickets.db → %s", path)
    return path


def get_sql_database() -> SQLDatabase:
    """Return a LangChain SQLDatabase instance bound to the local sqlite file."""
    db_path = ensure_db_downloaded()
    return SQLDatabase.from_uri(f"sqlite:///{db_path}")