# %%
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from pymongo import MongoClient

# pylint: disable=import-error,wrong-import-position

MODULE_NAME = os.getenv("AGENTS_MODULE")
ALSO_MODULE = True
if ALSO_MODULE:
    current_file = Path(__file__).resolve()
    package_root = current_file.parent
    while package_root.name != MODULE_NAME:
        package_root = package_root.parent

    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
load_dotenv(override=True)

os.chdir(package_root)


from tpp_agentic_graph_v4.tools.get_office_configurations_pipeline import (  # noqa: E402
    pipeline,
)

# ---------------------------------------------------------------------------
# MongoDB config
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "agentic-memory"
COLLECTION_NAME = "OfficeBuckets"


def _get_collection():
    """Return the OfficeBuckets collection or None on failure."""

    try:
        client = MongoClient(MONGO_URI)
        client.admin.command("ismaster")
        return client[DB_NAME][COLLECTION_NAME]
    except Exception as exc:  # pragma: no cover
        print(f"MongoDB connection error: {exc}")
        return None


collection = _get_collection()

OFFICE_NAME_TO_LEGACY_ID: dict[str, int] = {}
if collection is not None:
    OFFICE_NAME_TO_LEGACY_ID = {
        doc["_id"]: doc["legacyId"]
        for doc in collection.aggregate(
            [
                {"$match": {"_id.tenant": "BChile"}},
                {
                    "$group": {
                        "_id": "$office",
                        "legacyId": {"$first": "$_id.legacyId"},
                    }
                },
            ]
        )
    }


def handler(office_name: str, date: str) -> str:
    """
    Returns the office configurations for a given office name and date.
    """

    # Tenant is fixed for this tool
    tenant = "BChile"

    # Convert office name to legacy ID
    legacy_id = OFFICE_NAME_TO_LEGACY_ID.get(office_name)
    if legacy_id is None:
        return f"Error: oficina desconocida '{office_name}'."

    if collection is None:
        return "Error: No se pudo conectar a MongoDB."

    result = collection.aggregate(
        [
            {
                "$match": {
                    "_id.tenant": tenant,
                    "_id.legacyId": legacy_id,
                    "_id.date": date,
                }
            }
        ]
        + pipeline
    )

    # Logica y post procesado lado server
    result = {
        "office_name": office_name,
        "date": date,
        "configurations": list(result),
    }

    return result


@tool(
    description="Get office configurations for a given office name and date.",
)
def get_office_configurations(office_name: str, date: str) -> str:
    """
    Tool to get office configurations for a given office name and date.
    """
    return handler(office_name, date)


# CLI quick test

if __name__ == "__main__":
    from pprint import pprint

    # "office_name": "160 - Ñuñoa",
    # "today_date_str": "2025-05-08",
    # "start_date": "2025-06-01",
    # "num_days": 30,

    pprint(
        get_office_configurations.invoke(
            {"office_name": "160 - Ñuñoa", "date": "2025-05-08"}
        )
    )
