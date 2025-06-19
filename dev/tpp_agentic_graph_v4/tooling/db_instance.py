#!/usr/bin/env python
# %%
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import sqlalchemy
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)

# Attempt to load configuration from config.json in parent directories
config = {}
for directory in Path(__file__).resolve().parents:
    config_path = directory / "config.json"
    if config_path.is_file():
        try:
            with config_path.open() as f:
                config = json.load(f)
        except Exception:
            pass
        break

# Use config values with fallback to environment variables
_DB_USERNAME: str = config.get("DB_USERNAME") or os.getenv("DB_USERNAME", "")
_DB_PASSWORD: str = config.get("DB_PASSWORD") or os.getenv("DB_PASSWORD", "")
_DB_SERVER: str = config.get("DB_SERVER") or os.getenv("DB_SERVER", "")
_DB_DATABASE: str = config.get("DB_DATABASE") or os.getenv("DB_DATABASE", "")
_DB_PORT: int = int(config.get("DB_PORT") or os.getenv("DB_PORT", "1433"))
# %%
logging.basicConfig()
logger: logging.Logger = logging.getLogger("sqlalchemy.engine")
# logger.setLevel(logging.DEBUG)

if not all([_DB_USERNAME, _DB_PASSWORD, _DB_SERVER, _DB_DATABASE]):
    raise ValueError("Missing required database environment variables")

logger.info(f"{_DB_USERNAME = }\n{_DB_PASSWORD = }\n{_DB_SERVER = }\n{_DB_DATABASE = }")


# mssql: adaptador para Microsoft SQL Server database
# pymssql: Python driver.
connection_url = sqlalchemy.engine.url.URL.create(
    drivername="mssql+pymssql",
    username=_DB_USERNAME,
    password=_DB_PASSWORD,
    host=_DB_SERVER,
    database=_DB_DATABASE,
    port=_DB_PORT,
)

_engine = sqlalchemy.create_engine(
    url=connection_url,
    pool_recycle=3600,
    pool_pre_ping=True,
    pool_size=10,
)

assert isinstance(_engine, sqlalchemy.engine.base.Engine), (
    "SQLAlchemy Engine was not properly instantiated"
)


# SQLAlchemyInstrumentor().instrument(
#     engine=_engine,
# )

# These are the ones used by the LLM model
_relevant_tables: list[str] = ["Atenciones", "EjeEstado", "Series", "Oficinas"]

# db = SQLDatabase(
#     engine=_engine, include_tables=_relevant_tables, sample_rows_in_table_info=5
# )

# assert isinstance(
#     db, langchain_community.utilities.sql_database.SQLDatabase
# ), "langchain_community's SQLAlchemy wrapper was not properly instantiated"


# region Other methods


class GetOfficesResponseOffices(BaseModel):
    name: str  #  "Oficina 1"
    ref: str  #  "ofc_01"
    region: str = "Oficinas"  #  "Norte"


class GetOfficesResponse(BaseModel):
    offices: list[GetOfficesResponseOffices]


def get_offices(group_by_zone=False) -> GetOfficesResponse:
    """Get the list of offices from the database.

    Current implementation just returns a list of all offices as a string.
    """
    with _engine.connect() as conn:
        if group_by_zone:
            # TODO: Implement zone grouping
            return GetOfficesResponse(offices=[])
        else:
            offices = conn.execute(
                sqlalchemy.text(
                    """
                    SELECT o.[Oficina], o.[IdOficina]
                    FROM [Oficinas] o
                    JOIN [Atenciones] a ON a.IdOficina = o.IdOficina
                    WHERE o.fDel = 0
                    GROUP BY o.[Oficina], o.[IdOficina]
                    HAVING COUNT(*) > 0
                    ORDER BY [Oficina] ASC
                    """
                )
            )
            rows = offices.all()

            return GetOfficesResponse(
                offices=[
                    GetOfficesResponseOffices(name=row[0], ref=str(row[1]))
                    for row in rows
                ]
            )


def get_just_office_names() -> list[str]:
    """Not designed to be used with a chain per-se"""
    import pandas as pd

    query = """
    SELECT
          o.[Oficina] AS "Oficina"
    FROM [dbo].[Oficinas] o
    """

    with _engine.connect() as conn:
        data: pd.DataFrame = pd.read_sql_query(query, conn)

    return data["Oficina"].to_list()


def get_last_database_update() -> datetime:
    """Get the last time the database was updated."""
    with _engine.connect() as conn:
        query = 'SELECT MAX(a.[FH_Emi]) AS "Ultima atencion" FROM [dbo].[Atenciones] a'
        last_update = conn.execute(sqlalchemy.text(query)).scalar()

        if last_update is None:
            raise ValueError("No database update timestamp found")

        return last_update


if __name__ == "__main__":
    print(get_just_office_names())

# %%
