"""cypher_runner.py
Utility helpers to execute ad-hoc Cypher queries against the Neo4j instance
configured through environment variables.

Usage example
-------------
>>> from cypher_runner import run_cypher
>>> run_cypher("MATCH (n) RETURN n LIMIT 5")

This module keeps a single Neo4j driver alive for the lifetime of the Python
process and shares it across calls.
"""

# %%
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase, Record
from neo4j.exceptions import ServiceUnavailable, SessionExpired

# --------------------------------------------------------------------------- #
# Driver initialisation (singleton)
# --------------------------------------------------------------------------- #

load_dotenv(override=True)

_NEO4J_URI: str | None = os.getenv("NEO4J_CONNECTION_URI_UPGRADED")
_NEO4J_USER: str | None = os.getenv("NEO4J_USERNAME_UPGRADED")
_NEO4J_PWD: str | None = os.getenv("NEO4J_PASSWORD_UPGRADED")

if not all([_NEO4J_URI, _NEO4J_USER, _NEO4J_PWD]):
    raise EnvironmentError(
        "Missing Neo4j connection variables (NEO4J_CONNECTION_URI_UPGRADED, "
        "NEO4J_USERNAME_UPGRADED, NEO4J_PASSWORD_UPGRADED)."
    )

_DRIVER: Driver = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PWD))
_DRIVER.verify_connectivity()

# --------------------------------------------------------------------------- #
# Public helper
# --------------------------------------------------------------------------- #


def run_cypher(
    query: str,
    parameters: Dict[str, Any] | None = None,
    *,
    close_after: bool = False,
) -> List[Dict[str, Any]]:
    """Run an arbitrary Cypher *read* query and return the results.

    Parameters
    ----------
    query:
        The Cypher query string to execute.
    parameters:
        Optional dictionary of parameters to pass to the query, following Neo4j
        best-practices for parameterised queries.
    close_after:
        If *True*, the shared Neo4j driver will be closed immediately after the
        query finishes and results are collected. Set this to *False* (default)
        when you intend to run multiple queries in the same Python process.

    Returns
    -------
    list[dict[str, Any]]
        List of result records as ordinary Python dictionaries, where keys are
        the column names specified in the Cypher `RETURN` clause.
    """
    if parameters is None:
        parameters = {}

    global _DRIVER  # we may recreate it on failure

    try:
        with _DRIVER.session() as session:
            result = session.run(query, parameters)
            records: List[Record] = list(result)
    except (SessionExpired, ServiceUnavailable):
        # Attempt one automatic reconnection
        try:
            _DRIVER.close()
        except Exception:
            pass

        _DRIVER = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PWD))
        _DRIVER.verify_connectivity()

        with _DRIVER.session() as session:
            result = session.run(query, parameters)
            records = list(result)

    # Close the driver if requested *after* we have consumed all records.
    if close_after:
        close_driver()

    return [rec.data() for rec in records]


def close_driver() -> None:
    """Close the shared Neo4j driver (optional helper)."""
    _DRIVER.close()


if __name__ == "__main__":
    # --------------------------------------------------------------------------- #
    # Cargar consultas de sample_queries.yaml
    # --------------------------------------------------------------------------- #

    yaml_path = Path(__file__).with_name("sample_queries.yaml")
    sample_queries = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    # cada elemento: {'pregunta': str, 'cypher_query': str}

    # --------------------------------------------------------------------------- #
    # Ejecutar cada consulta y mostrar resultados
    # --------------------------------------------------------------------------- #

    for item in sample_queries:
        title = item["pregunta"]
        cypher = item["cypher_query"]
        print(f"\n📌 {title}\nCypher: {cypher.strip()}\n→ Resultados:")
        rows = run_cypher(cypher)
        for row in rows:
            print(row)

    # Close driver at the very end
    close_driver()
