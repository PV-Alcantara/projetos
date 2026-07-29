"""
MCP Server Bulas do Paciente — tools module (Sara IA).

TOOL INVENTORY (11 total — topic tools via TOPIC_TOOL_SPECS):
    get_composition, get_presentations
    get_indication, get_mechanism
    get_contraindications, get_precautions
    get_posology, get_missed_dose
    get_adverse_reactions, get_overdose, get_storage
"""

import os
import re
import time
from collections import OrderedDict
from typing import Any

try:
    from server import client
except ImportError:
    import client


TABLE_CHUNKS = "dev_features_corporativo.sara_ai.bulas_chunks"

_CACHED_WAREHOUSE_ID: str | None = None

# ------------------------------------------------------------------
# Tool metadata
#
# Cada entrada gera automaticamente uma ferramenta MCP.
# Para adicionar uma nova tool basta acrescentar uma nova
# especificação neste dicionário.
# ------------------------------------------------------------------

TOPIC_TOOL_SPECS: OrderedDict[str, dict[str, Any]] = OrderedDict({
    "get_composition": {
        "response_key": "composicao",
        "section": "IDENTIFICACAO_DO_MEDICAMENTO",
        "topic": "COMPOSICAO",
        "doc": """Retorna a composição completa do medicamento: princípios ativos e excipientes.

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {composicao: [chunk_id, ...]}
""",
    },
    "get_presentations": {
        "response_key": "apresentacoes",
        "section": "IDENTIFICACAO_DO_MEDICAMENTO",
        "topic": "APRESENTACOES",
        "doc": """Retorna as apresentações comerciais disponíveis: formas, concentrações e embalagens.

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {apresentacoes: [chunk_id, ...]}
""",
    },
    "get_indication": {
        "response_key": "indicacao",
        "section": "INFORMACOES_AO_PACIENTE",
        "topic": "INDICACAO",
        "doc": """Retorna as indicações terapêuticas: para quais doenças ou condições o medicamento é indicado.

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {indicacao: [chunk_id, ...]}
""",
    },
    "get_mechanism": {
        "response_key": "funcionamento",
        "section": "INFORMACOES_AO_PACIENTE",
        "topic": "FUNCIONAMENTO",
        "doc": """Retorna como o medicamento funciona no organismo (mecanismo de ação farmacológica).

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {funcionamento: [chunk_id, ...]}
""",
    },
    "get_contraindications": {
        "response_key": "contraindicacoes",
        "section": "INFORMACOES_AO_PACIENTE",
        "topic": "CONTRAINDICACAO",
        "doc": """Retorna as contraindicações: perfis de pacientes e condições em que o medicamento não deve ser usado.

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {contraindicacoes: [chunk_id, ...]}
""",
    },
    "get_precautions": {
        "response_key": "precaucoes",
        "section": "INFORMACOES_AO_PACIENTE",
        "topic": "PRECAUCOES",
        "doc": """Retorna a seção completa de precauções, advertências e interações medicamentosas.

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {precaucoes: [chunk_id, ...]}
""",
    },
    "get_posology": {
        "response_key": "posologia",
        "section": "INFORMACOES_AO_PACIENTE",
        "topic": "POSOLOGIA",
        "doc": """Retorna a posologia completa: doses, frequência, via de administração e duração do tratamento.

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {posologia: [chunk_id, ...]}
""",
    },
    "get_missed_dose": {
        "response_key": "esquecimento",
        "section": "INFORMACOES_AO_PACIENTE",
        "topic": "ESQUECIMENTO",
        "doc": """Retorna as instruções para esquecimento de dose: o que fazer se uma dose for esquecida.

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {esquecimento: [chunk_id, ...]}
""",
    },
    "get_adverse_reactions": {
        "response_key": "reacoes_adversas",
        "section": "INFORMACOES_AO_PACIENTE",
        "topic": "REACOES_ADVERSAS",
        "doc": """Retorna as reações adversas conhecidas e sua frequência conforme descrito na bula.

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {reacoes_adversas: [chunk_id, ...]}
""",
    },
    "get_overdose": {
        "response_key": "superdosagem",
        "section": "INFORMACOES_AO_PACIENTE",
        "topic": "SUPERDOSE",
        "doc": """Retorna as orientações de superdosagem: sintomas e conduta em caso de overdose.

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {superdosagem: [chunk_id, ...]}
""",
    },
    "get_storage": {
        "response_key": "armazenamento",
        "section": "INFORMACOES_AO_PACIENTE",
        "topic": "ARMAZENAMENTO",
        "doc": """Retorna as condições de armazenamento: temperatura, umidade, validade e cuidados especiais.

Args:
    medication_id: Código ANVISA com 9 dígitos.
Returns: {armazenamento: [chunk_id, ...]}
""",
    },
})


def _sanitize_medication_id(medication_id: str) -> str:
    if not medication_id or not re.fullmatch(r"\d{9}", medication_id):
        raise ValueError("O parâmetro 'medication_id' deve conter exatamente 9 dígitos.")
    return medication_id


def _get_warehouse_id() -> str:
    global _CACHED_WAREHOUSE_ID
    if _CACHED_WAREHOUSE_ID is not None:
        return _CACHED_WAREHOUSE_ID
    wh_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
    if not wh_id:
        workspace = client.get_workspace_client()
        wh_list = list(workspace.warehouses.list())
        if not wh_list: raise RuntimeError("Nenhum SQL Warehouse encontrado.")
        for warehouse in wh_list:
            if warehouse.state and warehouse.state.value == "RUNNING":
                wh_id = warehouse.id
                break
        if not wh_id and wh_list:
            wh_id = wh_list[0].id
    _CACHED_WAREHOUSE_ID = wh_id
    return wh_id


def _get_column_names(manifest) -> list[str]:
    schema_data = getattr(manifest, "schema")
    cols = getattr(schema_data, "columns")
    return [getattr(col, "name") for col in cols]


def _execute_sql(sql: str, max_retries: int = 2) -> list[dict[str, Any]]:
    workspace = client.get_workspace_client()
    for attempt in range(max_retries + 1):
        try:
            result = workspace.statement_execution.execute_statement(
                warehouse_id=_get_warehouse_id(),
                statement=sql,
                wait_timeout="30s",
            )
            if result.result is None or result.result.data_array is None:
                return []
            columns = _get_column_names(result.manifest)
            return [dict(zip(columns, row)) for row in result.result.data_array]
        except Exception as exc:
            if attempt < max_retries and any(k in str(exc).lower() for k in ("timeout", "unavailable")):
                time.sleep(2 ** attempt)
                continue
            raise
    return []


def _get_topic_chunk_ids(medication_id: str, section: str, topic: str) -> list[str]:
    sql = (
        f"SELECT chunk_id FROM {TABLE_CHUNKS}"
        f" WHERE codigo_anvisa = '{medication_id}'"
        f" AND secao = '{section}'"
        f" AND topico = '{topic}'"
        f" ORDER BY pagina, chunk_id"
        f" LIMIT 200"
    )
    rows = _execute_sql(sql)
    return [row["chunk_id"] for row in rows if row.get("chunk_id")]


def _make_topic_tool(tool_name: str, spec: dict):
    def tool(medication_id: str) -> dict:
        try:
            safe_id = _sanitize_medication_id(medication_id)
            return {spec["response_key"]: _get_topic_chunk_ids(safe_id, spec["section"], spec["topic"])}
        except Exception as exc:
            return {"error": str(exc)}

    tool.__name__ = tool_name
    tool.__doc__ = spec["doc"]
    tool.__annotations__ = {"medication_id": str, "return": dict}
    return tool


def load_tools(mcp_server) -> None:
    """Register all MCP tools."""
    for tool_name, spec in TOPIC_TOOL_SPECS.items():
        mcp_server.tool(_make_topic_tool(tool_name, spec))
