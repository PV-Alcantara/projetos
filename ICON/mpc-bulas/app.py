"""
MCP Server Bulas do Paciente — app entrypoint (Sara IA).
"""

from mcp.server.fastmcp import FastMCP

from tools import load_tools


mcp = FastMCP(
    name="Sara Bulas MCP",
    instructions="""
Servidor MCP especializado em bulas de medicamentos para pacientes.

As ferramentas deste servidor retornam apenas chunk_ids relacionados
a tópicos específicos da bula, como composição, contraindicações,
precauções, posologia, reações adversas, superdosagem e armazenamento.

Este MCP não recupera o texto completo dos chunks e não executa Vector Search.
A recuperação textual deve ser feita por outra ferramenta ou pelo agente consumidor.
""",
)


load_tools(mcp)


if __name__ == "__main__":
    mcp.run()