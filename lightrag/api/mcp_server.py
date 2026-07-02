"""MCP (Model Context Protocol) Server for LightRAG.

Exposes LightRAG's core capabilities as MCP tools so external AI agents
(Claude Desktop, Cursor, etc.) can query the knowledge graph and insert
documents via the MCP protocol over Streamable HTTP transport.

Mount this on the FastAPI app::

    from starlette.routing import Mount
    from lightrag.api.mcp_server import create_mcp_server, mcp_lifespan

    mcp = create_mcp_server(rag, top_k=60)
    app.router.routes.append(
        Mount("/mcp", app=mcp.streamable_http_app(), lifespan=mcp_lifespan)
    )
"""

from __future__ import annotations

import contextlib
from typing import Any

from lightrag.base import QueryParam
from lightrag.utils import logger


def create_mcp_server(rag, top_k: int = 60):
    """Create an MCP FastMCP server wrapping a LightRAG instance.

    Args:
        rag: A fully initialised ``LightRAG`` instance.
        top_k: Default number of top results for queries.

    Returns:
        ``FastMCP`` server ready to mount on FastAPI/Starlette.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "mcp package is required. Install with: pip install mcp"
        )

    mcp = FastMCP(
        "LightRAG",
        json_response=True,
        stateless_http=True,
    )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def lightrag_query(
        query: str,
        mode: str = "mix",
        max_results: int = 10,
    ) -> str:
        """Query the LightRAG knowledge graph.

        Use this tool when you need to retrieve information from documents
        that have been ingested into LightRAG. The knowledge graph combines
        vector search with entity-relationship reasoning.

        Args:
            query: The search query or question.
            mode: Retrieval mode — "local" (entity-focused), "global"
                (summary-based), "hybrid", "naive" (vector only), or
                "mix" (KG + vector, recommended).
            max_results: Maximum number of top results. Higher values
                return more context but use more tokens.
        """
        valid_modes = {"local", "global", "hybrid", "naive", "mix", "bypass"}
        if mode not in valid_modes:
            mode = "mix"

        param = QueryParam(
            mode=mode,
            top_k=max_results,
            chunk_top_k=max_results,
            stream=False,
        )
        result = await rag.aquery_llm(query, param=param)
        llm_response = result.get("llm_response", {})
        response_text = llm_response.get("content", "") or str(result)

        # Append image chunks not already in the LLM response,
        # but only from documents actually cited by the LLM
        chunks = result.get("data", {}).get("chunks", [])
        references = result.get("data", {}).get("references", [])
        referenced_files = {r.get("file_path", "") for r in references if r.get("file_path")}
        unused_images = [
            c["content"] for c in chunks
            if c.get("source_type") == "image"
            and c.get("image_path")
            and c.get("content", "")
            and c["content"] not in response_text
            and (not referenced_files or c.get("file_path", "") in referenced_files)
        ]
        if unused_images:
            response_text += "\n\n---\n### Related Images\n\n" + "\n\n".join(unused_images)

        return response_text

    @mcp.tool()
    async def lightrag_insert_text(
        text: str,
        file_path: str = "",
    ) -> str:
        """Insert a text document into the LightRAG knowledge graph.

        The document will be chunked, entities and relationships extracted,
        and added to the knowledge graph for future queries.

        Args:
            text: The full text content of the document.
            file_path: Optional source file path for citation (e.g.
                "notes/research.md"). Only used to identify the source
                in query results — does not read a file.
        """
        if not text.strip():
            return "Error: empty text"

        try:
            kwargs: dict[str, Any] = {}
            if file_path:
                kwargs["file_paths"] = [file_path]
            await rag.ainsert(text, **kwargs)
            return f"Document inserted successfully (length: {len(text)} chars)"
        except Exception as e:
            logger.error(f"lightrag_insert_text failed: {e}")
            return f"Error inserting document: {e}"

    @mcp.tool()
    async def lightrag_get_graph_info(
        label: str = "*",
        max_nodes: int = 20,
    ) -> str:
        """Get summary information about the LightRAG knowledge graph.

        Use this to understand what topics and entities are available
        before formulating a query.

        Args:
            label: Entity label filter (wildcard "*" for all).
            max_nodes: Maximum number of nodes to return in the summary.
        """
        try:
            # Build query using parameterised Cypher
            # (parameterised by the Pydantic model fields, not user input)
            knowledge_graph_inst = rag.chunk_entity_relation_graph
            node_data = []
            edge_data = []

            if hasattr(knowledge_graph_inst, "get_all_nodes"):
                nodes = knowledge_graph_inst.get_all_nodes()
                labels_set: set[str] = set()
                node_count = 0
                for node in nodes:
                    if label == "*" or node.get("labels", [""])[0] == label:
                        labels_set.update(node.get("labels", []))
                        node_count += 1
                node_data.append(
                    f"Total nodes: {node_count}, Labels: {sorted(labels_set)}"
                )

            if hasattr(knowledge_graph_inst, "get_all_edges"):
                edges = knowledge_graph_inst.get_all_edges()
                edge_count = len(edges)
                edge_data.append(f"Total edges: {edge_count}")

            lines = ["**Knowledge Graph Info**"]
            lines.extend(node_data[:max_nodes])
            lines.extend(edge_data[:max_nodes])
            return "\n".join(lines) if len(lines) > 1 else "Graph is empty"
        except Exception as e:
            logger.error(f"lightrag_get_graph_info failed: {e}")
            return f"Error fetching graph info: {e}"

    @contextlib.asynccontextmanager
    async def mcp_lifespan(app):
        """Lifespan that starts/stops the MCP session manager."""
        async with mcp.session_manager.run():
            yield

    # Attach lifespan to the server for caller convenience
    mcp._lifespan = mcp_lifespan  # type: ignore[attr-defined]

    return mcp
