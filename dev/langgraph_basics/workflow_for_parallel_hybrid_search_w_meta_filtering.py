"""
This workflow is a parallel workflow that uses the chains_for_hybrid_search_and_metadata_filtering.py file to create a workflow that can be used to search for information in the vectorstore.
"""

# %%
import asyncio
import os
from operator import add  # add at top near other imports
from typing import Annotated, Literal
from uuid import uuid4

import aiosqlite
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send
from pinecone import PineconeApiException  # For catching filter errors
from pydantic import BaseModel, Field, ValidationError

from dev.langgraph_basics.chains_for_hybrid_search_and_metadata_filtering import (
    chain_for_filter_fields,
    chain_for_filter_generation,
)
from dev.langgraph_basics.simple_hybrid_search_w_metadata_filtering import retriever
from dev.langgraph_basics.simple_ReAct import (
    LastLLMResponse,
    reduce_docs,
)

load_dotenv(override=True)


# Create a Runnable that exposes both sync and async retrieval


def _sync_retrieve(query: str, filter: dict | None = None) -> list[Document]:
    """Synchronous wrapper around the Pinecone retriever invoke method."""
    if filter is None:
        return retriever.invoke(query)
    return retriever.invoke(query, filter=filter)


async def _async_retrieve(query: str, filter: dict | None = None) -> list[Document]:
    """Execute the synchronous ``retriever.invoke`` call in a thread so it can
    be awaited from asyncio code. This avoids the issues with the built-in
    ``retriever.ainvoke`` method, which currently does not accept a *filter*
    argument.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_retrieve, query, filter)


# Expose the runnable with both sync and async behaviour.
async_retriever = RunnableLambda(_sync_retrieve, afunc=_async_retrieve)


class OneQuery(BaseModel):
    """One query."""

    query_str: str


class GeneratedQueries(BaseModel):
    """Generated queries."""

    queries_list: list[OneQuery]


class RetrievalGraphState(MessagesState):
    """State of the graph."""

    rag_input: Annotated[list[HumanMessage], add_messages] = Field(
        default_factory=lambda: [HumanMessage(content="")]
    )
    ai_generated_response: LastLLMResponse = Field(
        default_factory=lambda: LastLLMResponse(response="")
    )
    documents: Annotated[list[Document], reduce_docs] = Field(
        default_factory=lambda: []
    )
    generated_queries: GeneratedQueries = Field(
        default_factory=lambda: GeneratedQueries(queries_list=[])
    )
    query: str = Field(default_factory=lambda: "")
    completed_queries: Annotated[list[str], add] = Field(default_factory=list)
    scores_matrix: dict = Field(default_factory=dict)
    scores_rows: Annotated[list[dict], add] = Field(default_factory=list)


llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0,
    max_tokens=None,
    timeout=1200,
    max_retries=5,
    streaming=True,
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
PROMPT_FOR_GENERATE_QUERIES = """
Based on the user question, return seven (7) queries useful to retrieve documents in parallel.
Queries should expand/enrich the semantic space of the user question.
User question: {user_question}
"""
TEMPLATE_FOR_GENERATE_QUERIES = ChatPromptTemplate.from_template(
    PROMPT_FOR_GENERATE_QUERIES
)
chain_for_generate_queries = TEMPLATE_FOR_GENERATE_QUERIES | llm.with_structured_output(
    GeneratedQueries, method="function_calling"
)


async def generate_queries(
    state: RetrievalGraphState,
) -> Command[Literal["retrieve_in_parallel"]]:
    """Node that generates queries."""
    generated_queries = chain_for_generate_queries.invoke(
        {"user_question": state["rag_input"][-1].content}
    )
    return Command(
        goto="retrieve_in_parallel",
        update={"generated_queries": generated_queries},
    )


async def retrieve_in_parallel(state: RetrievalGraphState) -> Command[list[Send]]:
    """Node that retrieves documents in parallel."""
    lista_de_queries = [
        query.query_str for query in state["generated_queries"].queries_list
    ]
    print(f"lista_de_queries: {lista_de_queries}")
    sends = [
        Send(
            "metadata_filtering_and_hybrid_search_node",
            {"query": query, "num_pending": len(lista_de_queries)},
        )
        for query in lista_de_queries
    ]
    return Command(goto=sends)


async def metadata_filtering_and_hybrid_search_node(
    state: RetrievalGraphState,
) -> Command[Literal["metadata_filtering_and_hybrid_search_node", END]]:
    """Node that extracts fields and generates the final metadata filter.

    If the structured output from the LLM cannot be parsed (for example, the
    returned JSON does not contain the required ``filter`` key), we gracefully
    fall back to an *empty* filter to avoid hard failures in the workflow.
    """

    query_str = state["query"]

    extracted_fields = await chain_for_filter_fields.ainvoke({"query": query_str})
    extracted_json = extracted_fields.model_dump_json(indent=2)

    # Try to generate the final filter with structured output.
    try:
        final_filter_obj = await chain_for_filter_generation.ainvoke(
            {
                "query": query_str,
                "extracted_fields": extracted_json,
            }
        )
        filter_dict = final_filter_obj.filter  # type: ignore[attr-defined]

        docs = await async_retriever.ainvoke(query_str, filter=filter_dict)
        if len(docs) == 0:
            # retry this node.
            return Command(goto="metadata_filtering_and_hybrid_search_node")

    except (ValidationError, PineconeApiException):
        # retry this node.
        return Command(goto="metadata_filtering_and_hybrid_search_node")

    for doc in docs:
        doc.metadata["query"] = query_str

    print(f"query_str: {query_str}")
    for res in docs:
        score = res.metadata.get("score", "N/A")
        print(f"* [score: {score:.3f}] {res.page_content} [{res.metadata}]")

    # Build a row dict for this query
    row = {"query": query_str}
    for d in docs:
        enzyme = d.metadata.get("enzyme")
        score = d.metadata.get("score")
        if enzyme and score is not None:
            row[enzyme] = score

    return Command(
        update={
            "documents": docs,
            "completed_queries": [query_str],
            "scores_rows": [row],
        },
        goto="check_completion",
    )


async def check_completion(
    state: RetrievalGraphState,
) -> Command[Literal["print_scores_matrix"]]:
    scores_matrix_existing = state.get("scores_matrix", {})
    if scores_matrix_existing:
        return Command(goto="print_scores_matrix")

    total_queries = len(state["generated_queries"].queries_list)
    completed = state.get("completed_queries", [])
    if len(completed) < total_queries:
        # Not all queries have completed yet
        return Command(goto="print_scores_matrix")

    queries = [q.query_str for q in state["generated_queries"].queries_list]
    enzymes: set[str] = set()
    for row in state["scores_rows"]:
        enzymes.update(k for k in row.keys() if k != "query")

    data = {"query": queries}
    for enzyme in sorted(enzymes):
        data[enzyme] = []
        for q in queries:
            # find row for this query
            matching = next((r for r in state["scores_rows"] if r["query"] == q), None)
            val = matching.get(enzyme) if matching else None  # type: ignore[union-attr]
            data[enzyme].append(val)

    return Command(goto="print_scores_matrix", update={"scores_matrix": data})


async def print_scores_matrix(state: RetrievalGraphState):
    """Print the scores matrix."""
    import pandas as pd

    df = pd.DataFrame(state["scores_matrix"])
    print(f"#######################scores_matrix DF: {df}")
    return Command(goto="generate_clustergram")


async def generate_clustergram(state: RetrievalGraphState):
    """Generate and display a simple clustergram using Dash Bio.

    If dash_bio is not installed, fall back to a basic heatmap so the
    workflow still completes without errors.
    """
    scores_matrix = state["scores_matrix"]
    df = pd.DataFrame(scores_matrix).set_index("query").fillna(0)

    # --- Identify high-score enzyme group (to be highlighted) ---
    col_means = df.mean(axis=0, skipna=True)
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        col_linkage_global = linkage(
            pdist(df.values.T, metric="euclidean"), method="average"
        )
        cluster_labels = fcluster(col_linkage_global, t=2, criterion="maxclust")

        # Determine high-score cluster
        clusters = set(cluster_labels)
        avg_per_cluster = {
            cl: col_means[np.array(cluster_labels) == cl].mean() for cl in clusters
        }
        high_cluster = max(avg_per_cluster, key=avg_per_cluster.get)

        high_cols = [
            enzyme
            for idx, enzyme in enumerate(df.columns)
            if cluster_labels[idx] == high_cluster
        ]

    except ModuleNotFoundError:
        # Fallback: threshold by median
        threshold = col_means.median()
        high_cols = [
            enzyme for enzyme, score in col_means.items() if score >= threshold
        ]

    try:
        import dash_bio as dashbio  # type: ignore

        # Dash Bio's Clustergram automatically computes hierarchical
        # clustering and shows both dendrograms and heatmap in one call.
        import plotly.express as px

        clustergram_component = dashbio.Clustergram(
            data=df.values,
            row_labels=df.index.tolist(),
            column_labels=df.columns.tolist(),
            height=600,
            width=1500,
            color_map=px.colors.sequential.deep[::-1],
            display_ratio=[0.1, 0.85],
            color_list={
                "row": ["#1f77b4", "#003f5c", "#7a5195"],
                "col": ["#1f77b4", "#003f5c"],
                "bg": "#ffffff",
            },
            line_width=1,
        )

        # Extract the underlying Plotly figure for further styling.
        fig = clustergram_component.figure  # type: ignore[attr-defined]

        modern_font = "Inter, sans-serif"

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1c1c1c",
            plot_bgcolor="#1c1c1c",
            font=dict(family=modern_font, size=14, color="#e0e0e0"),
            width=1500,
            height=600,
            margin=dict(l=80, r=40, t=40, b=80),
            coloraxis_colorbar=dict(
                title=dict(text="Score", font=dict(color="#e0e0e0")),
                thickness=15,
                lenmode="pixels",
                len=300,
                outlinewidth=0,
                tickcolor="#e0e0e0",
                tickfont=dict(color="#e0e0e0"),
            ),
        )

        # Calculate adaptive font sizes based on number of columns/rows
        n_cols = len(df.columns)
        n_rows = len(df.index)
        avail_w = 1500 - (80 + 40)  # align with new width
        avail_h = 600 - (80 + 80)  # total height minus top/bottom margins
        cell_w = avail_w / max(1, n_cols)
        cell_h = avail_h / max(1, n_rows)
        font_size_x = max(8, min(14, int(cell_w * 0.4)))
        font_size_y = max(8, min(14, int(cell_h * 0.6)))

        fig.update_xaxes(
            title_text="Enzimas",
            title_standoff=30,
            tickangle=-45,
            showgrid=False,
            tickfont=dict(family=modern_font, size=font_size_x, color="#e0e0e0"),
            automargin=True,
        )
        fig.update_yaxes(
            showgrid=False,
            tickfont=dict(family=modern_font, size=font_size_y, color="#e0e0e0"),
            automargin=True,
        )

        # Remove text labels inside heatmap cells for a cleaner look
        fig.update_traces(
            selector=dict(type="heatmap"), showscale=True, texttemplate=None
        )

        # Highlight high-score enzyme columns with a translucent rectangle
        x_labels = (
            [str(t) for t in fig.layout.xaxis.ticktext]
            if fig.layout.xaxis.ticktext
            else list(df.columns)
        )
        indices = [i for i, lab in enumerate(x_labels) if lab in high_cols]
        if indices:
            x0_rect = min(indices) - 0.5
            x1_rect = max(indices) + 0.5
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=x0_rect,
                x1=x1_rect,
                y0=-0.15,
                y1=1,
                fillcolor="rgba(255, 99, 71, 0.15)",
                line=dict(color="rgba(255, 99, 71, 0.5)", width=6, dash="dot"),
                layer="above",
            )

        fig.show()
        print("Clustergram corporativo mostrado con dash_bio.Clustergram.")

    except ModuleNotFoundError:
        # Graceful fallback: perform clustering manually and display a heatmap.
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import pdist

            # Compute linkage for rows and columns
            row_linkage = linkage(
                pdist(df.values, metric="euclidean"), method="average"
            )
            col_linkage = linkage(
                pdist(df.values.T, metric="euclidean"), method="average"
            )

            # Obtain order of leaves
            row_order = leaves_list(row_linkage)
            col_order = leaves_list(col_linkage)

            # Reorder dataframe
            df_clustered = df.iloc[row_order, :].iloc[:, col_order]

        except ModuleNotFoundError:
            # SciPy not installed; proceed without clustering
            df_clustered = df
            print(
                "SciPy no está instalado; se mostrará el heatmap sin reordenar. "
                "Instala scipy para activar la clusterización."
            )

        import plotly.express as px

        # Reapply column renaming to clustered dataframe

        fig = px.imshow(
            df_clustered.values,
            x=df_clustered.columns,
            y=df_clustered.index,
            labels=dict(x="Enzimas", y="Consultas", color="Score"),
            aspect="auto",
            color_continuous_scale=px.colors.sequential.deep[::-1],
        )

        # Highlight high-score enzyme columns on reordered dataframe
        high_cols_clustered = [col for col in df_clustered.columns if col in high_cols]
        if high_cols_clustered:
            first_idx = df_clustered.columns.get_loc(high_cols_clustered[0])
            last_idx = df_clustered.columns.get_loc(high_cols_clustered[-1])
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=first_idx - 0.5,
                x1=last_idx + 0.5,
                y0=-0.15,
                y1=1,
                fillcolor="rgba(255, 99, 71, 0.15)",
                line=dict(color="rgba(255, 99, 71, 0.5)", width=6, dash="dot"),
                layer="above",
            )

        modern_font = "Inter, sans-serif"

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1c1c1c",
            plot_bgcolor="#1c1c1c",
            width=1500,
            height=600,
            font=dict(family=modern_font, size=14, color="#e0e0e0"),
            margin=dict(l=80, r=40, t=40, b=80),
            coloraxis_colorbar=dict(
                title=dict(text="Score", font=dict(color="#e0e0e0")),
                thickness=15,
                lenmode="pixels",
                len=300,
                outlinewidth=0,
                tickcolor="#e0e0e0",
                tickfont=dict(color="#e0e0e0"),
            ),
        )

        # Adaptive font sizes for fallback figure
        # Compute available space (same margins as above)
        avail_w = 1500 - (80 + 40)
        avail_h = 600 - (80 + 80)

        n_cols_f = len(df_clustered.columns)
        n_rows_f = len(df_clustered.index)
        cell_w_f = avail_w / max(1, n_cols_f)
        cell_h_f = avail_h / max(1, n_rows_f)
        font_size_x_f = max(8, min(14, int(cell_w_f * 0.4)))
        font_size_y_f = max(8, min(14, int(cell_h_f * 0.6)))

        fig.update_xaxes(
            title_standoff=30,
            tickangle=-45,
            showgrid=False,
            tickfont=dict(family=modern_font, size=font_size_x_f, color="#e0e0e0"),
            automargin=True,
        )
        fig.update_yaxes(
            showgrid=False,
            tickfont=dict(family=modern_font, size=font_size_y_f, color="#e0e0e0"),
            automargin=True,
        )

        # Ensure no text is shown in heatmap cells
        fig.update_traces(
            selector=dict(type="heatmap"), showscale=True, texttemplate=None
        )

        fig.show()
        print(
            "dash_bio no está instalado; se mostró un heatmap con "
            "clusterización jerárquica (si fue posible)."
        )

    return Command(goto=END)


builder = StateGraph(RetrievalGraphState)
builder.add_node(
    "metadata_filtering_and_hybrid_search_node",
    metadata_filtering_and_hybrid_search_node,
)
# builder.add_node("hybrid_async_retriever", hybrid_async_retriever)
builder.add_node("retrieve_in_parallel", retrieve_in_parallel)
builder.add_node("generate_queries", generate_queries)
builder.add_node("check_completion", check_completion)
builder.add_node("print_scores_matrix", print_scores_matrix)
builder.add_node("generate_clustergram", generate_clustergram)
builder.add_edge(START, "generate_queries")


def get_memory():
    """Get a memory."""
    conn = aiosqlite.connect(":memory:")
    return AsyncSqliteSaver(conn=conn)


def get_graph():
    """Get a graph."""
    memory = get_memory()
    return builder.compile(checkpointer=memory, debug=True)


async def aget_next_state(
    compiled_graph: CompiledStateGraph, config: dict
) -> RetrievalGraphState:
    """Get the next state of the graph."""
    latest_checkpoint = await compiled_graph.aget_state(config=config)
    return latest_checkpoint.next


if __name__ == "__main__":
    USER_QUERY = "enzimas del ciclo de Krebs que usan NAD y son reversibles"
    thread_config = {"configurable": {"thread_id": str(uuid4())}}
    graph = get_graph()

    next_state = aget_next_state(graph, thread_config)
    print(f"graph state: {next_state}")
    async for chunk in graph.astream(
        {"rag_input": [HumanMessage(content=USER_QUERY)]},
        config=thread_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        pass
