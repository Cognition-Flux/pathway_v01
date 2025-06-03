"""Multi query retriever."""

# %%

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langgraph.types import Send

from agentic_workflow.llm_chains import chain_for_multi_query_retieval
from agentic_workflow.schemas import MultiQueryRetrieverState, QueryState
from agentic_workflow.vectorstore.retriever import retriever


def generate_queries(
    state: MultiQueryRetrieverState,
) -> dict[str, list[str]]:
    """Generate multiple diverse search queries from an original user query.

    Args:
        state: The current state containing the original query

    Returns:
        Dictionary with 'queries' key containing the generated query list
    """
    queries = chain_for_multi_query_retieval.invoke(
        {"question": state["retrieval_query"]}
    ).queries

    detailed_messages = [
        f'🔍 Generando consultas para: "{state["retrieval_query"]}"',
        f"✅ Se generaron {len(queries)} consultas:",
    ]

    # Agregar cada consulta como un mensaje individual
    for i, query in enumerate(queries, 1):
        detailed_messages.append(f"  {i}. {query}")

    # Mensaje simple para el chat principal
    simple_message = ["Generando consultas para responder a tu pregunta..."]

    return {
        "queries": queries,
        "messages": simple_message,
        "reasoning": "\n".join(detailed_messages),
    }


def retrieve_documents(
    state: QueryState,
) -> dict[str, list[Document]]:
    """Retrieve documents for a single query.

    Args:
        state: State containing the query to retrieve documents for

    Returns:
        Dictionary with 'documents' key containing the retrieved documents
    """
    docs = retriever.get_relevant_documents(state.query)

    # Mensajes informativos detallados para el razonamiento
    detailed_messages = [
        f'🔎 Buscando documentos para: "{state.query}"',
        f"📄 Se encontraron {len(docs)} documentos relevantes",
    ]

    # Agregar información resumida de los documentos encontrados
    if docs:
        detailed_messages.append("📑 Documentos recuperados:")
        # Mostrar solo los primeros 3 documentos
        for i, doc in enumerate(docs[:3], 1):
            # Obtener el título o las primeras palabras del contenido
            content_preview = (
                doc.page_content[:50] + "..."
                if len(doc.page_content) > 50
                else doc.page_content
            )
            detailed_messages.append(f"  {i}. {content_preview}")

        if len(docs) > 3:
            detailed_messages.append(f"  ... y {len(docs) - 3} documentos más")

    # Mensaje simple para el chat principal
    simple_message = ["Buscando documentos relevantes..."]

    return {
        "documents": docs,
        "messages": simple_message,
        "reasoning": "\n".join(detailed_messages),
    }


def retrieve_in_parallel(
    state: MultiQueryRetrieverState,
) -> list[Send]:
    """Generate parallel retrieval tasks for multiple queries.

    Returns a list of Send objects to trigger parallel execution of retrieve_documents
    for each query in the state.

    Args:
        state: The current state containing the queries to process in parallel

    Returns:
        List of Send commands for parallel execution
    """
    # No necesitamos agregar mensajes aquí ya que esta función solo configura
    # las tareas paralelas y no actualiza el estado directamente

    return [
        Send("retrieve_documents", QueryState(query=query))
        for query in state["queries"]
    ]


builder = StateGraph(MultiQueryRetrieverState)
builder.add_node("generate_queries", generate_queries)
builder.add_node("retrieve_documents", retrieve_documents)

# Update graph connections to use conditional edges for parallel execution
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,
    path_map=["retrieve_documents"],
)


# Create the MultiRetrieverGraph
MultiRetrieverGraph = builder.compile()

if __name__ == "__main__":
    result = MultiRetrieverGraph.invoke(
        {"retrieval_query": "que es el envejecimiento"},
    )

    print(f"final number of documents: {len(result['documents'])}")
