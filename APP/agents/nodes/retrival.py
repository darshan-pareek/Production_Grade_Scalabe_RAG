import logfire
from APP.agents.state import AgentState
from APP.Services.Retrival.qdrant_service import search_enterprise_knowledge
from APP.Services.Retrival.ranking_service import rerank_documents

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]

    with logfire.span(" Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")

        # getting/ retriving 15 doc from the vector db for the query
        raw_results = search_enterprise_knowledge(query, limit=15)


        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        # storing the info of document
        doc_contents = [doc['content'] for doc in raw_results]

        with logfire.span("⚖️ Semantic Reranking"):
            # rarank and getting top 5 doc from those 15 documnets
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")

        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]

    return {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }