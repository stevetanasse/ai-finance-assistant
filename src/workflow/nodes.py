import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.rag.embedder.hybrid_search import hybrid_search

from .states import AgentState, RouteDecision

ROUTER_SYSTEM_PROMPT = (
    "You are a request router for a finance assistant. "
    "Classify the user's request into exactly one of these categories: "
    "financial_concepts_node, realtime_quotes_node, out_of_scope. "
    "Use financial_concepts_node for questions about financial concepts, "
    "definitions, or general education (e.g. 'What is a P/E ratio?'). "
    "Use realtime_quotes_node for requests about current stock prices or quotes "
    "(e.g. 'What is the current price of AAPL?'). "
    "Use out_of_scope for anything unrelated to finance. "
    "Provide your classification in the 'next' field and a brief explanation "
    "of your reasoning in the 'reasoning' field."
)


def make_router_node(llm):
    def router_node(state: AgentState) -> dict:
        call_counts = dict(state.get("call_counts", {}))
        call_counts["router_node"] = call_counts.get("router_node", 0) + 1

        user_text = state["messages"][-1].content

        structured_llm = llm.with_structured_output(RouteDecision)
        decision = structured_llm.invoke([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", user_text),
        ])

        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content=decision.next)],
        }

    return router_node


FINANCIAL_CONCEPTS_SYSTEM_PROMPT = (
    "You are a financial education assistant. Answer the user's question using ONLY "
    "the information in the provided context. Cite the source URL(s) for any claims "
    "you make. If the context does not contain enough information to answer the "
    "question, say so explicitly instead of guessing or relying on outside knowledge."
)

FINANCIAL_CONCEPTS_NOT_FOUND_MESSAGE = (
    "I couldn't find any information about that topic in the knowledge base. "
    "Try rephrasing your question or asking about a different financial concept."
)


def make_financial_concepts_node(llm, qdrant_manager, dense_embedder, sparse_embedder,
                                  collection_name: str, top_k: int = 5):
    """Create the financial concepts node, grounding answers in a Qdrant collection.

    The returned node performs hybrid (dense + sparse) retrieval against
    ``collection_name`` for the user's latest message, then asks ``llm`` to synthesize
    an answer from the retrieved chunks. If no chunks are retrieved, it responds with a
    graceful fallback instead of calling the LLM.
    """
    def financial_concepts_node(state: AgentState) -> dict:
        call_counts = dict(state.get("call_counts", {}))
        call_counts["financial_concepts_node"] = call_counts.get("financial_concepts_node", 0) + 1

        user_text = next(
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )

        chunks = hybrid_search(
            qdrant_manager, dense_embedder, sparse_embedder, collection_name, user_text, top_k=top_k
        )

        if not chunks:
            return {
                "call_counts": call_counts,
                "messages": [AIMessage(content=FINANCIAL_CONCEPTS_NOT_FOUND_MESSAGE)],
            }

        context = "\n\n".join(
            f"Source: {chunk['source_url']}\n{chunk['text']}" for chunk in chunks
        )

        response = llm.invoke([
            ("system", FINANCIAL_CONCEPTS_SYSTEM_PROMPT),
            ("human", f"Context:\n{context}\n\nQuestion: {user_text}"),
        ])

        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content=response.content)],
        }

    return financial_concepts_node


@tool
def get_stock_quotes(tickers: list[str]) -> str:
    """Get the current price for up to 3 stock ticker symbols."""
    if len(tickers) > 3:
        raise ValueError("A maximum of 3 stock tickers may be requested at once.")

    quotes = []
    for ticker in tickers:
        ticker_obj = yf.Ticker(ticker)
        price = ticker_obj.fast_info.last_price
        quotes.append(f"{ticker}: {price:.2f}")

    return "\n".join(quotes)


REALTIME_QUOTES_SYSTEM_PROMPT = (
    "You are a stock quote assistant. Extract the stock ticker symbols "
    "from the user's request and call get_stock_quotes with a list of "
    "up to 3 ticker symbols."
)


def make_realtime_quotes_node(llm):
    def realtime_quotes_node(state: AgentState) -> dict:
        call_counts = dict(state.get("call_counts", {}))
        call_counts["realtime_quotes_node"] = call_counts.get("realtime_quotes_node", 0) + 1

        user_text = next(
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )

        llm_with_tools = llm.bind_tools([get_stock_quotes])
        response = llm_with_tools.invoke([
            ("system", REALTIME_QUOTES_SYSTEM_PROMPT),
            ("human", user_text),
        ])

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            content = get_stock_quotes.invoke(tool_call["args"])
        else:
            content = response.content

        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content=content)],
        }

    return realtime_quotes_node
