import logging

import yfinance as yf
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.rag.embedder.hybrid_retriever import HybridQdrantRetriever

from .states import AgentState, RouteDecision

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = (
    "You are a request router for a finance assistant. "
    "Classify the user's request into one or more of these categories: "
    "financial_concepts, realtime_quotes, out_of_scope. "
    "Use financial_concepts for questions about financial concepts, "
    "definitions, or general education (e.g. 'What is a P/E ratio?'). "
    "Use realtime_quotes for requests about current stock prices or quotes "
    "(e.g. 'What is the current price of AAPL?'). "
    "If a request asks about BOTH a financial concept AND a real-time quote "
    "(e.g. 'What is a P/E ratio, and what is AAPL's current price?'), "
    "return both 'financial_concepts' and 'realtime_quotes' in the 'next' list. "
    "Use out_of_scope for anything unrelated to finance. "
    "A request can be compound, mixing an in-scope part with an out-of-scope "
    "part - in that case, include 'out_of_scope' ALONGSIDE 'financial_concepts' "
    "and/or 'realtime_quotes' in the 'next' list rather than picking only one "
    "(e.g. for 'What is the price of TSLA and is Godzilla a real animal?', "
    "return next=['realtime_quotes', 'out_of_scope'] with realtime_quotes_query "
    "set to 'TSLA'). "
    "When 'financial_concepts' is included in 'next', set "
    "'financial_concepts_query' to a restatement of ONLY the concept-related "
    "portion of the request, with any ticker symbols or price/quote requests "
    "removed entirely (e.g. for 'What is a dividend and what is TSLA's price?', "
    "use 'What is a dividend?'). "
    "When 'realtime_quotes' is included in 'next', set 'realtime_quotes_query' "
    "to just the relevant ticker symbol(s) (e.g. 'TSLA' or 'AAPL, MSFT'), "
    "omitting all other content. "
    "Leave 'financial_concepts_query' and 'realtime_quotes_query' as null when "
    "the corresponding route is not included in 'next', and leave both null "
    "when 'next' is ['out_of_scope']. "
    "Provide your classification as a list in the 'next' field and a brief "
    "explanation of your reasoning in the 'reasoning' field."
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

        # The routing decision now drives Send-based fan-out via the `route`
        # field instead of being smuggled into `messages` as an AIMessage,
        # so the synchronizer's message inspection only sees user-facing
        # conversation content. `route_decision` carries the full decision
        # (including per-node sub-queries) so financial_concepts_node and
        # realtime_quotes_node can each work from a decomposed, relevant
        # slice of the request instead of the raw user message.
        return {
            "call_counts": call_counts,
            "route": decision.next,
            "route_decision": decision.model_dump(),
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
                                  collection_name: str, top_k: int = 5, retriever=None):
    """Create the financial concepts node, grounding answers in a Qdrant collection.

    The returned node performs hybrid (dense + sparse) retrieval against
    ``collection_name`` using the router's ``financial_concepts_query`` sub-query
    (falling back to the user's latest message, with a logged warning, if the
    sub-query is missing or empty), then asks ``llm`` to synthesize an answer from
    the retrieved chunks. If no chunks are retrieved, it responds with a graceful
    fallback instead of calling the LLM.

    ``retriever`` lets callers (e.g. ``build_graph``) supply a pre-built
    ``HybridQdrantRetriever`` so retrieval appears as its own LangSmith span; if
    omitted, one is built from the other arguments.
    """
    if retriever is None:
        retriever = HybridQdrantRetriever(
            qdrant_manager=qdrant_manager,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            collection_name=collection_name,
            top_k=top_k,
        )

    def financial_concepts_node(state: AgentState) -> dict:
        call_counts = dict(state.get("call_counts", {}))
        call_counts["financial_concepts_node"] = call_counts.get("financial_concepts_node", 0) + 1

        route_decision = state.get("route_decision")
        sub_query = route_decision.get("financial_concepts_query") if route_decision else None

        if sub_query:
            user_text = sub_query
        else:
            user_text = next(
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            )
            logger.warning(
                "financial_concepts_node: route_decision.financial_concepts_query "
                "missing or empty; falling back to raw user message"
            )

        retriever_config = {"run_name": "financial_concepts_retriever"}
        docs = retriever.invoke(user_text, retriever_config)

        # `name="financial_concepts"` lets the synchronizer identify which
        # message came from this node during fan-in, since message order
        # after parallel Send execution is not guaranteed.
        if not docs:
            return {
                "call_counts": call_counts,
                "messages": [AIMessage(content=FINANCIAL_CONCEPTS_NOT_FOUND_MESSAGE, name="financial_concepts")],
            }

        context = "\n\n".join(
            f"Source: {doc.metadata['source_url']}\n{doc.page_content}" for doc in docs
        )

        response = llm.invoke([
            ("system", FINANCIAL_CONCEPTS_SYSTEM_PROMPT),
            ("human", f"Context:\n{context}\n\nQuestion: {user_text}"),
        ])

        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content=response.content, name="financial_concepts")],
        }

    return financial_concepts_node


@tool
def get_stock_quotes(tickers: list[str]) -> str:
    """Get the current price for up to 3 stock ticker symbols."""
    if len(tickers) > 3:
        raise ValueError("A maximum of 3 stock tickers may be requested at once.")

    quotes = []
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            price = ticker_obj.fast_info.last_price
            quotes.append(f"{ticker}: {price:.2f}")
        except Exception as e:
            quotes.append(f"Quote not found for {ticker}")

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

        route_decision = state.get("route_decision")
        sub_query = route_decision.get("realtime_quotes_query") if route_decision else None

        if sub_query:
            user_text = sub_query
        else:
            user_text = next(
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            )
            logger.warning(
                "realtime_quotes_node: route_decision.realtime_quotes_query "
                "missing or empty; falling back to raw user message"
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

        # `name="realtime_quotes"` lets the synchronizer identify which
        # message came from this node during fan-in, since message order
        # after parallel Send execution is not guaranteed.
        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content=content, name="realtime_quotes")],
        }

    return realtime_quotes_node


OUT_OF_SCOPE_MESSAGE = (
    "This agent does not have the ability to reliably answer or respond to "
    "part of your request."
)


def make_out_of_scope_node():
    """Create the out_of_scope node, a fixed canned response with no LLM call.

    Runs whenever 'out_of_scope' is present in route_decision.next, whether
    alone or alongside financial_concepts/realtime_quotes, via the same
    Send-based fan-out as the other routes.
    """
    def out_of_scope_node(state: AgentState) -> dict:
        call_counts = dict(state.get("call_counts", {}))
        call_counts["out_of_scope_node"] = call_counts.get("out_of_scope_node", 0) + 1

        # `name="out_of_scope"` lets the synchronizer identify which message
        # came from this node during fan-in, since message order after
        # parallel Send execution is not guaranteed.
        return {
            "call_counts": call_counts,
            "messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE, name="out_of_scope")],
        }

    return out_of_scope_node


SYNCHRONIZER_OUT_OF_SCOPE_MESSAGE = (
    "This agent does not have the ability to reliably answer or respond to the request."
)


def make_synchronizer_node():
    """Create the synchronizer node, the terminal fan-in point for all routes.

    - route == ["out_of_scope"]: emits the fixed canned response, no LLM call.
    - len(route) > 1: combines the AIMessage from each branch (identified via
      `.name`) into one labeled, multi-section response, in fixed order
      ("**Out of Scope**" / "**Financial Concept**" / "**Market Data**"),
      including only the sections for routes actually present in `route`.
    - single non-out-of-scope route: pass-through - returns no `messages`
      update, leaving the existing assistant message as the final answer.
    """
    def synchronizer_node(state: AgentState) -> dict:
        call_counts = dict(state.get("call_counts", {}))
        call_counts["synchronizer_node"] = call_counts.get("synchronizer_node", 0) + 1
        route = state.get("route", [])

        if route == ["out_of_scope"]:
            return {
                "call_counts": call_counts,
                "messages": [AIMessage(content=SYNCHRONIZER_OUT_OF_SCOPE_MESSAGE)],
            }

        if len(route) > 1:
            sections = []
            for r, header in [
                ("out_of_scope", "**Out of Scope**"),
                ("financial_concepts", "**Financial Concept**"),
                ("realtime_quotes", "**Market Data**"),
            ]:
                if r in route:
                    msg = next(
                        m for m in reversed(state["messages"])
                        if getattr(m, "name", None) == r
                    )
                    sections.append(f"{header}\n{msg.content}")
            return {
                "call_counts": call_counts,
                "messages": [AIMessage(content="\n\n".join(sections))],
            }

        return {"call_counts": call_counts}

    return synchronizer_node
