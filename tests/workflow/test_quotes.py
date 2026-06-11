from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.workflow.nodes import get_stock_quotes, make_realtime_quotes_node


class MockQuotesNodeLLM:
    def __init__(self, tickers):
        self.tickers = tickers

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "get_stock_quotes",
                "args": {"tickers": self.tickers},
                "id": "call_1",
                "type": "tool_call",
            }],
        )


def mock_get_stock_quotes(tickers: list[str]) -> str:
    fake_prices = {"AAPL": "189.45", "MSFT": "415.20"}
    return "\n".join(f"{ticker}: {fake_prices[ticker]}" for ticker in tickers)


def test_single_ticker_returns_formatted_quote():
    with patch("src.workflow.nodes.get_stock_quotes") as mock_tool:
        mock_tool.invoke.side_effect = lambda args: mock_get_stock_quotes(args["tickers"])

        node = make_realtime_quotes_node(MockQuotesNodeLLM(tickers=["AAPL"]))
        result = node({
            "messages": [HumanMessage(content="What is the price of AAPL?")],
            "call_counts": {},
        })

    assert result["messages"][-1].content == "AAPL: 189.45"
    assert result["call_counts"]["realtime_quotes_node"] == 1


def test_multiple_tickers_returns_formatted_quotes():
    with patch("src.workflow.nodes.get_stock_quotes") as mock_tool:
        mock_tool.invoke.side_effect = lambda args: mock_get_stock_quotes(args["tickers"])

        node = make_realtime_quotes_node(MockQuotesNodeLLM(tickers=["AAPL", "MSFT"]))
        result = node({
            "messages": [HumanMessage(content="What are the prices of AAPL and MSFT?")],
            "call_counts": {},
        })

    assert result["messages"][-1].content == "AAPL: 189.45\nMSFT: 415.20"


def test_exceeding_ticker_limit_raises_error():
    with pytest.raises(ValueError, match="A maximum of 3 stock tickers may be requested at once."):
        get_stock_quotes.invoke({"tickers": ["AAPL", "MSFT", "GOOG", "TSLA"]})


def test_realtime_quotes_call_count_increments():
    with patch("src.workflow.nodes.get_stock_quotes") as mock_tool:
        mock_tool.invoke.side_effect = lambda args: mock_get_stock_quotes(args["tickers"])

        node = make_realtime_quotes_node(MockQuotesNodeLLM(tickers=["AAPL"]))
        result = node({
            "messages": [HumanMessage(content="What is the price of AAPL?")],
            "call_counts": {},
        })

    assert result["call_counts"]["realtime_quotes_node"] == 1
