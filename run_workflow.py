import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.workflow.graph import build_graph

THREAD_ID = "finance-assistant-thread-1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name to query. Defaults to the name derived from config.yaml.",
    )
    args = parser.parse_args()

    llm = ChatOpenAI(model="gpt-4o-mini")
    compiled_graph = build_graph(llm, collection_name=args.collection)

    config = {"configurable": {"thread_id": THREAD_ID}}

    while True:
        user_input = input("Enter your request (or 'quit' to exit): ")
        if user_input.strip().lower() == "quit":
            break

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "call_counts": {},
        }
        result = compiled_graph.invoke(initial_state, config=config)

        print(result["messages"][-1].content)
        print(result["call_counts"])

# Nodes
if __name__ == "__main__":
    main()
