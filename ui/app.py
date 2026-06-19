import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("FINANCE_API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Finance Assistant", page_icon="💹", layout="centered")
st.title("AI Finance Assistant")
st.caption("Ask me about financial concepts or get real-time stock quotes.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "route" in msg:
            st.caption(f"Route: {', '.join(msg['route'])}")

if prompt := st.chat_input("Ask a finance question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/chat",
                    json={"message": prompt, "thread_id": st.session_state.thread_id},
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                reply = data["response"]
                route = data["route"]
                st.markdown(reply)
                # st.caption(f"Route: {', '.join(route)}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply, "route": route}
                )
            except (httpx.HTTPError, httpx.ConnectError):
                st.error("Could not reach the service. Is it running on port 8000?")
