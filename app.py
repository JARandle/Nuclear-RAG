import streamlit as st
from src.chain import ask
import json
import os

st.set_page_config(
    page_title="Nuclear Compliance Assistant",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Nuclear Compliance Assistant")
st.caption("Ask questions grounded in NRC Regulatory Guides and nuclear standards documents.")

# Sidebar
with st.sidebar:
    st.header("Document Library")

    data_folder = "data"
    pdf_files = [
        f for f in os.listdir(data_folder)
        if f.endswith(".pdf")
    ] if os.path.exists(data_folder) else []

    if pdf_files:
        st.success(f"{len(pdf_files)} document(s) loaded")
        for pdf in pdf_files:
            st.markdown(f"📄 {pdf}")
    else:
        st.warning("No documents found in data folder.")

    st.divider()
    st.subheader("Eval Scores")

    eval_path = "tests/eval_results.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_data = json.load(f)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Faithfulness",
                value=f"{eval_data['avg_faithfulness']:.0%}"
            )
        with col2:
            st.metric(
                label="Relevancy",
                value=f"{eval_data['avg_relevancy']:.0%}"
            )
        st.caption(f"Based on {eval_data['num_questions']} test questions")
    else:
        st.caption("No eval results found. Run tests/eval.py to generate.")
    st.divider()

    n_results = st.slider(
        "Chunks to retrieve",
        min_value=3,
        max_value=10,
        value=5,
        key="n_results_slider",
        help="How many document chunks to retrieve per query. Higher values give more context but may introduce noise."
    )

    st.divider()
    st.caption("Built with LangChain, ChromaDB, and Anthropic Claude")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.caption(
                        f"📄 {source['source']}, Page {source['page']}"
                    )

# Chat input
if prompt := st.chat_input(
    "Ask a question about nuclear compliance...",
    key="chat_input"
):
    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = ask(prompt, n_results=n_results)

                st.markdown(result["answer"])

                with st.expander("Sources"):
                    seen = set()
                    for source in result["sources"]:
                        key = f"{source['source']}_p{source['page']}"
                        if key not in seen:
                            seen.add(key)
                            st.caption(
                                f"📄 {source['source']}, Page {source['page']}"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")