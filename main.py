from dotenv import load_dotenv

load_dotenv()
from typing import Set

import streamlit as st

from backend.core import run_llm

st.set_page_config(
    page_title="LangChain Helper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


# Minimal styling for a clean, dark chat look
st.markdown(
    """
<style>
    .stApp { background-color: #1E1E1E; color: #FFFFFF; }
    .stChatInput textarea { background: #2D2D2D !important; color: #FFFFFF !important; }
    .stButton > button { background-color: #4CAF50; color: #FFFFFF; }
    .stChatMessage { background-color: #2D2D2D; }
</style>
""",
    unsafe_allow_html=True,
)


st.header("LangChain🦜🔗 Helper Bot")

# Initialize session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Display previous conversation
if st.session_state["chat_answers_history"]:
    for answer_text, user_text in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_text)
        st.chat_message("assistant").write(answer_text)

prompt = st.chat_input("Enter your message…")

if prompt:
    with st.spinner("Generating response..."):
        llm_result = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

    sources = set(doc.metadata.get("source", "") for doc in llm_result["context"])
    formatted_response = f"{llm_result['answer']}\n\n{create_sources_string(sources)}"

    # Persist conversation for context and rendering
    st.session_state["user_prompt_history"].append(prompt)
    st.session_state["chat_answers_history"].append(formatted_response)
    st.session_state["chat_history"].append(("human", prompt))
    st.session_state["chat_history"].append(("ai", llm_result["answer"]))

    # Immediately render the latest exchange
    st.chat_message("user").write(prompt)
    st.chat_message("assistant").write(formatted_response)

# Footer
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")
