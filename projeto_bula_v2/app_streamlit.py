import streamlit as st
from projeto_bula import rodar_agente

st.title("Assistente de Leitura de Bulas💊")

if "messages" not in st.session_state:
    st.session_state.messages = []

# mostra histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input estilo chat
prompt = st.chat_input("Pergunte algo sobre a bula...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando bulas..."):
            resposta = rodar_agente(prompt)

        st.markdown(resposta)

    st.session_state.messages.append(
        {"role": "assistant", "content": resposta}
    )