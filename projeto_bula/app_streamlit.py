# app.py (Seu arquivo Streamlit)

import streamlit as st
# Importa a função de execução do seu módulo
from projeto_bula import rodar_agente 

st.title("Assistente de Leitura de Bulas - ACHE 💊")

user_input = st.text_input("Qual sua dúvida sobre a bula do medicamento?")

if user_input:
    # Chama a função que executa todo o seu agente configurado
    response = rodar_agente(user_input)
    st.write(response)