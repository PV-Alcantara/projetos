#VERSAO STREAMLIT
import os
from dotenv import load_dotenv
import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import init_embeddings
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

@st.cache_resource
def setup_rag():

    print("🚀 Inicializando RAG (executa apenas uma vez)")

    

    # ========= API KEYS =========
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY não encontrada")

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY não encontrada")

    # ========= PINECONE =========
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Indexes:", pc.list_indexes().names())

    index_name = "ems-bula"
    index = pc.Index(index_name)

    # ========= EMBEDDINGS =========
    embeddings = init_embeddings(
        model="text-embedding-3-small",
        provider="openai",
        api_key=OPENAI_API_KEY
    )

    # ========= VECTOR STORE =========
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    # ========= LLM =========
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # ========= PROMPT =========
    prompt = ChatPromptTemplate.from_template("""
Você é um assistente especializado na leitura de bulas de medicamentos.

Regras obrigatórias:
- Responder somente usando o contexto.
- Não inventar informações.
- Não fornecer aconselhamento médico.

Contexto:
{context}

Pergunta:
{input}

Resposta:
""")

    # ========= DOCUMENT CHAIN =========
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # ========= RETRIEVER =========
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,
            "fetch_k": 30
        },
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    # ========= RETRIEVAL CHAIN =========
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    print("✅ RAG pronto")

    return retrieval_chain

RAG_CHAIN = setup_rag()

def rodar_agente(pergunta: str) -> str:

    try:
        response = RAG_CHAIN.invoke({
            "input": pergunta
        })

        return response.get(
    "answer",
    "Não consegui encontrar a informação na bula.")

    except Exception as e:
        print("Erro:", e)
        return "Erro ao consultar o agente."