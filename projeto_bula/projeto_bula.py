# agente.py

# 1) Importação das bibliotecas

# Importações Streamlit (necessárias para cache e secrets)
import streamlit as st 
import os
import shutil

# Importações para manipulação dos dados (mantidas, mas a lógica pesada foi movida)
import pandas as pd
from dotenv import load_dotenv # Mantida para a execução do preprocessamento/configuração local

# Importações para criação do agente, mensagens e ferramentas estruturadas
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks import get_openai_callback 

# Importações para RAG (apenas para carregamento)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# DocumentLoaders e TextSplitter não são mais necessários neste arquivo

# 2) Configuração e Variáveis Globais (Rodam em toda execução)
# Recomenda-se que a API Key seja lida do st.secrets no Streamlit
# Se rodar localmente, o dotenv ainda pode ser usado para chaves
load_dotenv() 

docs_folder = "bulas"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
chroma_db_path = os.path.join(BASE_DIR, "chroma_db_bulas") 

print("DEBUG: Variáveis e bibliotecas configuradas.")


# 3) Função de Configuração do Agente e RAG (Cacheada)
# Esta função cria e retorna o AgentExecutor. O @st.cache_resource garante 
# que ela só seja executada uma vez por app/deploy.
@st.cache_resource
def setup_agente_e_rag():
    """
    Configura e retorna o AgentExecutor completo com VectorStore e LLM.
    Esta função é cacheada para rodar apenas uma vez.
    """
    print("\n--- INICIANDO SETUP LENTO (CACHE) ---")
    
    # Obter a chave de API (Prioriza st.secrets, fallback para .env)
    try:
        # Se estiver rodando no Streamlit, use st.secrets
        chave = st.secrets["openai_api_key"]
    except Exception:
        # Fallback para uso local/desenvolvimento
        chave = os.getenv("api_key")
        
    if not chave:
        print("ERRO: Chave 'openai_api_key' não encontrada. Verifique st.secrets ou .env.")
        raise ValueError("Chave de API OpenAI não configurada.")
        
    # I. Carregamento do VectorStore (Rápido, pois o ChromaDB já existe no disco)
    
    # 1. Gerar embeddings (Necessário para a consistência do carregamento do Chroma)
    embeddings = OpenAIEmbeddings(api_key=chave)

    # 2. Carregar o VectorStore
    try:
        # Tenta carregar o VectorStore a partir do diretório persistido.
        vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
        
        # VERIFICAÇÃO CRÍTICA: Se o banco carregar, mas estiver vazio
        if vectorstore._collection.count() == 0:
             raise Exception("O banco de dados vetorial está vazio ou corrompido.")

        print(f"DEBUG: ChromaDB carregado com sucesso! Contagem de documentos: {vectorstore._collection.count()}")
        
    except Exception as e:
        # Este bloco captura falhas no carregamento ou um erro de banco vazio.
        print(f"ERRO CRÍTICO: Não foi possível carregar o VectorStore persistido: {e}")
        st.error("Erro no RAG: O banco de dados de bulas não pôde ser carregado. Certifique-se de que a pasta 'chroma_db_bulas' foi incluída no repositório.")
        
        # Falha na inicialização do agente
        raise RuntimeError(f"Falha na configuração do VectorStore: {e}")

    # II. Criação da Ferramenta
    @tool
    def buscar_documentos(query: str) -> str:
        """
        Busca informações relevantes na base de conhecimento das bulas para responder a dúvidas.
        Utilize esta ferramenta sempre que precisar de dados factuais contidos nos documentos.
        """
        # Note: 'vectorstore' é acessível aqui por ter sido definido dentro da função cacheada
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Aumentando 'k' para 5 (Recomendação)
        docs = retriever.invoke(query)
        
        if docs:
            # Inclui o nome da fonte/documento (se disponível nos metadados)
            results = []
            for doc in docs:
                 source = doc.metadata.get("source", "Fonte Desconhecida")
                 results.append(f"Documento ({source}): {doc.page_content}")
            return "\n\n---\n\n".join(results)
        else:
            return "Não encontrei informações relevantes nos documentos para esta consulta. Por favor, tente reformular a pergunta."
    
    print("DEBUG: ferramenta 'buscar_documentos' criada com sucesso.")


    # III. Setup LLM, Memória e AgenteExecutor
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.25,
        api_key=chave,
        max_tokens=800
    )
    
    main_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5 
    )
    
    tools = [buscar_documentos]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """você é um auxiliar prestativo que faz a leitura de bulas de medicamentos quando solicitado para sanar dúvidas.
            Sua principal prioridade é a segurança e a precisão da informação.
            
            Suas regras:
            ->Responder de forma cordial ao consumidor.
            ->Sempre pesquisar informações nos documentos de bulas utilizando a ferramenta **buscar_documentos**.
            ->**NUNCA** explicar, dar conselhos ou fornecer informações que não estejam explicitamente documentadas nas bulas.
            ->Se a busca não retornar informações relevantes, informe de forma amigável que a informação não foi encontrada em sua base de conhecimento.
            ->Caso tenha dúvida ou não encontre o medicamento nas bulas informe que "não encontrada a informação no banco de dados"
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=main_memory,
    verbose=True,
    handle_parsing_errors=True
    )
    
    print("--- SETUP CONCLUÍDO. AGENTE PRONTO. ---\n")
    return agent_executor


# 4) Execução do Setup e Atribuição da Variável Global
# Esta linha é executada na primeira vez (slow) e retorna o objeto AgentExecutor pronto.
# Nas execuções subsequentes do Streamlit, ela retorna o objeto do cache (fast).
AGENTE_EXECUTOR = setup_agente_e_rag()


# 5) Função de Execução para o Streamlit (Chamada no app_streamlit.py)
def rodar_agente(user_input: str) -> str:
    """
    Recebe a pergunta do usuário e a executa no AgentExecutor configurado.
    """
    try:
        # Usa o AGENTE_EXECUTOR global que foi cacheado.
        response = AGENTE_EXECUTOR.invoke({"input": user_input})
        
        # O resultado da invocação é um dicionário, e a resposta está em 'output'
        return response.get("output", "Desculpe, não consegui gerar uma resposta.")
    except Exception as e:
        print(f"ERRO durante a execução do agente: {e}")
        return "Desculpe, ocorreu um erro na comunicação com o agente. Por favor, tente novamente."
