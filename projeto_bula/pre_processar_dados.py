import os
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
docs_folder = os.path.join(BASE_DIR, "bulas")
chroma_db_path = os.path.join(BASE_DIR, "chroma_db_bulas") 

def criar_banco_vetorial():
    print("INICIANDO CRIAÇÃO DO BANCO VETORIAL...")
    
    # 1. Carregar Documentos
    if not os.path.exists(docs_folder) or not any(fname.endswith('.pdf') for fname in os.listdir(docs_folder)):
        print(f"ERRO: Pasta '{docs_folder}' não existe ou não contém PDFs.")
        return

    loader = DirectoryLoader(
        path=docs_folder,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True
    )
    documents = loader.load()
    print(f"Documentos carregados: {len(documents)}")

    # 2. Dividir em chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Chunks criados: {len(splits)}")

    # 3. Gerar Embeddings (Chamada de API)
    load_dotenv()
    chave = os.getenv("api_key")
    print(f"API carregada? {'SIM ✅' if chave else 'NÃO ❌'}")

    embeddings = OpenAIEmbeddings(
    api_key=chave,
    model="text-embedding-3-small")
    
    # 4. Remover e Recriar (Garante que está atualizado)
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)
        print(f"Pasta '{chroma_db_path}' existente removida.")

    # 5. Criar e Persistir o ChromaDB
    print(f"Criando novo ChromaDB em '{chroma_db_path}'...")
    try:
        Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=chroma_db_path)
        print("SUCESSO: ChromaDB criado e persistido.")
    except Exception as e:
        print(f"ERRO CRÍTICO ao criar ChromaDB: {e}")

if __name__ == "__main__":
    criar_banco_vetorial()