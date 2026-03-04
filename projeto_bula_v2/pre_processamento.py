# ========================================
# 1️⃣ Imports
# ========================================
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


# ========================================
# 2️⃣ Carregar variáveis de ambiente
# ========================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ========================================
# 3️⃣ Carregar PDFs
# ========================================
pdf_dir = "pdfs"

loader = PyPDFDirectoryLoader(pdf_dir)
documents = loader.load()

print(f"✅ {len(documents)} páginas carregadas")


# ========================================
# 4️⃣ Split em chunks
# ========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"✅ {len(chunks)} chunks criados")


# ========================================
# 5️⃣ Criar Embeddings (LangChain moderno)
# ========================================
embeddings = init_embeddings(
    model="text-embedding-3-small",
    provider="openai",
    api_key=OPENAI_API_KEY
)

texts = [chunk.page_content for chunk in chunks]

vectors = embeddings.embed_documents(texts)

print(f"✅ {len(vectors)} embeddings gerados")


# ========================================
# 6️⃣ Conectar ao Pinecone (SDK v7+)
# ========================================
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "ems-bula"

# cria índice se não existir
if index_name not in pc.list_indexes().names():
    print(f"Criando índice {index_name}...")

    pc.create_index(
        name=index_name,
        dimension=1536,  # text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

print(f"✅ Conectado ao índice '{index_name}'")


# ========================================
# 7️⃣ Preparar dados para UPSERT
# ========================================


ids = [f"{doc.metadata['source']}-{i}" for i, doc in enumerate(chunks)]

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name,
    ids=ids
)


print("🚀 Vetores enviados com sucesso para o Pinecone!")