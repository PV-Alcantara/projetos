import os
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader #PyPDFLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
docs_folder = os.path.join(BASE_DIR, "bulas")
chroma_db_path = os.path.join(BASE_DIR, "chroma_db_bulas")


def criar_banco_vetorial():

    print("INICIANDO CRIAÇÃO DO BANCO VETORIAL...")

    # 1️⃣ Validar PDFs
    if not os.path.exists(docs_folder) or not any(
        fname.endswith(".pdf") for fname in os.listdir(docs_folder)
    ):
        print(f"ERRO: Pasta '{docs_folder}' não existe ou não contém PDFs.")
        return

    # 2️⃣ Loader
    loader = DirectoryLoader(
        path=docs_folder,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        loader_kwargs={"extract_images": False},
        silent_errors=True,
    )

    documents = loader.load()

    # adiciona metadata do medicamento
    for doc in documents:
        caminho = doc.metadata["source"]
        nome_arquivo = os.path.basename(caminho)
        medicamento = nome_arquivo.split("_")[0]
        doc.metadata["medicamento"] = medicamento

    print(f"Documentos carregados: {len(documents)}")

    # 3️⃣ Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
    )

    splits = text_splitter.split_documents(documents)
    print(f"Chunks criados: {len(splits)}")

    # 4️⃣ Embeddings
    load_dotenv()
    chave = os.getenv("api_key")
    print(f"API carregada? {'SIM ✅' if chave else 'NÃO ❌'}")

    embeddings = OpenAIEmbeddings(
        api_key=chave,
        model="text-embedding-3-small",
    )

    # 5️⃣ Recriar banco
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)
        print(f"Pasta '{chroma_db_path}' existente removida.")

    print(f"Criando novo ChromaDB em '{chroma_db_path}'...")

    try:
        from tqdm import tqdm
        import time

        vectorstore = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embeddings,
        )

        batch_size = 100

        for i in tqdm(range(0, len(splits), batch_size)):
            batch = splits[i : i + batch_size]
            vectorstore.add_documents(batch)
            time.sleep(0.3)  # evita rate limit

        vectorstore.persist()

        print("✅ ChromaDB criado com sucesso!")

    except Exception as e:
        print(f"ERRO CRÍTICO ao criar ChromaDB: {e}")


if __name__ == "__main__":
    criar_banco_vetorial()