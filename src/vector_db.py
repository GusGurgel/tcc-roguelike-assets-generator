from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from os.path import join
from utils import MAIN_PATH
from typing import Literal

import dotenv
import os
import pandas as pd
import math

dotenv.load_dotenv(join(MAIN_PATH, "..", ".env"))

# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embeddings= NVIDIAEmbeddings(
  model="nvidia/nv-embed-v1", 
  truncate="NONE"
)
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


StoreType = Literal["items", "environments", "entities"]

DATABASES = {
    "items": {
        "csv_path": join(MAIN_PATH, "tiles_data", "items_data.csv"),
        "db_path": join(MAIN_PATH, "chroma_items_db"),
        "collection_name": "Items_Descriptions",
    },
    "environments": {
        "csv_path": join(MAIN_PATH, "tiles_data", "environment_data.csv"),
        "db_path": join(MAIN_PATH, "chroma_environments_db"),
        "collection_name": "Environments_Descriptions",
    },
    "entities": {
        "csv_path": join(MAIN_PATH, "tiles_data", "entities_data.csv"),
        "db_path": join(MAIN_PATH, "chroma_entities_db"),
        "collection_name": "Entities_Descriptions",
    },
}


def get_full_csv() -> pd.DataFrame:
    dataframes_list = []

    for category, config in DATABASES.items():
        csv_path = config["csv_path"]

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                df["category"] = category

                dataframes_list.append(df)
            except Exception as e:
                print(f"Erro ao ler o arquivo {csv_path}: {e}")
        else:
            print(f"Aviso: Arquivo não encontrado em {csv_path}")

    if not dataframes_list:
        return pd.DataFrame()

    full_df = pd.concat(dataframes_list, ignore_index=True)

    return full_df


full_csv = get_full_csv()


def query_by_tileset_position(x: int, y: int) -> list[dict]:
    # Filtra o DataFrame onde a coluna 'x' é igual ao parametro x E a coluna 'y' é igual a y
    matches = full_csv[(full_csv["x"] == x) & (full_csv["y"] == y)]

    # Se não houver correspondência, retorna lista vazia
    if matches.empty:
        return []

    # Converte as linhas encontradas para uma lista de dicionários (records)
    # Isso facilita o uso posterior (ex: tile['base64'], tile['description'])
    return matches.to_dict(orient="records")


def get_vector_store(store_type: StoreType) -> Chroma:
    """
    Recupera o vector store baseado no tipo (items, environments, entities).
    Cria o banco se ele ainda não existir.
    """
    if store_type not in DATABASES:
        raise ValueError(
            f"Tipo de store inválido: {store_type}. Escolha entre: {list(DATABASES.keys())}"
        )

    db_config = DATABASES[store_type]

    is_vector_database_created = os.path.exists(db_config["db_path"])

    if not is_vector_database_created:
        print(f"Criando vector store para '{store_type}'...")
        create_vector_store(store_type)

    return Chroma(
        collection_name=db_config["collection_name"],
        persist_directory=db_config["db_path"],
        embedding_function=embeddings,
    )


def create_vector_store(store_type: StoreType):
    """
    Lê o CSV específico do tipo e cria o banco vetorial correspondente.
    """
    if store_type not in DATABASES:
        raise ValueError(f"Tipo de store inválido: {store_type}")

    db_config = DATABASES[store_type]

    if not os.path.exists(db_config["csv_path"]):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {db_config['csv_path']}")

    df_tiles = pd.read_csv(db_config["csv_path"])

    documents = []
    ids = []

    for i, row in df_tiles.iterrows():
        metadata = {
            "b64image": row.get("base64", ""),
            "x": row.get("x", 0),
            "y": row.get("y", 0),
            "type": store_type,  # Útil para identificar a origem depois se necessário
        }

        document = Document(
            page_content=str(row["description"]),
            id=str(i),
            metadata=metadata,
        )
        ids.append(str(i))
        documents.append(document)

    vector_store = Chroma(
        collection_name=db_config["collection_name"],
        persist_directory=db_config["db_path"],
        embedding_function=embeddings,
    )

    vector_store.add_documents(documents=documents, ids=ids)
    print(f"Vector store '{store_type}' criado com sucesso em {db_config['db_path']}")


def get_cosine_similarity(text1: str, text2: str) -> float:
    """
    Calcula a similaridade cosseno entre duas strings usando o modelo de embedding global.
    Retorna um valor entre -1 e 1 (geralmente entre 0 e 1 para textos).
    Quanto maior o valor (mais próximo de 1), maior a similaridade.
    """
    vec1 = embeddings.embed_query(text1)
    vec2 = embeddings.embed_query(text2)

    # Produto escalar
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Magnitudes (normas)
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    # Prevenção de divisão por zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


def query_vector_store(
    query: str, store_type: StoreType, documents_count: int = 4
) -> list:
    """
    Faz uma busca no vector store especificado pelo store_type.
    """
    vector_store = get_vector_store(store_type)
    tiles = []

    retriever = vector_store.as_retriever(search_kwargs={"k": documents_count})

    relevant_docs = retriever.invoke(query)

    for document in relevant_docs:
        tile = {
            "b64image": document.metadata.get("b64image"),
            "x": int(document.metadata.get("x", 0)),
            "y": int(document.metadata.get("y", 0)),
            "description": document.page_content,
        }
        tiles.append(tile)

    return tiles


if __name__ == "__main__":
    pass
    # original = ""
    # reconstruction = ""

    # with open(join(MAIN_PATH, "original.txt"), "r") as file:
    #     original = file.read()

    # with open(join(MAIN_PATH, "reconstruction.txt"), "r") as file:
    #     reconstruction = file.read()

    # print(original)
    # print("-" * 80)
    # print(reconstruction)
    # print("-" * 80)
    # print(get_cosine_similarity(original, reconstruction))
    # get_vector_store("entities")
    # get_vector_store("environments")
    # get_vector_store("entities")

    # print(query_vector_store("stone wall", "environments", 1))
    
