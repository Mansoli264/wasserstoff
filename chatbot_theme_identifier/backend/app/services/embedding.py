import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# Initialize Qdrant
client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "documents"

def init_collection():
    if COLLECTION_NAME not in client.get_collections().collections:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

def embed_text(text: str):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

def store_document_chunks(chunks: list):
    init_collection()
    points = [
        PointStruct(id=i, vector=embed_text(chunk), payload={"text": chunk})
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search_similar_chunks(query: str, top_k: int = 5):
    vector = embed_text(query)
    result = client.search(collection_name=COLLECTION_NAME, query_vector=vector, limit=top_k)
    return [hit.payload["text"] for hit in result]