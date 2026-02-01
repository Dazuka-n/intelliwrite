
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(override=True)

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

print(f"Testing connection to: {url}")
try:
    client = QdrantClient(url=url, api_key=api_key)
    cols = client.get_collections()
    print("[SUCCESS] Connected to Qdrant!")
    print(f"Collections: {cols}")
except Exception as e:
    print(f"[FAILURE] Could not connect: {e}")
