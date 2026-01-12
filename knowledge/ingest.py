import os
import sys
# import asyncio # No longer needed
from pathlib import Path
from uuid import uuid4

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from knowledge.knowledge_base import get_knowledge_base
from qdrant_client.http.models import PointStruct, models # Import Qdrant models

def ingest_docs(): # No longer async
    """
    Reads markdown/text files from the docs/ directory, embeds them, and loads them directly into Qdrant.
    """
    vector_db = get_knowledge_base() # This is the agno.vectordb.qdrant.Qdrant instance
    qdrant_client = vector_db.client # Get the underlying QdrantClient
    embedder = vector_db.embedder # Get the OpenAIEmbedder
    collection_name = vector_db.collection

    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = Path(os.path.join(current_dir, "docs"))
    
    print(f"Scanning for documents in: {docs_dir}")
    
    points_to_upsert = []

    for root, _, files in os.walk(docs_dir):
        for file_name in files:
            if file_name.endswith(".md") or file_name.endswith(".txt"):
                file_path = Path(root) / file_name
                print(f"Found: {file_path}")
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if not content.strip():
                        print(f"Skipping empty file: {file_path}")
                        continue

                    # Generate embedding for the entire file content
                    # For larger files, a proper chunking strategy would be needed.
                    # For now, we'll embed the whole file content.
                    embedding = embedder.get_embedding(content) # No await

                    points_to_upsert.append(PointStruct(
                        id=str(uuid4()), # Generate a unique ID for each point
                        vector=embedding,
                        payload={
                            "name": file_name,
                            "meta_data": {"file_path": str(file_path)}, # Required by agno
                            "content": content, # Often expected by agno
                            "content_preview": content[:200]
                        }
                    ))
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    if points_to_upsert:
        print(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{collection_name}'...")
        try:
            # Clean start: Delete collection if it exists to remove old incompatible points
            if qdrant_client.collection_exists(collection_name=collection_name):
                qdrant_client.delete_collection(collection_name=collection_name)
            
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=len(points_to_upsert[0].vector), distance=models.Distance.COSINE),
            )
            
            operation_info = qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points_to_upsert
            )
            print(f"Upsert operation info: {operation_info}")
            print("Ingestion complete.")
        except Exception as e:
            print(f"Error during Qdrant upsert: {e}")
    else:
        print("No documents found to ingest.")

if __name__ == "__main__":
    ingest_docs() # Call directly, no asyncio.run
