import os
import sys
# import asyncio # No longer needed
from pathlib import Path
from uuid import uuid4

from aeo_blog_engine.knowledge.knowledge_base import get_knowledge_base
from qdrant_client.http.models import PointStruct, models # Import Qdrant models
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

def ingest_docs(upload_dir=None): # No longer async
    """
    Reads markdown/text/pdf files from the docs/ directory (and optional upload_dir), embeds them, and loads them directly into Qdrant.
    """
    vector_db = get_knowledge_base() # This is the agno.vectordb.qdrant.Qdrant instance
    qdrant_client = vector_db.client # Get the underlying QdrantClient
    embedder = vector_db.embedder # Get the OpenAIEmbedder
    collection_name = vector_db.collection

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_docs_dir = Path(os.path.join(current_dir, "docs"))
    
    directories_to_scan = [base_docs_dir]
    if upload_dir:
        directories_to_scan.append(Path(upload_dir))
    
    points_to_upsert = []

    for docs_dir in directories_to_scan:
        if not os.path.exists(docs_dir):
            continue
            
        print(f"Scanning for documents in: {docs_dir}")

        for root, _, files in os.walk(docs_dir):
            for file_name in files:
                content = ""
                file_path = Path(root) / file_name

                if file_name.endswith(".md") or file_name.endswith(".txt"):
                    print(f"Found text file: {file_path}")
                    try:
                        content = file_path.read_text(encoding='utf-8')
                    except Exception as e:
                        print(f"Error reading text file {file_path}: {e}")
                        continue
                
                elif file_name.endswith(".pdf"):
                    print(f"Found PDF file: {file_path}")
                    if PdfReader is None:
                        print(f"Skipping PDF {file_path}: pypdf not installed.")
                        continue
                    try:
                        reader = PdfReader(str(file_path))
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                content += text + "\n"
                    except Exception as e:
                        print(f"Error reading PDF file {file_path}: {e}")
                        continue
                
                if content:
                    if not content.strip():
                        print(f"Skipping empty file: {file_path}")
                        continue

                    try:
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
                        print(f"Error embedding/processing file {file_path}: {e}")

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
