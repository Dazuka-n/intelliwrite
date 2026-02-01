import sys
import os
import asyncio

# Add project root parent to path to allow 'aeo_blog_engine' imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from knowledge.knowledge_base import get_knowledge_base

def verify():
    print("Initializing Knowledge Base connection...")
    vector_db = get_knowledge_base()
    
    query = "What is Answer Engine Optimization?"
    print(f"\nTesting retrieval for query: '{query}'")
    
    # search() returns a list of results
    try:
        results = vector_db.search(query, limit=3)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, res in enumerate(results):
                # Handle different result formats depending on agno version
                content = getattr(res, 'content', str(res))
                print(f"\n--- Result {i+1} ---")
                print(content[:500] + "..." if len(content) > 500 else content)
        else:
            print("\nNo results found. Ingestion might have failed.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError during search: {e}")

if __name__ == "__main__":
    verify()
