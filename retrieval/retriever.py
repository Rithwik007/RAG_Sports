from typing import List, Dict, Any, Optional
from embeddings.embedder import Embedder
from vectordb.qdrant_client import QdrantManager
from retrieval.query_understanding import QueryUnderstanding
import sys
sys.path.append('.')
from config import settings


class Retriever:
    """Retrieve relevant chunks using vector search with optional metadata filters"""

    # Sets up the embedder, Qdrant client, query understanding module, and default top_k.
    def __init__(self, qdrant=None, embedder=None):
        self.embedder = embedder or Embedder()
        self.qdrant = qdrant or QdrantManager()
        self.query_understanding = QueryUnderstanding()
        self.top_k = settings.top_k
    
    # Runs the full retrieval pipeline: understands the query, embeds it, and searches the vector DB.
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        top_k = top_k or self.top_k
        
        # Step 1: Understand the query and extract filters
        filters = {}
        if self.query_understanding.should_use_filters(query):
            filters = self.query_understanding.extract_filters(query)
        
        # Step 2: Embed the query
        query_vector = self.embedder.embed_query(query)
        
        # Step 3: Search in vector database
        results = self.qdrant.search(
            query_vector=query_vector,
            limit=top_k,
            filters=filters if filters else None
        )
        
        # Step 4: Fallback to unfiltered search if filters yielded 0 results
        if not results and filters:
            results = self.qdrant.search(
                query_vector=query_vector,
                limit=top_k,
                filters=None
            )
        
        return results
    
    # Embeds the query and searches the vector DB using explicitly provided metadata filters.
    def retrieve_with_filters(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Retrieve with explicit filters"""
        top_k = top_k or self.top_k
        
        # Embed the query
        query_vector = self.embedder.embed_query(query)
        
        # Search with filters
        results = self.qdrant.search(
            query_vector=query_vector,
            limit=top_k,
            filters=filters
        )
        
        return results


if __name__ == "__main__":
    retriever = Retriever()
    query = "How do I setup the VPN on my laptop in IT?"
    print(f"\n--- Phase 7: Retrieval Test ---")
    print(f"Query: '{query}'")
    results = retriever.retrieve(query, top_k=3)
    print(f"\nRetrieved Top {len(results)} chunks:")
    for idx, r in enumerate(results, 1):
        print(f"[{idx}] Score: {r['score']:.4f} | Source: {r['metadata'].get('source')} | Dep: {r['metadata'].get('department')}")
        print(f"    Content preview: {r['content'][:120]}...\n")

    if hasattr(retriever.qdrant, 'close'):
        retriever.qdrant.close()
