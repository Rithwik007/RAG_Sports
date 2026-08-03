from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SearchRequest
from chunking.splitter import Chunk
import sys
sys.path.append('.')
from config import settings


class QdrantManager:
    """Manage vector storage using Qdrant"""
    
    # Connects to Qdrant and sets the collection name from config.
    def __init__(self, location: str = "local_qdrant"):
        # Initialize Qdrant client (local persistent mode)
        self.client = QdrantClient(path=location)
        self.collection_name = settings.qdrant_collection_name
    
    # Creates a new Qdrant collection with cosine distance if it doesn't already exist.
    def create_collection(self, vector_size: int):
        """Create a new collection"""
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name in collection_names:
            return
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
    
    # Inserts chunk text, metadata, and embedding vectors as points into the Qdrant collection.
    def store_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Store chunks with their embeddings in Qdrant"""
        points = []
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload={
                    "content": chunk.content,
                    **chunk.metadata
                }
            )
            points.append(point)
        
        # Insert points in batches
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    # Performs a cosine similarity search with optional metadata filters and returns ranked results.
    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)
        
        # Search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        ).points
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content"),
                "metadata": {k: v for k, v in result.payload.items() if k != "content"}
            })
        
        return formatted_results
    
    # Returns the collection name, point count, and vector size for the active collection.
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size
            }
        except Exception as e:
            return {}

    # Closes the Qdrant client connection to release resources.
    def close(self):
        """Explicitly close the Qdrant client connection"""
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()


if __name__ == "__main__":
    from pathlib import Path
    from ingestion.parser import DocumentParser
    from metadata.metadata_manager import MetadataManager
    from chunking.splitter import DocumentSplitter
    from embeddings.embedder import Embedder

    data_dir = Path("data")
    if data_dir.exists():
        # Phase 1-4 pipeline
        raw_docs = DocumentParser().parse_directory(data_dir)
        enriched_docs = MetadataManager().enrich_documents(raw_docs)
        chunks = DocumentSplitter().split_documents(enriched_docs)
        embedder = Embedder()
        vectors = embedder.embed_chunks(chunks)

        # Phase 5: Vector Storage (Qdrant)
        qdrant = QdrantManager()
        qdrant.create_collection(vector_size=embedder.dimensions)
        qdrant.store_chunks(chunks, vectors)

        info = qdrant.get_collection_info()
        print(f"\nPhase 5 Complete! Collection '{info.get('name')}' stored {info.get('points_count')} vectors with payload metadata.")
        qdrant.close()
    else:
        print(f"Directory '{data_dir}' not found.")
