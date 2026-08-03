from typing import List
from sentence_transformers import SentenceTransformer
import time
from chunking.splitter import Chunk
import sys
sys.path.append('.')
from config import settings
from rag_logging.logger import log_embeddings, setup_logger


class Embedder:
    """Generate embeddings for text chunks using all-MiniLM-L6-v2"""
    
    # Loads the sentence transformer embedding model and records the vector dimensionality.
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.logger = setup_logger("embedder")
        
        # Load the model (let SentenceTransformer handle device automatically)
        self.logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimensions = self.model.get_sentence_embedding_dimension()
        
        self.logger.info(f"Model loaded. Dimensions: {self.dimensions}")
    
    # Batch-encodes all chunk texts into vector embeddings and logs timing/count details.
    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings for a list of chunks"""
        start_time = time.time()
        
        # Extract text from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        time_taken = time.time() - start_time
        
        # Log embedding details
        log_embeddings(
            self.logger,
            self.model_name,
            self.dimensions,
            len(chunks),
            time_taken
        )
        
        return embeddings.tolist()
    
    # Encodes a single query string into a vector embedding for similarity search.
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()


if __name__ == "__main__":
    from pathlib import Path
    from ingestion.parser import DocumentParser
    from metadata.metadata_manager import MetadataManager
    from chunking.splitter import DocumentSplitter

    data_dir = Path("data")
    if data_dir.exists():
        # Phase 1: Parse
        raw_docs = DocumentParser().parse_directory(data_dir)
        # Phase 2: Enrich Metadata
        enriched_docs = MetadataManager().enrich_documents(raw_docs)
        # Phase 3: Chunking
        chunks = DocumentSplitter().split_documents(enriched_docs)

        # Phase 4: Embedding Generation
        embedder = Embedder()
        vectors = embedder.embed_chunks(chunks)
        print(f"\nSuccessfully generated {len(vectors)} vector embeddings (dimensions: {embedder.dimensions}).")
    else:
        print(f"Directory '{data_dir}' not found.")
