from typing import List
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ingestion.parser import Document
import sys
sys.path.append('.')
from config import settings
from rag_logging.logger import log_chunking, setup_logger


@dataclass
class Chunk:
    """Text chunk with metadata"""
    content: str
    metadata: dict
    chunk_id: int


class DocumentSplitter:
    """Split documents into chunks using Recursive Character Text Splitter"""
    
    # Initializes the splitter with chunk size/overlap from config and creates the LangChain text splitter.
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.logger = setup_logger("document_splitter")
        
        # Initialize the splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    # Splits a single Document's content into Chunk objects, each inheriting the document's metadata.
    def split_document(self, document: Document) -> List[Chunk]:
        """Split a single document into chunks"""
        # Split the content
        texts = self.splitter.split_text(document.content)
        
        chunks = []
        for idx, text in enumerate(texts):
            # Create chunk with inherited metadata
            chunk_metadata = document.metadata.copy()
            chunk_metadata["chunk_id"] = idx
            chunk_metadata["chunk_index"] = idx
            
            chunk = Chunk(
                content=text,
                metadata=chunk_metadata,
                chunk_id=idx
            )
            chunks.append(chunk)
        
        return chunks
    
    # Splits a list of Documents into chunks, assigns globally unique IDs, and logs the result.
    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split multiple documents into chunks"""
        all_chunks = []
        chunk_counter = 0
        
        for doc in documents:
            chunks = self.split_document(doc)
            
            # Update chunk IDs to be globally unique
            for chunk in chunks:
                chunk.chunk_id = chunk_counter
                chunk.metadata["chunk_id"] = chunk_counter
                chunk_counter += 1
            
            all_chunks.extend(chunks)
        
        # Log chunking details
        log_chunking(
            self.logger,
            self.chunk_size,
            self.chunk_overlap,
            len(all_chunks)
        )
        
        return all_chunks


if __name__ == "__main__":
    from pathlib import Path
    from ingestion.parser import DocumentParser
    from metadata.metadata_manager import MetadataManager

    data_dir = Path("data")
    if data_dir.exists():
        # Step 1: Parse documents (Phase 1)
        parser = DocumentParser()
        raw_docs = parser.parse_directory(data_dir)

        # Step 2: Enrich metadata (Phase 2)
        manager = MetadataManager()
        enriched_docs = manager.enrich_documents(raw_docs)

        # Step 3: Chunking (Phase 3)
        splitter = DocumentSplitter()
        chunks = splitter.split_documents(enriched_docs)
        print(f"\nSuccessfully generated {len(chunks)} text chunks across all documents.")
    else:
        print(f"Directory '{data_dir}' not found.")
