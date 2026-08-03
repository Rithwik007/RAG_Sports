from pathlib import Path
from typing import Dict, Any
from ingestion.parser import Document
import sys
sys.path.append('.')
from rag_logging.logger import log_metadata, setup_logger


class MetadataManager:
    """Enrich documents with structured metadata"""
    
    # Sets up the logger and defines mappings for department names and document types.
    def __init__(self):
        self.logger = setup_logger("metadata_manager")
        
        # Department mapping based on folder structure
        self.department_mapping = {
            "HR": "HR",
            "IT": "IT",
            "Engineering": "Engineering",
            "Company": "Company",
        }
        
        # Document type mapping based on file extension
        self.document_type_mapping = {
            ".pdf": "PDF",
            ".docx": "Document",
            ".md": "Markdown",
            ".csv": "Spreadsheet",
            ".json": "Data",
            ".xlsx": "Spreadsheet",
            ".xls": "Spreadsheet",
        }
    
    # Determines the department/category dynamically by inspecting subfolders under data/.
    def extract_department(self, file_path: Path) -> str:
        """Extract department from file path"""
        parts = file_path.parts
        for i, part in enumerate(parts):
            if part.lower() == "data" and i + 1 < len(parts) - 1:
                return parts[i + 1]
        for folder_name, department in self.department_mapping.items():
            if folder_name in parts:
                return department
        return "General"
    
    # Maps the file extension (.pdf, .docx, .md, .csv, .json) to a human-readable document type string.
    def extract_document_type(self, file_path: Path) -> str:
        """Extract document type from file extension"""
        suffix = file_path.suffix.lower()
        return self.document_type_mapping.get(suffix, "Unknown")
    
    # Returns the file name without extension as the document name.
    def extract_document_name(self, file_path: Path) -> str:
        """Extract document name from file path"""
        return file_path.stem
    
    # Performs basic language detection on the content (defaults to English).
    def detect_language(self, content: str) -> str:
        """Simple language detection (default to English)"""
        # In production, use a proper language detection library
        return "English"
    
    # Attaches department, document name, type, and language metadata to a single Document.
    def enrich_document(self, document: Document) -> Document:
        """Enrich a document with metadata"""
        file_path = Path(document.source)
        
        # Extract metadata
        metadata = {
            "department": self.extract_department(file_path),
            "document_name": self.extract_document_name(file_path),
            "document_type": self.extract_document_type(file_path),
            "language": self.detect_language(document.content),
            "total_pages": document.page_number,  # Will be updated during chunking
        }
        
        # Merge with existing metadata
        document.metadata.update(metadata)
        
        # Log metadata
        log_metadata(
            self.logger,
            document.metadata.get("document_name", "Unknown"),
            document.metadata
        )
        
        return document
    
    # Iterates over a list of Documents and enriches each one with metadata.
    def enrich_documents(self, documents: list) -> list:
        """Enrich multiple documents with metadata"""
        enriched = []
        for doc in documents:
            enriched_doc = self.enrich_document(doc)
            enriched.append(enriched_doc)
        return enriched


if __name__ == "__main__":
    from ingestion.parser import DocumentParser

    data_dir = Path("data")
    if data_dir.exists():
        # Step 1: Parse documents (Phase 1)
        parser = DocumentParser()
        raw_docs = parser.parse_directory(data_dir)

        # Step 2: Enrich with Metadata (Phase 2)
        manager = MetadataManager()
        enriched_docs = manager.enrich_documents(raw_docs)
        print(f"\nSuccessfully enriched {len(enriched_docs)} documents with metadata.")
    else:
        print(f"Directory '{data_dir}' not found.")
