from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from .loaders import DocumentLoader
import sys
sys.path.append('.')
from rag_logging.logger import log_document_loading, setup_logger


@dataclass
class Document:
    """Unified document structure"""
    content: str
    metadata: Dict[str, Any]
    page_number: int
    source: str


class DocumentParser:
    """Parse and normalize documents from various formats"""
    
    # Initializes the parser with a structured logger and a DocumentLoader instance.
    def __init__(self):
        self.logger = setup_logger("document_parser")
        self.loader = DocumentLoader()
    
    # Loads a single file, converts each page into a Document object, and logs the result.
    def parse_file(self, file_path: Path) -> List[Document]:
        """Parse a single file into Document objects"""
        try:
            pages = self.loader.load_document(file_path)
            documents = []
            
            for page_data in pages:
                doc = Document(
                    content=page_data["content"],
                    metadata={
                        "source": page_data["source"],
                        "page_number": page_data["page_number"],
                    },
                    page_number=page_data["page_number"],
                    source=page_data["source"]
                )
                documents.append(doc)
            
            # Log loading details
            total_chars = sum(len(doc.content) for doc in documents)
            log_document_loading(
                self.logger,
                file_path.name,
                len(documents),
                total_chars,
                "Success"
            )
            
            return documents
            
        except Exception as e:
            log_document_loading(
                self.logger,
                file_path.name,
                0,
                0,
                f"Failed: {str(e)}"
            )
            raise
    
    # Recursively walks a directory, parses all supported files, and returns a combined list of Documents.
    def parse_directory(self, directory: Path) -> List[Document]:
        """Parse all supported files in a directory"""
        all_documents = []
        
        supported_extensions = {'.pdf', '.docx', '.md', '.csv', '.json', '.xlsx', '.xls'}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents = self.parse_file(file_path)
                all_documents.extend(documents)
        
        return all_documents


if __name__ == "__main__":
    parser = DocumentParser()
    data_dir = Path("data")
    if data_dir.exists():
        docs = parser.parse_directory(data_dir)
        print(f"\nSuccessfully parsed {len(docs)} documents from '{data_dir}'.")
    else:
        print(f"Directory '{data_dir}' not found.")
