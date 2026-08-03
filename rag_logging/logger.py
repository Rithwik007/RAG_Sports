import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    # Formats log records as structured JSON objects with timestamp, level, logger name, and extra data.
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
            
        return json.dumps(log_data, indent=2)


# Creates and configures a logger with structured JSON formatting for both console and file output.
def setup_logger(name: str = "enterprise_rag", level: int = logging.INFO) -> logging.Logger:
    """Set up structured logger for the application"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler with structured formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)
    
    return logger


# Logs a named pipeline phase along with its structured detail dictionary.
def log_phase(logger: logging.Logger, phase: str, details: Dict[str, Any]):
    """Log a phase with structured details"""
    logger.info(
        f"Phase: {phase}",
        extra={"extra_data": {"phase": phase, **details}}
    )


# Logs details of a document being loaded: filename, page count, character count, and status.
def log_document_loading(logger: logging.Logger, filename: str, pages: int, characters: int, status: str):
    """Log document loading details"""
    logger.info(
        f"Loading {filename}",
        extra={"extra_data": {
            "filename": filename,
            "pages": pages,
            "characters": characters,
            "status": status
        }}
    )


# Logs the metadata (department, type, language, etc.) attached to a document.
def log_metadata(logger: logging.Logger, document_name: str, metadata: Dict[str, Any]):
    """Log document metadata"""
    logger.info(
        f"Document: {document_name}",
        extra={"extra_data": {
            "document": document_name,
            "metadata": metadata
        }}
    )


# Logs chunking results: chunk size, overlap, and total number of generated chunks.
def log_chunking(logger: logging.Logger, chunk_size: int, overlap: int, num_chunks: int):
    """Log chunking details"""
    logger.info(
        f"Chunking complete",
        extra={"extra_data": {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "num_chunks": num_chunks
        }}
    )


# Logs embedding generation details: model name, dimensions, chunk count, and time taken.
def log_embeddings(logger: logging.Logger, model: str, dimensions: int, num_chunks: int, time_taken: float):
    """Log embedding generation details"""
    logger.info(
        f"Embeddings generated",
        extra={"extra_data": {
            "model": model,
            "dimensions": dimensions,
            "num_chunks": num_chunks,
            "time_seconds": time_taken
        }}
    )


# Logs the collection name and number of vectors stored in the vector database.
def log_vector_storage(logger: logging.Logger, collection: str, num_vectors: int):
    """Log vector storage details"""
    logger.info(
        f"Vectors stored",
        extra={"extra_data": {
            "collection": collection,
            "num_vectors": num_vectors
        }}
    )


# Logs the user query and the metadata filters extracted by the LLM.
def log_query_understanding(logger: logging.Logger, query: str, filters: Dict[str, Any]):
    """Log query understanding and extracted filters"""
    logger.info(
        f"Query understanding",
        extra={"extra_data": {
            "query": query,
            "extracted_filters": filters
        }}
    )


# Logs retrieval results: applied filters, top_k setting, and the scored chunk results.
def log_retrieval(logger: logging.Logger, filters: Dict[str, Any], top_k: int, results: list):
    """Log retrieval details"""
    logger.info(
        f"Retrieval complete",
        extra={"extra_data": {
            "applied_filters": filters,
            "top_k": top_k,
            "num_results": len(results),
            "results": [{"chunk_id": r.get("id"), "score": r.get("score")} for r in results]
        }}
    )


# Logs LLM generation metrics: model name, prompt/completion token counts, and latency.
def log_generation(logger: logging.Logger, model: str, prompt_tokens: int, completion_tokens: int, latency: float):
    """Log LLM generation details"""
    logger.info(
        f"Answer generated",
        extra={"extra_data": {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_seconds": latency
        }}
    )


# Logs RAGAS evaluation metric scores (faithfulness, relevancy, precision, recall).
def log_evaluation(logger: logging.Logger, metrics: Dict[str, float]):
    """Log evaluation metrics"""
    logger.info(
        f"Evaluation complete",
        extra={"extra_data": {
            "metrics": metrics
        }}
    )
