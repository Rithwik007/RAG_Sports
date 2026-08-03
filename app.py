"""
Main application script for Enterprise Knowledge Assistant RAG pipeline.

This script orchestrates the complete RAG pipeline:
1. Document ingestion
2. Metadata enrichment
3. Chunking
4. Embedding generation
5. Vector storage
6. Query processing
"""

from pathlib import Path
from ingestion.parser import DocumentParser
from metadata.metadata_manager import MetadataManager
from chunking.splitter import DocumentSplitter
from embeddings.embedder import Embedder
from vectordb.qdrant_client import QdrantManager
from retrieval.retriever import Retriever
from prompting.prompt_builder import PromptBuilder
from generation.llm import LLMGenerator
from evaluation.ragas_eval import RAGASEvaluator
from observability.langfuse import LangfuseTracer
from rag_logging.logger import setup_logger, log_phase
from config import settings


class RAGPipeline:
    """Complete RAG pipeline orchestration"""
    
    # Initializes all pipeline stage components (parser, metadata, splitter, LLM, evaluator, tracer).
    def __init__(self):
        self.logger = setup_logger("rag_pipeline")
        self.parser = DocumentParser()
        self.metadata_manager = MetadataManager()
        self.splitter = DocumentSplitter()
        self.embedder = None
        self.qdrant = None
        self.retriever = None
        self.prompt_builder = PromptBuilder()
        self.llm = LLMGenerator()
        self.evaluator = RAGASEvaluator()
        self.tracer = LangfuseTracer()
    
    # Runs Stages 1-5 sequentially: parse docs, enrich metadata, chunk, embed, and store in Qdrant.
    def ingest_documents(self, data_dir: str = None):
        """Phase 1-5: Ingest documents and store in	vector database"""
        data_dir = data_dir or settings.data_dir
        data_path = Path(data_dir)
        
        self.logger.info(f"Starting document ingestion from {data_dir}")
        
        # Phase 1: Document Ingestion
        log_phase(self.logger, "Document Ingestion", {"directory": str(data_dir)})
        documents = self.parser.parse_directory(data_path)
        self.logger.info(f"Parsed {len(documents)} documents")
        
        # Phase 2: Metadata Enrichment
        log_phase(self.logger, "Metadata Enrichment", {"documents": len(documents)})
        enriched_documents = self.metadata_manager.enrich_documents(documents)
        
        # Phase 3: Chunking
        log_phase(self.logger, "Chunking", {"documents": len(enriched_documents)})
        chunks = self.splitter.split_documents(enriched_documents)
        self.logger.info(f"Created {len(chunks)} chunks")
        
        # Phase 4: Embedding Generation
        log_phase(self.logger, "Embedding Generation", {"chunks": len(chunks)})
        self.embedder = Embedder()
        embeddings = self.embedder.embed_chunks(chunks)
        
        # Phase 5: Vector Storage
        log_phase(self.logger, "Vector Storage", {"vectors": len(embeddings)})
        self.qdrant = QdrantManager()
        self.qdrant.create_collection(self.embedder.dimensions)
        self.qdrant.store_chunks(chunks, embeddings)
        
        # Initialize retriever with the populated Qdrant instance
        self.retriever = Retriever(qdrant=self.qdrant, embedder=self.embedder)
        
        self.logger.info("Document ingestion complete")
        return len(chunks)
    
    # Runs Stages 6-9: understands the query, retrieves chunks, builds prompt, generates an answer, and traces everything.
    def query(self, question: str, top_k: int = None) -> dict:
        """Process a user query through the RAG pipeline"""
        top_k = top_k or settings.top_k
        
        self.logger.info(f"Processing query: {question}")
        
        # Create trace
        trace = self.tracer.create_trace(question)
        
        # Phase 6: Query Understanding (handled in retriever)
        # Phase 7: Retrieval
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        
        # Log retrieval to trace
        self.tracer.log_retrieval(trace, question, retrieved_chunks)
        
        # Phase 8: Context Augmentation
        prompt = self.prompt_builder.build_rag_prompt(question, retrieved_chunks)
        
        # Phase 9: Answer Generation
        result = self.llm.generate(prompt)
        
        # Log generation to trace
        self.tracer.log_generation(
            trace,
            prompt,
            result['answer'],
            settings.llm_model,
            result['prompt_tokens'],
            result['completion_tokens'],
            result['latency']
        )
        
        # End trace
        self.tracer.end_trace(trace, result['answer'], {
            "num_chunks_retrieved": len(retrieved_chunks),
            "latency": result['latency']
        })
        
        return {
            "answer": result['answer'],
            "retrieved_chunks": retrieved_chunks,
            "latency": result['latency'],
            "prompt_tokens": result['prompt_tokens'],
            "completion_tokens": result['completion_tokens']
        }
    
    # Runs Stage 10: evaluates RAG output quality using RAGAS metrics against provided test data.
    def evaluate(self, questions: list, answers: list, contexts: list, ground_truths: list = None) -> dict:
        """Phase 10: Evaluate RAG pipeline using RAGAS"""
        log_phase(self.logger, "Evaluation", {"questions": len(questions)})
        metrics = self.evaluator.evaluate(questions, answers, contexts, ground_truths)
        return metrics


# CLI entry point: parses --ingest and --query arguments to run the pipeline from the command line.
def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Enterprise Knowledge Assistant RAG Pipeline")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into vector database")
    parser.add_argument("--query", type=str, help="Ask a question")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")

    args = parser.parse_args()

    if not args.ingest and not args.query:
        print("Please specify --ingest, --query, or both.")
        print("\nExamples:")
        print("  python app.py --ingest")
        print("  python app.py --ingest --query 'How many annual leave days do employees get?'")
        print("  python app.py --query 'How many annual leave days do employees get?'")
        return

    pipeline = RAGPipeline()

    # --- Ingestion stage ---
    if args.ingest:
        data_path = Path(args.data_dir)
        if not data_path.exists():
            print(f"\nData directory '{args.data_dir}' not found!")
            print("    Please provide a valid directory with --data-dir.")
            return

        if not any(data_path.iterdir()):
            print(f"\nData directory '{args.data_dir}' is empty!")
            print("    Add documents to the directory before ingesting.")
            return

        print("Ingesting documents...")
        try:
            num_chunks = pipeline.ingest_documents(args.data_dir)
            print(f"Ingestion complete! Stored {num_chunks} chunks.\n")
        except Exception as e:
            print(f"\nIngestion failed: {e}")
            print("   Fix the issue above and try again.")
            return

    # --- Query stage ---
    if args.query:
        if pipeline.retriever is None:
            pipeline.retriever = Retriever()

        print(f"Query: {args.query}")
        try:
            result = pipeline.query(args.query, top_k=args.top_k)
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nRetrieved {len(result['retrieved_chunks'])} chunks")
            print(f"Latency: {result['latency']:.2f}s")
        except Exception as e:
            if "not found" in str(e).lower() and "collection" in str(e).lower():
                print(f"\nError: The knowledge base is empty.")
                print("Please run the ingest command first:")
                print("  python app.py --ingest")
            else:
                print(f"\nQuery failed: {e}")


if __name__ == "__main__":
    main()

