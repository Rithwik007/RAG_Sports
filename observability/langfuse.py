from typing import Dict, Any, Optional
from langfuse import Langfuse
import sys
import uuid
sys.path.append('.')
from config import settings
from rag_logging.logger import setup_logger


class LangfuseTracer:
    """Trace RAG pipeline using Langfuse for observability"""
    
    # Initializes the Langfuse client with API keys from settings; disables tracing if keys are missing.
    def __init__(self):
        self.logger = setup_logger("langfuse_tracer")
        self.session_id = str(uuid.uuid4())
        
        # Initialize Langfuse if keys are available
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                self.langfuse = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                    debug=True
                )
                self.enabled = True
                self.logger.info(f"Langfuse tracing enabled (session: {self.session_id})")
            except Exception as e:
                self.langfuse = None
                self.enabled = False
                self.logger.warning(f"Langfuse tracing disabled due to error: {e}")
        else:
            self.langfuse = None
            self.enabled = False
            self.logger.info("Langfuse tracing disabled (no API keys)")
    
    # Creates a new Langfuse trace for a user query, tagging it with session and model info.
    def create_trace(self, query: str) -> Any:
        """Create a new trace for a request"""
        if not self.enabled:
            self.logger.debug("Langfuse disabled, skipping trace creation")
            return None
        
        try:
            trace = self.langfuse.trace(
                name="rag_query",
                input={"query": query},
                session_id=self.session_id,
                metadata={"model": settings.llm_model}
            )
            self.logger.info(f"Created trace for query: {query[:50]}...")
            return trace
        except Exception as e:
            self.logger.error(f"Failed to create trace: {e}")
            return None
    
    # Logs the retrieval step (query, top_k, retrieved chunks with scores) as a span in the trace.
    def log_retrieval(
        self,
        trace: Any,
        query: str,
        retrieved_chunks: list,
        top_k: int = 5
    ):
        """Log retrieval step in the trace"""
        if not self.enabled or not trace:
            return
        
        try:
            span = trace.span(
                name="retrieval",
                input={
                    "query": query,
                    "top_k": top_k
                },
                output={
                    "num_chunks": len(retrieved_chunks),
                    "chunks": [
                        {
                            "id": c.get("id"),
                            "score": c.get("score"),
                            "document": c.get("metadata", {}).get("document_name"),
                            "content_preview": c.get("content", "")[:100]
                        }
                        for c in retrieved_chunks
                    ]
                },
                metadata={"retrieval_method": "semantic_search"}
            )
            span.end()
            self.logger.info(f"Logged retrieval span with {len(retrieved_chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Failed to log retrieval: {e}")
    
    # Logs the LLM generation step (prompt, answer, model, tokens, latency) as a span in the trace.
    def log_generation(
        self,
        trace: Any,
        prompt: str,
        answer: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float
    ):
        """Log generation step in the trace"""
        if not self.enabled or not trace:
            return
        
        try:
            generation = trace.generation(
                name="answer_generation",
                input=prompt,
                output=answer,
                model=model,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                latency=latency,
                metadata={"temperature": settings.llm_temperature}
            )
            self.logger.info(f"Logged generation span: {prompt_tokens}+{completion_tokens} tokens, {latency:.2f}s")
        except Exception as e:
            self.logger.error(f"Failed to log generation: {e}")
    
    # Finalizes the trace with the answer output and optional metadata, then flushes to Langfuse.
    def end_trace(self, trace: Any, answer: str, metadata: Dict[str, Any] = None):
        """End the trace with final output and flush"""
        if not self.enabled or not trace:
            return
        
        try:
            trace.update(
                output={
                    "answer": answer,
                    "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                    **(metadata or {})
                }
            )
            # Flush to ensure trace is sent
            self.langfuse.flush()
            self.logger.info("Trace ended and flushed")
        except Exception as e:
            self.logger.error(f"Failed to end trace: {e}")
    
    # Forces all pending Langfuse traces to be sent to the server immediately.
    def flush(self):
        """Manually flush all pending traces"""
        if self.enabled and self.langfuse:
            try:
                self.langfuse.flush()
                self.logger.info("Langfuse traces flushed")
            except Exception as e:
                self.logger.error(f"Failed to flush traces: {e}")


if __name__ == "__main__":
    print("\n--- Observability (Langfuse Tracing) Setup Test ---")
    tracer = LangfuseTracer()

    if tracer.enabled:
        query = "How do I setup VPN on my laptop?"
        trace = tracer.create_trace(query)
        tracer.log_retrieval(trace, query, [], top_k=3)
        tracer.end_trace(trace, "Test answer for observability check.")
        tracer.flush()
        print("Langfuse tracing is ENABLED and trace was sent successfully!")
    else:
        print("Langfuse tracing is DISABLED (no LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY set in .env).")
        print("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env to enable observability tracing.")
