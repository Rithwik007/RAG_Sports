from typing import List, Dict, Any
import sys
sys.path.append('.')
from rag_logging.logger import setup_logger


class PromptBuilder:
    """Build prompts for RAG - combining question, context, and instructions"""
    
    # Initializes the prompt builder with a structured logger.
    def __init__(self):
        self.logger = setup_logger("prompt_builder")
    
    # Assembles the full RAG prompt: combines the user question, retrieved context chunks (with sources/scores), and grounding instructions.
    def build_rag_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """Build a RAG prompt with question and retrieved context"""
        
        # Build context section
        context_sections = []
        for idx, chunk in enumerate(retrieved_chunks):
            source = chunk.get("metadata", {}).get("document_name", "Unknown")
            page = chunk.get("metadata", {}).get("page_number", "Unknown")
            score = chunk.get("score", 0)
            
            context_section = f"""
Context Chunk {idx + 1} (Source: {source}, Page: {page}, Score: {score:.2f}):
{chunk.get("content", "")}
"""
            context_sections.append(context_section)
        
        context_text = "\n".join(context_sections)
        
        # Build the full prompt
        prompt = f"""You are an Enterprise Knowledge Assistant. Answer the user's question based ONLY on the provided context from the company's knowledge base.

QUESTION:
{question}

CONTEXT:
{context_text}

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context above.
- If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
- Do not use any outside knowledge or make assumptions beyond what's in the context.
- When answering, cite the source document and page number when possible.
- Be concise but thorough.
- If the context contains conflicting information, mention it.

ANSWER:
"""
        
        # Log the prompt structure
        self.logger.info(
            f"Built RAG prompt with {len(retrieved_chunks)} context chunks",
            extra={"extra_data": {
                "question": question,
                "num_context_chunks": len(retrieved_chunks),
                "context_sources": [c.get("metadata", {}).get("document_name") for c in retrieved_chunks]
            }}
        )
        
        return prompt
    
    # Returns a system-level prompt that sets the LLM's role as an enterprise knowledge assistant.
    def build_system_prompt(self) -> str:
        """Build a system prompt for the LLM"""
        return """You are a helpful Enterprise Knowledge Assistant. You answer questions based on the company's internal knowledge base. Always provide accurate, grounded answers based on the context provided."""


if __name__ == "__main__":
    from retrieval.retriever import Retriever

    retriever = Retriever()
    question = "How do I setup the VPN on my laptop in IT?"
    retrieved_chunks = retriever.retrieve(question, top_k=2)

    pb = PromptBuilder()
    rag_prompt = pb.build_rag_prompt(question, retrieved_chunks)

    print("\n--- Phase 8: Context Augmentation (Prompt Builder) Test ---")
    print(rag_prompt)

    if hasattr(retriever.qdrant, 'close'):
        retriever.qdrant.close()
