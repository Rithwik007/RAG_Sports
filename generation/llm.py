import time
from groq import Groq
from typing import Dict, Any
import sys
sys.path.append('.')
from config import settings
from rag_logging.logger import log_generation, setup_logger


class LLMGenerator:
    """Generate answers using Groq LLM"""
    
    # Initializes the Groq LLM client with the API key and model name from settings.
    def __init__(self):
        self.logger = setup_logger("llm_generator")
        
        # Initialize Groq
        if settings.groq_api_key:
            self.client = Groq(api_key=settings.groq_api_key)
            self.model = settings.llm_model or "llama-3.3-70b-versatile"
            self.logger.info(f"Initialized Groq model: {self.model}")
        else:
            self.logger.warning("GROQ_API_KEY not set. LLM generation will not work.")
            self.client = None
            self.model = None
    
    # Sends a single user prompt to the LLM and returns the answer, token counts, and latency.
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate an answer from the prompt"""
        
        if not self.model:
            return {
                "answer": "Error: GROQ_API_KEY not configured",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency": 0
            }
        
        start_time = time.time()
        
        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.llm_temperature
            )
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Extract token usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Log generation details
            log_generation(
                self.logger,
                self.model,
                prompt_tokens,
                completion_tokens,
                latency
            )
            
            return {
                "answer": response.choices[0].message.content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency": latency,
                "model": self.model
            }
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency": time.time() - start_time
            }
    
    # Sends both a system prompt and a user prompt to the LLM for more controlled generation.
    def generate_with_system_prompt(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Generate with system prompt"""
        
        if not self.model:
            return {
                "answer": "Error: GROQ_API_KEY not configured",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency": 0
            }
        
        start_time = time.time()
        
        try:
            # Generate response with system prompt
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.llm_temperature
            )
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Extract token usage
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Log generation details
            log_generation(
                self.logger,
                self.model,
                prompt_tokens,
                completion_tokens,
                latency
            )
            
            return {
                "answer": response.choices[0].message.content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency": latency,
                "model": self.model
            }
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency": time.time() - start_time
            }


if __name__ == "__main__":
    from retrieval.retriever import Retriever
    from prompting.prompt_builder import PromptBuilder

    question = "How do I setup the VPN on my laptop in IT?"
    print(f"\n--- Phase 9: Answer Generation Test ---")
    print(f"Question: '{question}'\n")

    retriever = Retriever()
    retrieved_chunks = retriever.retrieve(question, top_k=2)

    pb = PromptBuilder()
    prompt = pb.build_rag_prompt(question, retrieved_chunks)

    llm = LLMGenerator()
    result = llm.generate(prompt)

    print(f"Generated Answer:\n{result['answer']}\n")
    print(f"Metrics -> Model: {result.get('model')}, Prompt Tokens: {result.get('prompt_tokens')}, Completion Tokens: {result.get('completion_tokens')}, Latency: {result.get('latency'):.2f}s")

    if hasattr(retriever.qdrant, 'close'):
        retriever.qdrant.close()
