from typing import List, Dict, Any
import sys
import os
sys.path.append('.')
from rag_logging.logger import log_evaluation, setup_logger
from config import settings


class RAGASEvaluator:
    """Evaluate RAG pipeline using RAGAS metrics"""
    
    # Sets the OpenAI API key (required by RAGAS) and initializes the logger.
    def __init__(self):
        self.logger = setup_logger("ragas_evaluator")
        
        # Set OpenAI API key for RAGAS (required by the library)
        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            self.logger.info("OpenAI API key set for RAGAS evaluation")
        else:
            self.logger.warning("OPENAI_API_KEY not set. RAGAS evaluation requires it.")
    
    # Runs RAGAS evaluation on a batch of Q&A pairs, computing faithfulness, relevancy, and context metrics.
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate RAG pipeline using RAGAS metrics
        
        Args:
            questions: List of user questions
            answers: List of generated answers
            contexts: List of retrieved contexts for each question
            ground_truths: List of ground truth answers (optional for some metrics)
        """
        
        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics._faithfulness import Faithfulness
            from ragas.metrics._answer_relevance import AnswerRelevancy
            from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference, LLMContextPrecisionWithReference
            from ragas.metrics._context_recall import LLMContextRecall

            # Prepare dataset
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
            
            if ground_truths:
                data["ground_truth"] = ground_truths
            
            dataset = Dataset.from_dict(data)
            
            # Define metrics
            metrics = [
                Faithfulness(),
                AnswerRelevancy(),
                LLMContextPrecisionWithoutReference(),
            ]
            
            # Add context precision and recall if ground truths are available
            if ground_truths:
                metrics.extend([LLMContextPrecisionWithReference(), LLMContextRecall()])
            
            # Run evaluation
            result = ragas_evaluate(
                dataset=dataset,
                metrics=metrics
            )
            
            # Convert to dictionary
            scores = result.to_pandas().to_dict('records')[0]
            
            # Format scores
            formatted_scores = {}
            for key, value in scores.items():
                if hasattr(value, 'item'):  # Handle numpy types
                    formatted_scores[key] = float(value.item())
                elif isinstance(value, (int, float)):
                    formatted_scores[key] = float(value)
            
            # Log evaluation results
            log_evaluation(self.logger, formatted_scores)
            
            return formatted_scores
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return {}
    
    # Convenience wrapper to evaluate a single question-answer-context triple.
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, float]:
        """Evaluate a single Q&A pair"""
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None
        )


if __name__ == "__main__":
    evaluator = RAGASEvaluator()
    question = "How do I setup the VPN on my laptop in IT?"
    answer = "To set up the VPN, install the VPN client from the IT portal and authenticate using your credentials."
    contexts = [
        "VPN Setup Guide: Install the VPN client software downloaded from the IT portal and sign in with your employee credentials."
    ]

    print("\n--- Phase 10: RAGAS Evaluation Test ---")
    print(f"Question: '{question}'")
    print(f"Answer: '{answer}'")

    if not settings.openai_api_key:
        print("\nNote: RAGAS requires OPENAI_API_KEY for metric LLM evaluation. Structured evaluator setup is verified.")
    else:
        scores = evaluator.evaluate_single(question, answer, contexts)
        print(f"\nEvaluation Scores: {scores}")
