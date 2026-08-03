from typing import Dict, Any, Optional
import sys
sys.path.append('.')
from config import settings
from rag_logging.logger import log_query_understanding, setup_logger


class QueryUnderstanding:
    """Understand user queries and extract metadata filters using LLM"""
    
    # Initializes the LLM client (Groq or Gemini) for extracting metadata filters from queries.
    def __init__(self):
        self.logger = setup_logger("query_understanding")
        self.model = None
        
        # Try to initialize with available LLM
        try:
            if settings.groq_api_key:
                from groq import Groq
                self.client = Groq(api_key=settings.groq_api_key)
                self.model = settings.llm_model
                self.logger.info("Query understanding using Groq")
            elif settings.gemini_api_key:
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                self.client = genai.GenerativeModel(settings.llm_model)
                self.model = settings.llm_model
                self.logger.info("Query understanding using Gemini")
            else:
                self.logger.warning("No LLM API key set. Query understanding disabled.")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM for query understanding: {e}")
    
    def _get_available_departments(self) -> str:
        from pathlib import Path
        data_dir = Path(settings.data_dir)
        if data_dir.exists():
            subfolders = [d.name for d in data_dir.iterdir() if d.is_dir()]
            if subfolders:
                return ", ".join(subfolders)
        return "HR, IT, Engineering, Company, Sports"

    # Sends the query to an LLM to extract department/document_type metadata filters as JSON.
    def extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract metadata filters from the query using LLM"""
        
        if not self.model:
            return {}
        
        dept_list = self._get_available_departments()
        
        # Define the prompt for filter extraction
        prompt = f"""
You are a metadata filter extractor for an enterprise knowledge base. 
Analyze the user's query and extract relevant metadata filters.

Available metadata fields:
- department: {dept_list}
- document_type: PDF, Document, Markdown, Spreadsheet, Data

User Query: "{query}"

Extract filters ONLY if the query explicitly mentions a specific department/category or document type.
If the query is general or broad, return empty filters.

Return your answer as a JSON object with this format:
{{
    "department": "Sports" or null,
    "document_type": "Spreadsheet" or null
}}

If a field is not mentioned, set it to null.
"""
        
        try:
            # Use Groq or Gemini based on what's available
            if settings.groq_api_key:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                response_text = response.choices[0].message.content
            else:
                response = self.client.generate_content(prompt)
                response_text = response.text.strip()
            
            # Robust JSON extraction using regex
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            filters = json.loads(response_text)
            
            # Remove null values
            filters = {k: v for k, v in filters.items() if v is not None}
            
            # Log the extraction
            log_query_understanding(self.logger, query, filters)
            
            return filters
            
        except Exception as e:
            self.logger.error(f"Error extracting filters: {e}")
            return {}
    
    # Quick heuristic check: returns True if the query mentions any department-related keywords.
    def should_use_filters(self, query: str) -> bool:
        """Determine if filters should be applied based on query"""
        # Simple heuristic: if query mentions specific department keywords
        department_keywords = ["HR", "IT", "Engineering", "Company", "human resources", "technology"]
        query_lower = query.lower()
        
        for keyword in department_keywords:
            if keyword.lower() in query_lower:
                return True
        
        return False


if __name__ == "__main__":
    qu = QueryUnderstanding()
    sample_queries = [
        "What is the HR leave policy for annual leaves?",
        "How do I set up the printer in IT?",
        "Tell me about company holidays."
    ]

    print("\n--- Phase 6: Query Understanding Test ---")
    for q in sample_queries:
        filters = qu.extract_filters(q)
        print(f"Query: '{q}' -> Extracted Filters: {filters}")
