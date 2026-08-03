- Include the following in your repository README:

          ◦ Team Name: Asgardians
          ◦ Team No: 2
          ◦ PS ID: 5 (Cited Sports Encyclopedia)
          ◦ Drive link - https://drive.google.com/drive/folders/144vV5CR3YTAgngynga8DBpL8N-MVV7T8
          


Build with RAG Workshop
This project is a hands-on example of a Retrieval-Augmented Generation (RAG) system. It shows how to turn documents into searchable knowledge and use an LLM to answer questions with grounded context.

Key Points
The project ingests documents from a knowledge base and prepares them for retrieval.
It enriches documents with metadata, splits them into chunks, and creates embeddings.
These embeddings are stored in Qdrant so relevant information can be retrieved quickly.
User questions are answered by combining retrieved context with an LLM response.
It includes a Streamlit UI, logging, tracing, and basic evaluation support.
Main Workflow
Load documents from the data folder.
Extract metadata and chunk the text.
Generate embeddings for each chunk.
Store the vectors in Qdrant.
Retrieve relevant chunks for a user query.
Build a prompt and generate an answer using an LLM.
Technologies Used
Python
Streamlit for the UI
Qdrant for vector search
Sentence Transformers for embeddings
Groq / LLM for answer generation
Langfuse for tracing and observability
RAGAS for evaluation
Quick Start
pip install -r requirements.txt
python app.py --ingest
python app.py --query "What information is available in the knowledge base?"
You can also run the web interface:

streamlit run ui/streamlit_app.py
What This Project Demonstrates
End-to-end RAG pipeline design
Document ingestion and chunking
Semantic retrieval with embeddings
Prompt-based answer generation
Practical integration of search and LLMs in a real application
RAGAS evaluation not running:

RAGAS requires OPENAI_API_KEY to be set (it uses OpenAI for metric computation)
Without it, the evaluator setup is verified but metrics won't execute
Langfuse tracing disabled:

Set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env
Tracing is optional; the pipeline works without it
License
This project is for educational and workshop purposes.

Acknowledgments
Groq — Ultra-fast LLM inference
Sentence Transformers — Embedding models
Qdrant — Vector database
LangChain — Text splitting utilities
RAGAS — RAG evaluation framework
Langfuse — LLM observability
Streamlit — Web UI framework          