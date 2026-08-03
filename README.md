# Build with RAG Workshop

This project is a hands-on example of a Retrieval-Augmented Generation (RAG) system. It shows how to turn documents into searchable knowledge and use an LLM to answer questions with grounded context.

## Key Points

- The project ingests documents from a knowledge base and prepares them for retrieval.
- It enriches documents with metadata, splits them into chunks, and creates embeddings.
- These embeddings are stored in Qdrant so relevant information can be retrieved quickly.
- User questions are answered by combining retrieved context with an LLM response.
- It includes a Streamlit UI, logging, tracing, and basic evaluation support.

## Main Workflow

1. Load documents from the data folder.
2. Extract metadata and chunk the text.
3. Generate embeddings for each chunk.
4. Store the vectors in Qdrant.
5. Retrieve relevant chunks for a user query.
6. Build a prompt and generate an answer using an LLM.

## Technologies Used

- Python
- Streamlit for the UI
- Qdrant for vector search
- Sentence Transformers for embeddings
- Groq / LLM for answer generation
- Langfuse for tracing and observability
- RAGAS for evaluation

## Quick Start

```bash
pip install -r requirements.txt
python app.py --ingest
python app.py --query "What information is available in the knowledge base?"
```

You can also run the web interface:

```bash
streamlit run ui/streamlit_app.py
```

## What This Project Demonstrates

- End-to-end RAG pipeline design
- Document ingestion and chunking
- Semantic retrieval with embeddings
- Prompt-based answer generation
- Practical integration of search and LLMs in a real application

**RAGAS evaluation not running:**
- RAGAS requires `OPENAI_API_KEY` to be set (it uses OpenAI for metric computation)
- Without it, the evaluator setup is verified but metrics won't execute

**Langfuse tracing disabled:**
- Set both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`
- Tracing is optional; the pipeline works without it

---

## License

This project is for educational and workshop purposes.

## Acknowledgments

- [Groq](https://groq.com) — Ultra-fast LLM inference
- [Sentence Transformers](https://www.sbert.net) — Embedding models
- [Qdrant](https://qdrant.tech) — Vector database
- [LangChain](https://www.langchain.com) — Text splitting utilities
- [RAGAS](https://docs.ragas.io) — RAG evaluation framework
- [Langfuse](https://langfuse.com) — LLM observability
- [Streamlit](https://streamlit.io) — Web UI framework
