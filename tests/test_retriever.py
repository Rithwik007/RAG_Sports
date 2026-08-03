from retrieval.retriever import Retriever


def test_list_queries_use_wider_retrieval_window():
    retriever = Retriever.__new__(Retriever)
    retriever.top_k = 3
    retriever.query_understanding = type("Q", (), {"should_use_filters": lambda self, q: False})()
    retriever.embedder = type("E", (), {"embed_query": lambda self, q: [0.1, 0.2]})()
    retriever.qdrant = type(
        "Qdrant",
        (),
        {"search": lambda self, query_vector, limit, filters=None: [{"id": i, "score": 1.0 - i * 0.01, "content": f"chunk {i}", "metadata": {"document_name": f"doc{i}"}} for i in range(limit)]}
    )()

    results = retriever.retrieve("List all Indian players", top_k=3)

    assert len(results) >= 6
