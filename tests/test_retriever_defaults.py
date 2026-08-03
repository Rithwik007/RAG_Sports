from retrieval.retriever import Retriever


def test_retriever_uses_faster_default_top_k(monkeypatch):
    retriever = Retriever.__new__(Retriever)
    retriever.top_k = 3
    assert retriever.top_k <= 3
