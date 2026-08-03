from prompting.prompt_builder import PromptBuilder


def test_prompt_builder_trims_context_for_long_inputs():
    builder = PromptBuilder()
    long_chunk = {
        "content": "A" * 7000,
        "metadata": {"document_name": "doc1", "page_number": 1},
        "score": 0.95,
    }

    prompt = builder.build_rag_prompt("What is this?", [long_chunk])

    assert "A" * 1200 in prompt
    assert "A" * 1300 not in prompt


def test_prompt_builder_keeps_more_context_for_list_queries():
    builder = PromptBuilder()
    chunks = [
        {
            "content": f"Chunk {idx}",
            "metadata": {"document_name": "doc1", "page_number": 1},
            "score": 0.95,
        }
        for idx in range(4)
    ]

    prompt = builder.build_rag_prompt("List all Australian cricketers.", chunks)

    assert "Context Chunk 4" in prompt
