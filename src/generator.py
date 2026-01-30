from src.llm_openai import llm_answer

def _build_context_text(retrieved):
    lines = []
    for ch, _score in retrieved:
        lid = ch.get("laptop_id", "")
        field = ch.get("field", "")
        lines.append(f"[{lid}:{field}] {ch.get('text','')}")
    return "\n".join(lines)

def generate_answer(query: str, retrieved, max_words: int = 120, extra_rules: str = ""):
    """
    Generación (Retrieve -> Generate):
    - Usa OBLIGATORIAMENTE src/llm_openai.py para generar la respuesta.
    - Devuelve respuesta <= max_words con citas [laptop_id:campo].

    Retorna:
      (answer, context_text, citations_bracketed, llm_meta)
    """
    retrieved_chunks = [ch for ch, _ in retrieved]
    context_text = _build_context_text(retrieved)

    llm_out = llm_answer(
        question=query,
        retrieved_chunks=retrieved_chunks,
        extra_rules=extra_rules,
    )

    answer = llm_out["answer"]
    citations_bracketed = [f"[{c}]" for c in llm_out.get("citations", [])]

    words = answer.split()
    if len(words) > max_words:
        answer = " ".join(words[:max_words])

    return answer, context_text, citations_bracketed, llm_out
