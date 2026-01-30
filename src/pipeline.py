import json, time, os
from src.config import TOP_K, MAX_ANSWER_WORDS, CRITIC_LOGS_PATH
from src.generator import generate_answer
from src.critic_agent import critic_review

def _append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def run_query(query: str, retriever, top_k: int = TOP_K, max_attempts: int = 2):
    """Retrieve -> Generate -> Critic con reintento y logs JSONL."""
    t0 = time.time()
    retrieved = retriever.search(query, top_k=top_k)
    t_retr = time.time() - t0

    attempt = 0
    last = None

    while attempt < max_attempts:
        extra_rules = ""
        if attempt == 1:
            extra_rules = (
                "- Verifica que cada oración tenga cita(s) [laptop_id:campo].\n"
                "- Si una oración no puede sostenerse con evidencia explícita, elimínala.\n"
                "- Prioriza 2-4 hechos bien citados (evita listas largas)."
            )

        answer, context_text, citations_bracketed, llm_meta = generate_answer(
            query, retrieved, max_words=MAX_ANSWER_WORDS, extra_rules=extra_rules
        )

        retrieved_chunks = [ch for ch, _ in retrieved]
        review = critic_review(answer, retrieved_chunks, citations_bracketed)

        _append_jsonl(CRITIC_LOGS_PATH, {
            "ts": time.time(),
            "query": query,
            "attempt": attempt + 1,
            "top_k": top_k,
            "latency_retrieval_s": t_retr,
            "latency_llm_s": float(llm_meta.get("latency", 0.0)),
            "critic_ok": review["ok"],
            "critic_issues": review["issues"],
            "critic_stats": review.get("stats", {}),
        })

        last = (answer, context_text, citations_bracketed, llm_meta, review)
        if review["ok"]:
            break
        attempt += 1

    answer, context_text, citations_bracketed, llm_meta, review = last

    return {
        "query": query,
        "retrieved": [
            {
                "chunk_id": ch.get("chunk_id"),
                "laptop_id": ch.get("laptop_id"),
                "field": ch.get("field"),
                "text": ch.get("text"),
                "score": score,
            }
            for ch, score in retrieved
        ],
        "answer_raw": answer,
        "answer_final": review["revised_answer"],
        "critic_ok": review["ok"],
        "critic_issues": review["issues"],
        "critic_stats": review.get("stats", {}),
        "latency_retrieval_s": t_retr,
        "latency_llm_s": float(llm_meta.get("latency", 0.0)),
    }
