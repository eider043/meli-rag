from typing import Set, Tuple

def precision_recall_at_k(result: dict, relevant_ids: Set[str], k: int) -> Tuple[float, float]:
    retrieved_ids = [str(r["laptop_id"]) for r in result["retrieved"][:k]]
    retrieved_set = set(retrieved_ids)
    hit = len(retrieved_set & relevant_ids)
    prec = hit / max(1, len(retrieved_set))
    rec = hit / max(1, len(relevant_ids))
    return prec, rec

def faithfulness(result: dict) -> float:
    stats = result.get("critic_stats", {}) or {}
    return float(stats.get("faithfulness", 0.0))

def answer_coverage(result: dict) -> float:
    """
    Coverage simple: % de chunks recuperados (por par laptop_id+field) citados en la respuesta final.
    """
    answer = result.get("answer_final", "")
    cited = set()
    for r in result.get("retrieved", []):
        lid = str(r.get("laptop_id","")).strip()
        field = str(r.get("field","")).strip()
        if lid and field and f"[{lid}:{field}]" in answer:
            cited.add((lid, field))

    retrieved_pairs = set()
    for r in result.get("retrieved", []):
        lid = str(r.get("laptop_id","")).strip()
        field = str(r.get("field","")).strip()
        if lid and field:
            retrieved_pairs.add((lid, field))

    return len(cited) / max(1, len(retrieved_pairs))
