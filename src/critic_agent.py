import re
from typing import List

def _sentences(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])

def _extract_citations(text: str) -> List[str]:
    return re.findall(r"\[([^\[\]]+?:[^\[\]]+?)\]", text)

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _keyword_overlap(sentence: str, evidence: str, min_hits: int = 2) -> bool:
    s = _normalize(sentence)
    e = _normalize(evidence)
    toks = [t for t in s.split() if len(t) >= 4]
    if not toks:
        return True
    hits = sum(1 for t in toks[:12] if t in e)
    return hits >= min_hits

def critic_review(answer: str, retrieved_chunks: List[dict], citations_bracketed: List[str]):
    """
    Agente crítico:
    - exige citas
    - cada oración debe tener >= 1 cita
    - citas deben pertenecer a chunks recuperados
    - soporte (heurístico) por overlap con evidencia citada

    Retorna: {"ok": bool, "revised_answer": str, "issues": list[str], "stats": {...}}
    """
    issues = []

    allowed = set()
    evidence_by_cit = {}
    for ch in retrieved_chunks:
        lid = str(ch.get("laptop_id","")).strip()
        field = str(ch.get("field","")).strip()
        cit = f"{lid}:{field}" if lid and field else ""
        if cit:
            allowed.add(cit)
            evidence_by_cit[cit] = str(ch.get("text",""))

    all_cits = _extract_citations(answer)
    if not all_cits:
        issues.append("No se encontraron citas en la respuesta (requisito: [laptop_id:campo]).")

    sents = _sentences(answer)
    supported_flags = []
    unsupported = []

    for i, s in enumerate(sents):
        cits = _extract_citations(s)
        if not cits:
            supported_flags.append(False)
            unsupported.append((i, "Oración sin citas."))
            continue

        invalid = [c for c in cits if c not in allowed]
        if invalid:
            supported_flags.append(False)
            unsupported.append((i, f"Citas no permitidas (no recuperadas): {invalid}"))
            continue

        ok = False
        for c in cits:
            ev = evidence_by_cit.get(c, "")
            if ev and _keyword_overlap(s, ev):
                ok = True
                break

        supported_flags.append(ok)
        if not ok:
            unsupported.append((i, "Oración con cita pero sin soporte claro en el texto recuperado."))

    total = len(sents)
    faithfulness = (sum(1 for x in supported_flags if x) / total) if total else 0.0

    if faithfulness < 1.0:
        issues.append("Hay oraciones sin soporte suficiente en los chunks recuperados.")

    if issues:
        kept = [sents[i] for i, ok in enumerate(supported_flags) if ok]
        if kept:
            revised = " ".join(kept).strip()
            if "[" not in revised and citations_bracketed:
                revised = revised + " " + " ".join(citations_bracketed)
            revised_answer = revised
        else:
            revised_answer = "No hay evidencia suficiente en los datos para responder."

        return {
            "ok": False,
            "revised_answer": revised_answer,
            "issues": issues + [f"{idx}: {msg}" for idx, msg in unsupported],
            "stats": {
                "faithfulness": float(faithfulness),
                "unsupported_sentences": int(len(unsupported)),
                "total_sentences": int(total),
            },
        }

    return {
        "ok": True,
        "revised_answer": answer,
        "issues": [],
        "stats": {
            "faithfulness": float(faithfulness),
            "unsupported_sentences": 0,
            "total_sentences": int(total),
        },
    }
