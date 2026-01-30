import re
from src.config import CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS

def approx_token_count(text: str) -> int:
    """Aproximación simple: tokens ~ palabras."""
    return max(1, len(str(text).split()))

def _clean_text(x: str) -> str:
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _split_to_token_windows(prefix: str, value: str, min_tokens: int, max_tokens: int):
    """Divide value en ventanas de tamaño aproximado [min_tokens, max_tokens]."""
    value = _clean_text(value)
    if not value:
        return []

    words = value.split()
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + max_tokens)
        candidate = " ".join(words[i:j]).strip()
        txt = f"{prefix}: {candidate}".strip()

        # si queda muy corto, amplía un poco
        while approx_token_count(txt) < min_tokens and j < len(words):
            j2 = min(len(words), j + 10)
            candidate = " ".join(words[i:j2]).strip()
            txt = f"{prefix}: {candidate}".strip()
            j = j2

        chunks.append(txt)
        i = j

    return chunks

def make_chunks(row: dict, laptop_id_field: str = "laptop_id"):
    """
    Retorna lista de chunks dict:
    {
      "chunk_id": str,
      "laptop_id": str,
      "field": str,
      "text": str,
      "citations": ["laptop_id:field"]
    }

    Cumple: chunks cortos (50-120 tokens aprox).
    """
    laptop_id = str(row.get(laptop_id_field, row.get("id", ""))).strip()
    chunks = []
    idx = 0

    for field, value in row.items():
        if field == laptop_id_field:
            continue
        value = _clean_text(value)
        if not value:
            continue

        texts = _split_to_token_windows(
            prefix=str(field).strip(),
            value=value,
            min_tokens=CHUNK_MIN_TOKENS,
            max_tokens=CHUNK_MAX_TOKENS,
        )

        if not texts:
            texts = [f"{field}: {value}".strip()]

        for t in texts:
            f = str(field).strip()
            chunks.append({
                "chunk_id": f"{laptop_id}_{idx}",
                "laptop_id": laptop_id,
                "field": f,
                "text": t,
                "citations": [f"{laptop_id}:{f}"],
            })
            idx += 1

    return chunks
