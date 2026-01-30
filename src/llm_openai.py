import os, json, time, re
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en .env")

client = OpenAI(api_key=API_KEY)

def _safe_json_extract(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def _trim_to_words(text: str, max_words: int = 120) -> str:
    words = str(text).split()
    return " ".join(words[:max_words])

def _allowed_citations(retrieved_chunks: list[dict]) -> set[str]:
    out = set()
    for r in retrieved_chunks:
        lid = str(r.get("laptop_id", "")).strip()
        field = str(r.get("field", "")).strip()
        if lid and field:
            out.add(f"{lid}:{field}")
    return out

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def llm_answer(question: str, retrieved_chunks: list[dict], max_chars_ctx: int = 6000, extra_rules: str = ""):
    """
    Retorna:
      {"answer": str, "citations": ["laptop_id:campo", ...], "latency": float}
    """
    ctx_lines = []
    total = 0
    for r in retrieved_chunks:
        lid = str(r.get("laptop_id", "")).strip()
        field = str(r.get("field", "")).strip()
        txt = str(r.get("text", "")).strip()

        tag = f"{lid}:{field}" if (lid and field) else lid
        line = f"[{tag}] {txt}"

        total += len(line)
        if total > max_chars_ctx:
            break
        ctx_lines.append(line)

    context = "\n".join(ctx_lines)

    rules = """Reglas obligatorias:
- Responde en español.
- Si la evidencia no es suficiente, di exactamente:
  "No hay evidencia suficiente en los datos para responder."
- Respuesta máxima: 120 palabras.
- NO inventes especificaciones ni supongas datos no presentes.
- CADA oración debe tener al menos una cita.
- Usa citas en el formato exacto: [laptop_id:campo]
- Devuelve SOLO JSON válido con esta estructura EXACTA:
{
  "answer": "texto de la respuesta con citas [laptop_id:campo]",
  "citations": ["laptop_id:campo", "laptop_id:campo"]
}
""".strip()

    if extra_rules:
        rules += "\n\nReglas adicionales:\n" + extra_rules.strip()

    prompt = f"""Eres un asistente experto en laptops. Responde SOLO usando la evidencia dada.

{rules}

Pregunta:
{question}

Evidencia:
{context}

JSON:
""".strip()

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    latency = time.time() - t0

    raw = resp.choices[0].message.content.strip()
    parsed = _safe_json_extract(raw)

    answer = parsed.get("answer", raw)
    cits = parsed.get("citations", [])

    if not isinstance(cits, list):
        cits = []

    allowed = _allowed_citations(retrieved_chunks)
    cits = [str(x).strip() for x in cits if ":" in str(x)]
    cits = [x for x in cits if (not allowed) or (x in allowed)]
    cits = cits[:10]

    answer = _trim_to_words(answer, 120)

    if cits and "[" not in answer:
        answer = answer.strip() + " " + " ".join([f"[{c}]" for c in cits])

    return {"answer": answer, "citations": cits, "latency": latency}
