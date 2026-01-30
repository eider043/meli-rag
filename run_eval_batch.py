import json
from pathlib import Path
import pandas as pd

from src.data_loader import load_laptops
from src.chunking import make_chunks
from src.retriever_bm25 import BM25Retriever
from src.pipeline import run_query
from src.evaluation import precision_recall_at_k, faithfulness, answer_coverage
from src.config import TOP_K, EVAL_QUERIES_PATH

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_eval_queries(path: str):
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    # Soporta formato {"queries":[...]} o lista [...]
    if isinstance(data, dict) and "queries" in data:
        queries = data["queries"]
    elif isinstance(data, list):
        queries = data
    else:
        raise ValueError("Formato de eval_queries.json inválido. Debe ser lista o dict con key 'queries'.")

    # Normaliza llaves esperadas
    norm = []
    for q in queries:
        norm.append({
            "id": q.get("id", ""),
            "query": q["query"],
            "relevant_laptop_ids": q.get("relevant_laptop_ids", q.get("relevant_ids", []))
        })
    return norm


def build_retriever():
    df = load_laptops()

    if "laptop_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "laptop_id"})

    rows = df.to_dict(orient="records")
    chunks = []
    for r in rows:
        chunks.extend(make_chunks(r, laptop_id_field="laptop_id"))

    pd.DataFrame(chunks[:200]).to_csv(OUT_DIR / "index_preview.csv", index=False)
    return BM25Retriever(chunks)


def main():
    eval_set = load_eval_queries(EVAL_QUERIES_PATH)
    retriever = build_retriever()

    runs_path = OUT_DIR / "eval_runs.jsonl"
    if runs_path.exists():
        runs_path.unlink()  # evita append de ejecuciones anteriores

    metrics_rows = []

    for ex in eval_set:
        qid = ex["id"]
        query = ex["query"]
        relevant = set(map(str, ex.get("relevant_laptop_ids", [])))

        result = run_query(query, retriever, top_k=TOP_K)

        # métricas retrieval
        prec, rec = precision_recall_at_k(result, relevant, k=TOP_K)

        # métricas generación (reales)
        faith = faithfulness(result)
        cov = answer_coverage(result)

        metrics_rows.append({
            "id": qid,
            "query": query,
            f"precision@{TOP_K}": prec,
            f"recall@{TOP_K}": rec,
            "faithfulness": faith,
            "answer_coverage": cov,
            "critic_ok": result["critic_ok"],
            "n_retrieved": len(result["retrieved"]),
        })

        # log completo por query
        with open(runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    dfm = pd.DataFrame(metrics_rows)

    # agregados
    if len(dfm) > 0:
        summary = {
            "id": "MEAN",
            "query": "",
            f"precision@{TOP_K}": dfm[f"precision@{TOP_K}"].mean(),
            f"recall@{TOP_K}": dfm[f"recall@{TOP_K}"].mean(),
            "faithfulness": dfm["faithfulness"].mean(),
            "answer_coverage": dfm["answer_coverage"].mean(),
            "critic_ok": dfm["critic_ok"].mean(),
            "n_retrieved": dfm["n_retrieved"].mean(),
        }
        dfm = pd.concat([dfm, pd.DataFrame([summary])], ignore_index=True)

    out_csv = OUT_DIR / "metrics_eval.csv"
    dfm.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"Saved logs -> {runs_path}")
    print(f"Saved metrics -> {out_csv}")
    if len(dfm) > 0:
        print("\n=== SUMMARY (MEAN) ===")
        print(dfm.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
