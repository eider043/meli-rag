import os, json, argparse
import pandas as pd

from src.data_loader import load_laptops
from src.chunking import make_chunks
from src.retriever_bm25 import BM25Retriever
from src.pipeline import run_query
from src.config import TOP_K, OUTPUT_DIR, RUNS_PATH, INDEX_PREVIEW_PATH, EVAL_QUERIES_PATH
from src.evaluation import precision_recall_at_k, faithfulness, answer_coverage


def build_retriever():
    df = load_laptops()

    # Asegura que exista una columna laptop_id
    if "laptop_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "laptop_id"})

    rows = df.to_dict(orient="records")
    chunks = []
    for r in rows:
        chunks.extend(make_chunks(r, laptop_id_field="laptop_id"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(chunks[:200]).to_csv(INDEX_PREVIEW_PATH, index=False)

    return BM25Retriever(chunks)


def load_eval_queries(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Soporta lista o {"queries":[...]}
    if isinstance(data, dict) and "queries" in data:
        return data["queries"]
    if isinstance(data, list):
        return data

    raise ValueError("Formato inválido en eval_queries.json: usa lista o dict con key 'queries'.")


def run_demo(retriever):
    demo_queries = [
        "laptop con 16GB RAM y SSD",
        "laptop con procesador intel i7",
        "pantalla 15 pulgadas y peso liviano",
    ]
    runs = [run_query(q, retriever, top_k=TOP_K) for q in demo_queries]
    return runs, None


def run_single(retriever, query: str):
    out = run_query(query, retriever, top_k=TOP_K)
    return [out], None


def run_eval(retriever):
    eval_set = load_eval_queries(EVAL_QUERIES_PATH)

    runs = []
    metrics_rows = []

    for ex in eval_set:
        qid = ex.get("id", "")
        query = ex["query"]
        relevant = set(map(str, ex.get("relevant_laptop_ids", ex.get("relevant_ids", []))))

        result = run_query(query, retriever, top_k=TOP_K)
        runs.append(result)

        prec, rec = precision_recall_at_k(result, relevant, k=TOP_K)
        metrics_rows.append({
            "id": qid,
            "query": query,
            f"precision@{TOP_K}": prec,
            f"recall@{TOP_K}": rec,
            "faithfulness": faithfulness(result),          # real
            "answer_coverage": answer_coverage(result),    # real
            "critic_ok": result["critic_ok"],
            "n_retrieved": len(result["retrieved"]),
        })

    dfm = pd.DataFrame(metrics_rows)

    # Agregados
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

    return runs, dfm


def save_runs(runs, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "eval", "single"], default="demo")
    ap.add_argument("--query", type=str, default="")
    args = ap.parse_args()

    retriever = build_retriever()

    if args.mode == "single":
        if not args.query.strip():
            raise ValueError("En modo single debes pasar --query.")
        runs, _ = run_single(retriever, args.query)

        path = save_runs(runs, "runs_single.jsonl")
        print(f"Saved -> {path}")
        print(json.dumps(runs[0], ensure_ascii=False, indent=2))
        return

    if args.mode == "eval":
        runs, dfm = run_eval(retriever)

        runs_path = save_runs(runs, "eval_runs.jsonl")
        metrics_path = os.path.join(OUTPUT_DIR, "metrics_eval.csv")
        dfm.to_csv(metrics_path, index=False, encoding="utf-8")

        print(f"Saved logs -> {runs_path}")
        print(f"Saved metrics -> {metrics_path}")
        if len(dfm) > 0:
            print("\n=== SUMMARY (MEAN) ===")
            print(dfm.tail(1).to_string(index=False))
        return

    # demo
    runs, _ = run_demo(retriever)
    path = save_runs(runs, "runs.jsonl")
    print(f"Listo. Revisa {path} y {os.path.join(OUTPUT_DIR, 'index_preview.csv')}")


if __name__ == "__main__":
    main()
