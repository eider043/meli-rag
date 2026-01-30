import re
from rank_bm25 import BM25Okapi

def tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]

class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5):
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]
