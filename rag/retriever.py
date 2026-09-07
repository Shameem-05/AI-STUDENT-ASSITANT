"""
retriever.py
------------
Implements the RAG (Retrieval-Augmented Generation) capability of the agent.

Pure Python standard library only -- no scikit-learn / numpy. This matters
for low-end devices (old laptops, Raspberry Pi, Termux on Android, etc.):
no compiled wheels to install, near-instant startup, tiny memory footprint.

Pipeline:
  1. Load all .txt knowledge-base documents from data_dir.
  2. Chunk each document into paragraph-sized passages.
  3. Build a TF-IDF index over the chunks using plain dicts.
  4. At query time, vectorize the question the same way and rank chunks by
     cosine similarity.
"""

import os
import re
import math
from dataclasses import dataclass

STOPWORDS = set("""a an the is are was were be been being of to in on at for with by from as and or but
if then than so this that these those it its i you he she they we my your his her their our what when
where who how do does did can could would should will just about not no""".split())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


@dataclass
class Chunk:
    text: str
    source: str


def _chunk_text(raw_text: str, source: str) -> list[Chunk]:
    raw_chunks = re.split(r"\n\s*\n", raw_text.strip())
    return [Chunk(text=c.strip(), source=source) for c in raw_chunks if len(c.strip()) > 20]


class Retriever:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.chunks: list[Chunk] = []
        self.doc_freq: dict[str, int] = {}
        self.doc_vectors: list[dict[str, float]] = []
        self.doc_norms: list[float] = []
        self._build_index()

    def _load_documents(self) -> None:
        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(self.data_dir, fname), "r", encoding="utf-8") as f:
                self.chunks.extend(_chunk_text(f.read(), source=fname))

    def _idf(self, term: str, n: int) -> float:
        return math.log((n + 1) / (self.doc_freq.get(term, 0) + 1)) + 1

    def _build_index(self) -> None:
        self._load_documents()
        if not self.chunks:
            raise RuntimeError(f"No .txt documents found in {self.data_dir}")

        tokenized = [tokenize(c.text) for c in self.chunks]
        n = len(tokenized)

        for tokens in tokenized:
            for term in set(tokens):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        for tokens in tokenized:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            vec = {t: c * self._idf(t, n) for t, c in tf.items()}
            norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            self.doc_vectors.append(vec)
            self.doc_norms.append(norm)

        self._n = n

    def _vectorize_query(self, text: str):
        tf: dict[str, int] = {}
        for t in tokenize(text):
            tf[t] = tf.get(t, 0) + 1
        vec = {t: c * self._idf(t, self._n) for t, c in tf.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return vec, norm

    def query(self, question: str, top_k: int = 3, min_score: float = 0.06) -> list[dict]:
        q_vec, q_norm = self._vectorize_query(question)
        scores = []
        for d_vec, d_norm in zip(self.doc_vectors, self.doc_norms):
            small, big = (q_vec, d_vec) if len(q_vec) < len(d_vec) else (d_vec, q_vec)
            dot = sum(w * big.get(t, 0.0) for t, w in small.items())
            scores.append(dot / (q_norm * d_norm))

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"text": self.chunks[i].text, "source": self.chunks[i].source, "score": round(scores[i], 4)}
            for i in ranked if scores[i] >= min_score
        ]

    def stats(self) -> dict:
        sources = sorted(set(c.source for c in self.chunks))
        return {"num_chunks": len(self.chunks), "sources": sources}
