"""
Personal Knowledge Search Engine — Hybrid Retrieval
=====================================================
Learn-by-building project for the Advanced RAG Engineering path.

Covers:
  - Dense retrieval (sentence-transformers + FAISS)
  - Sparse retrieval (BM25)
  - Reciprocal Rank Fusion (RRF)
  - Cross-encoder reranking

Usage:
  python hybrid_search.py

Add your own documents to the DOCUMENTS list at the bottom.
"""

import math
import numpy as np
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────
# STEP 1 — Dense Retrieval
# Uses a sentence-transformer model to embed text into vectors.
# FAISS finds the nearest vectors (most semantically similar docs).
# ─────────────────────────────────────────────

class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Dense] Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        import faiss

        self.model = SentenceTransformer(model_name)
        self.faiss = faiss
        self.docs: List[str] = []
        self.index = None

    def index_documents(self, docs: List[str]):
        """Embed all documents and build a FAISS index."""
        self.docs = docs
        print(f"[Dense] Embedding {len(docs)} documents...")
        embeddings = self.model.encode(docs, convert_to_numpy=True, show_progress_bar=False)

        # Normalize so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

        dim = embeddings.shape[1]
        self.index = self.faiss.IndexFlatIP(dim)  # Inner Product (cosine after normalization)
        self.index.add(embeddings)
        print(f"[Dense] Index built. Vector dimension: {dim}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return list of (doc_index, score) sorted by relevance."""
        q_vec = self.model.encode([query], convert_to_numpy=True)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        scores, indices = self.index.search(q_vec, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]


# ─────────────────────────────────────────────
# STEP 2 — Sparse Retrieval (BM25)
# Exact keyword matching with term frequency saturation
# and document length normalization.
# ─────────────────────────────────────────────

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        k1: controls term frequency saturation (1.2–2.0 typical)
        b:  controls document length normalization (0.75 typical)
        """
        self.k1 = k1
        self.b = b
        self.docs: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self.df: Dict[str, int] = {}      # document frequency per term
        self.idf: Dict[str, float] = {}   # inverse document frequency
        self.avgdl: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer. Swap for a better one if needed."""
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def index_documents(self, docs: List[str]):
        """Build BM25 index from documents."""
        self.docs = docs
        self.tokenized_docs = [self._tokenize(d) for d in docs]
        N = len(docs)
        self.avgdl = sum(len(td) for td in self.tokenized_docs) / N

        # Compute document frequencies
        self.df = {}
        for td in self.tokenized_docs:
            for term in set(td):
                self.df[term] = self.df.get(term, 0) + 1

        # Compute IDF using Robertson-Sparck Jones formula
        self.idf = {
            term: math.log((N - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.df.items()
        }
        print(f"[BM25] Index built. Vocabulary size: {len(self.df)} terms")

    def _score_doc(self, query_terms: List[str], doc_idx: int) -> float:
        """BM25 score for one document."""
        td = self.tokenized_docs[doc_idx]
        dl = len(td)
        tf_map: Dict[str, int] = {}
        for term in td:
            tf_map[term] = tf_map.get(term, 0) + 1

        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return list of (doc_index, score) sorted by relevance."""
        query_terms = self._tokenize(query)
        scores = [(i, self._score_doc(query_terms, i)) for i in range(len(self.docs))]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ─────────────────────────────────────────────
# STEP 3 — Reciprocal Rank Fusion
# Merges two ranked lists by rank position, not raw scores.
# A document ranked highly in both lists bubbles to the top.
# Formula: RRF(d) = Σ 1 / (k + rank(d))  where k=60
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    ranked_lists: list of [(doc_idx, score), ...] each sorted best-first
    k:            constant to prevent high scores from dominating (default 60)
    Returns merged list of (doc_idx, rrf_score) sorted best-first
    """
    rrf_scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_idx, _) in enumerate(ranked, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


# ─────────────────────────────────────────────
# STEP 4 — Cross-Encoder Reranking
# Reads query + document TOGETHER for deeper relevance scoring.
# Only applied to top-k candidates (slow but accurate).
# ─────────────────────────────────────────────

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[Reranker] Loading cross-encoder: {model_name}")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float]],
        docs: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Re-score the top candidates using the cross-encoder.
        Returns (doc_idx, new_score) sorted best-first.
        """
        pairs = [(query, docs[idx]) for idx, _ in candidates]
        scores = self.model.predict(pairs)
        reranked = sorted(
            [(candidates[i][0], float(scores[i])) for i in range(len(candidates))],
            key=lambda x: x[1],
            reverse=True
        )
        return reranked[:top_k]


# ─────────────────────────────────────────────
# STEP 5 — The Full Pipeline
# Puts it all together: dense + BM25 → RRF → rerank
# ─────────────────────────────────────────────

class HybridSearchEngine:
    def __init__(self, use_reranker: bool = True):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self.docs: List[str] = []

    def index(self, docs: List[str]):
        """Index all documents in both dense and sparse systems."""
        self.docs = docs
        self.dense.index_documents(docs)
        self.bm25.index_documents(docs)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 20,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Full hybrid search:
          1. Dense retrieval → top candidate_pool results
          2. BM25 retrieval  → top candidate_pool results
          3. RRF fusion      → merged ranked list
          4. Cross-encoder reranking (optional) → final top_k
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: \"{query}\"")
            print('='*60)

        # Stage 1: retrieve candidates from both systems
        dense_results = self.dense.search(query, top_k=candidate_pool)
        bm25_results = self.bm25.search(query, top_k=candidate_pool)

        if verbose:
            print(f"\n[Dense top 3]")
            for idx, score in dense_results[:3]:
                print(f"  ({score:.3f}) {self.docs[idx][:80]}...")

            print(f"\n[BM25 top 3]")
            for idx, score in bm25_results[:3]:
                print(f"  ({score:.3f}) {self.docs[idx][:80]}...")

        # Stage 2: fuse with RRF
        fused = reciprocal_rank_fusion([dense_results, bm25_results])

        if verbose:
            print(f"\n[After RRF — top 3]")
            for idx, score in fused[:3]:
                print(f"  (rrf={score:.4f}) {self.docs[idx][:80]}...")

        # Stage 3: rerank if available
        if self.reranker:
            final = self.reranker.rerank(query, fused[:candidate_pool], self.docs, top_k=top_k)
            stage = "cross-encoder"
        else:
            final = fused[:top_k]
            stage = "RRF"

        # Format results
        results = []
        for rank, (idx, score) in enumerate(final, start=1):
            results.append({
                "rank": rank,
                "doc_index": idx,
                "score": round(score, 4),
                "text": self.docs[idx],
                "stage": stage
            })

        if verbose:
            print(f"\n[Final results — {stage}]")
            for r in results:
                print(f"\n  #{r['rank']} (score={r['score']})")
                print(f"  {r['text']}")

        return results


# ─────────────────────────────────────────────
# DEMO — run this to see it all working
# Replace DOCUMENTS with your own content!
# ─────────────────────────────────────────────

DOCUMENTS = [
    # Machine learning
    "Dense retrieval uses neural embeddings to represent text as vectors in high-dimensional space, enabling semantic similarity search.",
    "BM25 is a classic sparse retrieval algorithm based on term frequency and inverse document frequency, great for exact keyword matching.",
    "Reciprocal Rank Fusion combines results from multiple retrieval systems by using rank positions rather than raw scores.",
    "Cross-encoder models read the query and document together to produce highly accurate relevance scores but are slow at scale.",
    "FAISS is a library for efficient similarity search over dense vector collections, developed by Meta AI Research.",
    "Sentence transformers fine-tune BERT-style models to produce semantically meaningful sentence embeddings.",
    "Hybrid retrieval outperforms pure dense or sparse methods by combining the strengths of both approaches.",
    "RAG (retrieval-augmented generation) grounds language model outputs in retrieved documents to reduce hallucination.",

    # Python / coding
    "Python list comprehensions provide a concise way to create lists: [x*2 for x in range(10)].",
    "The BM25 k1 parameter controls term frequency saturation. Higher values reward repeated terms more.",
    "FAISS IndexFlatIP performs exact inner product search. For large datasets, use IndexIVFFlat for speed.",
    "Cosine similarity measures the angle between two vectors, making it robust to differences in document length.",

    # General knowledge
    "The Indus Valley Civilization thrived around 2500 BCE in what is now Pakistan and northwest India.",
    "Photosynthesis converts light energy into chemical energy stored in glucose using chlorophyll.",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
    "Lahore is the capital of Punjab province in Pakistan and is known for Mughal-era architecture.",
]


if __name__ == "__main__":
    print("Building hybrid search engine...\n")
    engine = HybridSearchEngine(use_reranker=False)
    engine.index(DOCUMENTS)

    # Test queries — these show different strengths
    queries = [
        "how does semantic search find similar meanings",   # dense wins
        "BM25 k1 parameter",                               # bm25 wins (exact term)
        "combining multiple search result lists",           # both contribute
        "Lahore Punjab Pakistan",                           # geographic exact match
    ]

    for q in queries:
        engine.search(q, top_k=3, candidate_pool=10)

    print("\n\nAll done! Try adding your own documents and queries.")
