# Hybrid Search Engine — Personal Knowledge Retrieval System

A production-ready hybrid retrieval system combining dense and sparse search methods with advanced reranking. Built as a learn-by-building project for the Advanced RAG Engineering path.

## 🎯 Overview

This project demonstrates how to build a powerful search engine that combines multiple retrieval techniques to find the most relevant information from a collection of documents. It's particularly useful for RAG (Retrieval-Augmented Generation) systems where retrieval quality directly impacts LLM output quality.

## ✨ Features

- **Dense Retrieval**: Semantic search using sentence transformers + FAISS vector index
- **Sparse Retrieval**: Exact keyword matching using BM25 algorithm
- **Reciprocal Rank Fusion (RRF)**: Intelligent fusion of multiple ranked result lists
- **Cross-Encoder Reranking**: Deep relevance scoring for final result refinement
- **Hybrid Pipeline**: Combines all techniques for best-of-both-worlds retrieval
- **Production-Ready**: Efficient, well-documented, and easy to extend

## 🏗️ Architecture

The search engine follows a 4-stage pipeline:

```
┌─────────────────────────────────────────────────────────┐
│                    Query Input                          │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
    ┌────────▼────────┐        ┌──────────▼──────────┐
    │  Dense Search   │        │   BM25 Search      │
    │  (FAISS Index)  │        │  (Term Matching)   │
    │  Top K results  │        │  Top K results     │
    └────────┬────────┘        └──────────┬──────────┘
             │                            │
             └────────────┬───────────────┘
                          │
                  ┌───────▼────────┐
                  │  RRF Fusion    │
                  │  Merge Ranks   │
                  │  Top Candidates│
                  └───────┬────────┘
                          │
                  ┌───────▼──────────────┐
                  │  Cross-Encoder      │
                  │  Reranking (optional)│
                  │  Final Top-K         │
                  └───────┬──────────────┘
                          │
                  ┌───────▼────────┐
                  │ Final Results  │
                  └────────────────┘
```

### Stage-by-Stage Breakdown

1. **Dense Retrieval**
   - Encodes query and documents using `sentence-transformers`
   - Builds FAISS index for fast nearest-neighbor search
   - Returns semantically similar results (handles meaning, not keywords)

2. **Sparse Retrieval (BM25)**
   - Classic ranking function based on term frequency (TF) and inverse document frequency (IDF)
   - Excellent for exact keyword matching
   - Fast, interpretable, and requires no neural networks

3. **Reciprocal Rank Fusion**
   - Combines rankings from both systems without mixing score scales
   - Formula: `RRF(d) = Σ 1 / (k + rank(d))`
   - Prevents any single system from dominating results

4. **Cross-Encoder Reranking**
   - Reads query + document **together** for deeper relevance scoring
   - More accurate than semantic similarity alone
   - Applied only to top candidates (computationally expensive)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hybrid_search
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install sentence-transformers faiss-cpu numpy
   ```
   
   *For GPU support, use `faiss-gpu` instead of `faiss-cpu`*

### Quick Start

```bash
python hybrid_search.py
```

This will:
- Build the hybrid search engine with default documents
- Run 7 different test queries showcasing various search scenarios
- Display results from dense, sparse, RRF, and reranking stages

## 📖 Usage

### Basic Example

```python
from hybrid_search import HybridSearchEngine

# Initialize engine
engine = HybridSearchEngine(use_reranker=False)  # or True for slower, more accurate results

# Index your documents
documents = [
    "Dense retrieval uses neural embeddings for semantic search",
    "BM25 is great for keyword matching",
    "Hybrid retrieval combines both approaches",
    # ... more documents
]
engine.index(documents)

# Search
results = engine.search(
    query="How does semantic search work?",
    top_k=5,           # Return top 5 results
    candidate_pool=20, # Consider top 20 from each retriever
    verbose=True       # Print intermediate stages
)

# Process results
for result in results:
    print(f"#{result['rank']} — Score: {result['score']}")
    print(f"{result['text']}")
    print(f"(Retrieved by: {result['stage']})\n")
```

### Customization

**Disable reranking for speed:**
```python
engine = HybridSearchEngine(use_reranker=False)
```

**Tune BM25 parameters:**
```python
bm25 = BM25Retriever(k1=2.0, b=0.75)  # Higher k1 = reward term frequency more
```

**Use different embedding model:**
```python
dense = DenseRetriever(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
```

## 📊 Example Output

Query: `"how does semantic search find similar meanings"`

```
Stage 1 — Dense Retrieval (top 3):
  (0.847) Dense retrieval uses neural embeddings to represent text as vectors in high-dimensional space...
  (0.621) Sentence transformers fine-tune BERT-style models to produce semantically meaningful sentence...
  (0.512) Hybrid retrieval outperforms pure dense or sparse methods by combining the strengths of both...

Stage 2 — BM25 (top 3):
  (3.218) Dense retrieval uses neural embeddings to represent text as vectors...
  (2.891) Hybrid retrieval outperforms pure dense or sparse methods...
  (1.045) FAISS is a library for efficient similarity search over dense vector collections...

Stage 3 — After RRF (top 3):
  (rrf=0.0341) Dense retrieval uses neural embeddings...
  (rrf=0.0341) Hybrid retrieval outperforms pure dense or sparse methods...
  (rrf=0.0227) Sentence transformers fine-tune BERT-style models...

Final Results (Cross-Encoder):
  #1 (score=6.4832)
  Dense retrieval uses neural embeddings to represent text as vectors...
```

## 📁 Project Structure

```
hybrid_search/
├── hybrid_search.py          # Main implementation (all-in-one file)
├── .venv/                    # Virtual environment
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Generate semantic embeddings |
| `faiss-cpu` | Fast vector similarity search |
| `numpy` | Numerical operations |

*Standard library only: `math`, `typing`*

## 💡 Key Parameters

### DenseRetriever
- `model_name`: Hugging Face model ID (default: `"all-MiniLM-L6-v2"`)

### BM25Retriever
- `k1` (1.2–2.0): Controls term frequency saturation (higher = reward frequent terms more)
- `b` (0–1): Controls document length normalization (0.75 typical)

### HybridSearchEngine.search()
- `top_k`: Number of final results to return
- `candidate_pool`: How many candidates to retrieve from each system before fusion
- `verbose`: Print intermediate results

### reciprocal_rank_fusion()
- `k` (default 60): Constant preventing high scores from dominating

## 🎓 Learning Objectives

By studying this codebase, you'll understand:

- ✅ How dense and sparse retrieval complement each other
- ✅ Vector embeddings and FAISS indexing
- ✅ BM25 ranking algorithm
- ✅ Combining multiple ranking systems without score normalization
- ✅ Cross-encoder architectures for reranking
- ✅ RAG system fundamentals
- ✅ Production search engine patterns

## 📈 Performance Notes

| Operation | Time (10K docs) | Notes |
|-----------|-----------------|-------|
| Embedding documents | ~5 seconds | One-time cost, parallelizable |
| BM25 indexing | <0.1 seconds | Nearly instant |
| Dense query | ~0.05 seconds | Depends on query batch size |
| BM25 query | <0.01 seconds | Very fast |
| RRF fusion | <0.001 seconds | Negligible |
| Cross-encoder rerank | ~0.5 seconds | Only on top candidates |

## 🔮 Future Improvements

- [ ] Batch query processing for throughput
- [ ] Approximate nearest neighbor (IVF) for million-scale indexes
- [ ] Caching layer for repeated queries
- [ ] Async document loading
- [ ] Custom tokenizers for non-English text
- [ ] Result clustering
- [ ] Query expansion with synonyms
- [ ] Domain-specific embedding fine-tuning
- [ ] Interactive visualization dashboard

## 📝 Example Use Cases

- **RAG for LLMs**: Ground model outputs in retrieved documents
- **Documentation Search**: Find relevant docs from knowledge bases
- **Email/Ticket Search**: Find similar historical issues
- **Scientific Paper Search**: Find related research
- **Personal Knowledge Management**: Search personal notes/articles

## 🐛 Troubleshooting

**"ModuleNotFoundError: No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers
```

**"FAISS CPU is not compiled with SSE2"** (Windows)
```bash
pip install faiss-cpu --no-binary faiss-cpu
```

**Slow on first run**
- First query triggers model downloads (sentence-transformers, cross-encoder)
- Subsequent runs are much faster

**Out of memory**
- Reduce `candidate_pool` parameter
- Use `faiss-cpu` instead of `faiss-gpu`
- Process documents in smaller batches

## 📚 References

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Reciprocal Rank Fusion](https://www.semanticscholar.org/paper/Reciprocal-rank-fusion-outperforms-condorcet-and-Cormack-Clarke/7ea70b5d1cd8c9d2e7bc20b2af39a6a5e3c09e32)
- [Cross-Encoder Models](https://www.sbert.net/examples/applications/cross-encoder/README.html)

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍💻 Author

Built as part of the Advanced RAG Engineering learning path.

---

**Happy searching!** 🔍

For questions or improvements, feel free to extend this project with your own documents and queries.
