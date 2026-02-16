# Vector Database Implementation

Qdrant integration for advanced filtering and hybrid search capabilities.

---

## 🏗️ Architecture

```
ml_service/src/embeddings/
├── archive/faiss/              # FAISS implementation (legacy)
│   ├── clip_embeddings.py     # Original CLIP encoder
│   └── candidate_generator_faiss.index
│
└── vector_db/                  # Vector DB implementation (NEW)
    ├── qdrant/                # Qdrant client
    │   ├── __init__.py
    │   └── client.py          # Qdrant operations
    ├── hybrid_retriever.py    # Smart FAISS/Qdrant dispatcher
    ├── migrate_to_qdrant.py   # Migration script
    ├── requirements_vectordb.txt
    └── README.md (this file)
```

---

## 🚀 Quick Start

### Step 1: Install Qdrant Client

```bash
cd ml_service
pip install qdrant-client>=1.7.0
```

**No Docker required!** Qdrant runs in local storage mode.

### Step 2: Migrate Data from FAISS

```bash
cd ml_service
python src/embeddings/vector_db/migrate_to_qdrant.py
```

**Expected output:**
```
FAISS → QDRANT MIGRATION
======================================================================
1. Loading data...
   ✓ Loaded 44072 embeddings
   ✓ Loaded 44072 products
2. Connecting to Qdrant...
   ✓ Qdrant initialized (local storage)
3. Creating collection...
   ✓ Collection 'products' created
4. Indexing products...
   Indexed 1000/44072 products...
   ...
   ✓ Successfully indexed 44072 products
5. Verifying migration...
   ✓ Qdrant contains 44072 products

✓ MIGRATION SUCCESSFUL!
Storage location: ./qdrant_data (227MB)
```

---

## 📊 FAISS vs Qdrant Comparison

| Feature | FAISS (Archive) | Qdrant (New) |
|---------|----------------|----------------|
| **Raw Speed** | 9-12ms ⚡ | 20-40ms |
| **Metadata Filtering** | Post-filter (slow) | Pre-filter (fast!) ✅ |
| **Hybrid Search** | No | Yes (vector + payload) ✅ |
| **Complex Queries** | Manual | Built-in ✅ |
| **Real-time Updates** | Rebuild index | Dynamic add/delete ✅ |
| **Setup** | Simple | Simple (no Docker!) ✅ |
| **Storage** | Memory only | Persistent local files ✅ |

**When to Use Each:**

- **FAISS:** Simple similarity, speed-critical, no filters
- **Qdrant:** Complex filters, conversational search, persistent storage

---

## 🔍 Usage Examples

### Using Qdrant Client Directly

```python
from src.embeddings.vector_db.qdrant.client import QdrantVectorDB
import numpy as np

# Initialize client
qdrant = QdrantVectorDB(storage_path="./qdrant_data")

# Simple vector search
query_vector = np.random.rand(512)  # Your CLIP-encoded query
results = qdrant.search(
    query_vector=query_vector,
    limit=10
)

# Search with filters (THIS IS WHERE QDRANT SHINES!)
results = qdrant.search(
    query_vector=query_vector,
    filters={
        "gender": "Men",
        "articleType": "Watches",
        "usage": "Casual",
        "baseColour": "Black"
    },
    limit=10
)
```

### Using Hybrid Retriever (Recommended)

```python
from src.embeddings.vector_db.hybrid_retriever import HybridRetriever

# Initialize hybrid retriever
retriever = HybridRetriever(qdrant_storage_path="./qdrant_data")

# Smart dispatch: FAISS for simple, Qdrant for filtered
results = retriever.search(
    query_vector=query_vector,
    filters={"gender": "Men", "articleType": "Watches"},
    limit=10
)
# Automatically uses Qdrant because filters are present
```

### Conversational Service Integration

```python
# In conversational_service/src/api/main.py

from src.embeddings.vector_db.hybrid_retriever import HybridRetriever

state.hybrid_retriever = HybridRetriever(qdrant_storage_path="../ml_service/qdrant_data")

# Use hybrid retriever for all queries
candidates = state.hybrid_retriever.search(
    query_vector=clip_encode(query),
    filters=intent.get('filters', {}),
    limit=n
)
```

---

## 🎯 Migration Benefits

### Before (FAISS Only):
```python
# Search all 44K products
results = faiss.search(query_vec, k=1000)  # 10ms

# Filter in Python (slow!)
filtered = []
for r in results:
    product = products_df[products_df['id'] == r.id]
    if (product['gender'] == 'Men' and
        product['articleType'] == 'Watches' and
        product['baseColour'] == 'Black'):
        filtered.append(r)
# Total: 10ms + 40ms = 50ms
```

### After (Qdrant):
```python
# Search with pre-filtering (fast!)
results = qdrant.search(
    query_vector=query_vec,
    filters={
        "gender": "Men",
        "articleType": "Watches",
        "baseColour": "Black"
    },
    limit=10
)
# Total: 25ms (including filtering!)
# 2x faster for filtered queries!
```

---

## 🧪 Testing

### Test Hybrid Retriever
```bash
python ml_service/test_hybrid_system.py
```

### Performance Comparison
```bash
python ml_service/test_qdrant_vs_faiss.py
```

### Full Integration Test
```bash
python test_final_hybrid.py
```

---

## 📁 File Organization

**Archived (FAISS):**
- `embeddings/archive/faiss/` - Original FAISS implementation
- Still accessible via HybridRetriever for simple queries

**Active (Qdrant):**
- `embeddings/vector_db/qdrant/` - Qdrant implementation
- `embeddings/vector_db/hybrid_retriever.py` - Smart dispatcher
- `../../qdrant_data/` - Persistent storage (227MB)

**Data Storage:**
- `ml_service/qdrant_data/` - Qdrant local storage (227MB SQLite)
- No Docker, no external services required!

---

## 🐛 Troubleshooting

**Qdrant data corrupted:**
```bash
# Delete and re-migrate
rm -rf ml_service/qdrant_data
python src/embeddings/vector_db/migrate_to_qdrant.py
```

**Slow queries:**
- Check if filters are being used (should trigger Qdrant)
- Verify FAISS is used for simple queries (faster)
- Review HybridRetriever logs for dispatch decisions

**Migration encoding errors (Windows):**
- The migration script handles Windows cp1252 encoding issues
- If errors persist, check migration_log.txt

---

## 🎯 Current Status

✅ **Completed:**
1. Qdrant migration (44,072 products)
2. Local storage setup (no Docker needed)
3. Hybrid retriever with smart FAISS/Qdrant dispatch
4. Integration with conversational service
5. Test suite for validation

📋 **Next Steps:**
1. Add price filtering (TODO in filter_engine.py)
2. Integrate with main ml_service for filtered queries
3. Add conversation history persistence
4. Build chat interface in frontend

---

**Storage:** `./qdrant_data` (227MB local SQLite)
**Status:** Production ready
**Recommendation:** Use HybridRetriever for best performance
