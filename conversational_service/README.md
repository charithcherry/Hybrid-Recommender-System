# Conversational Recommendation Service

Natural language interface for product search and recommendations using OpenAI GPT models.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd conversational_service
pip install -r requirements.txt
```

### 2. Configure Environment
Your `.env` file is already configured with:
- ✅ OpenAI API key
- ✅ Model: gpt-4o-mini (cost-effective!)
- ✅ Service port: 8001

### 3. Start Service
```bash
python -m uvicorn src.api.main:app --port 8001 --reload
```

### 4. Test It
```bash
# In another terminal
cd conversational_service
python tests/test_conversational.py
```

---

## 💬 Example Queries

### Simple Searches
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "casual watches for men", "n": 5}'
```

### Complex Queries with Filters
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "red dress for summer wedding under $200", "n": 10}'
```

### Personalized Queries
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "show me items similar to what I liked", "user_id": 1001, "n": 10}'
```

---

## 🏗️ Architecture

```
User Query
    ↓
LLM Query Parser (GPT-4o-mini)
    ↓ Extracts: intent, filters, preferences
FAISS Vector Search
    ↓ Searches: 44K products
Filter Engine
    ↓ Applies: gender, category, color, etc.
LLM Explanation Generator
    ↓ Generates: "Perfect for summer weddings..."
Response to User
```

---

## 📊 Cost Estimation (with $20/month plan)

### Using gpt-4o-mini (Current Config):
- **Cost per query:** ~$0.0002 (very cheap!)
- **Queries per $20:** ~100,000 queries
- **Daily budget:** ~3,300 queries/day
- **Perfect for:** Development, testing, moderate production

### If You Switch to gpt-4o:
- **Cost per query:** ~$0.0025 (10x more expensive)
- **Queries per $20:** ~8,000 queries
- **Daily budget:** ~265 queries/day
- **Use for:** Higher quality responses when needed

---

## 🎯 Features

✅ **Natural Language Understanding**
- "red dress for summer wedding" → Parsed filters + intent

✅ **Multi-Filter Extraction**
- Automatically detects: gender, category, color, season, usage, price

✅ **Vector Search Integration**
- Uses existing FAISS index (44K products)

✅ **LLM-Powered Explanations**
- "Perfect for summer weddings, matches your preference for elegant styles"

✅ **Cost-Efficient**
- Uses gpt-4o-mini (40x cheaper than GPT-4)
- Batch explanations to save tokens

---

## 🔧 Configuration

Edit `.env` to customize:

```env
# Switch models based on needs
LLM_MODEL=gpt-4o-mini  # Cheap, 100K queries/month
# LLM_MODEL=gpt-4o     # Better quality, 8K queries/month
# LLM_MODEL=gpt-5.2    # Best quality, 2.5K queries/month

# Adjust response length (lower = cheaper)
MAX_TOKENS=500  # Default

# Temperature (0 = deterministic, 1 = creative)
LLM_TEMPERATURE=0.7
```

---

## 📁 Project Structure

```
conversational_service/
├── src/
│   ├── api/
│   │   └── main.py                   # FastAPI endpoints
│   ├── llm/
│   │   ├── query_parser.py          # Parse natural language
│   │   └── explanation_generator.py  # Generate explanations
│   └── retrieval/
│       ├── faiss_retriever.py       # FAISS wrapper
│       └── filter_engine.py         # Apply filters
├── tests/
│   └── test_conversational.py       # Integration tests
├── .env                              # API keys (gitignored!)
├── .env.template                     # Template for setup
└── requirements.txt
```

---

## 🧪 Testing

### Start Service
```bash
python -m uvicorn src.api.main:app --port 8001 --reload
```

### Run Tests
```bash
python tests/test_conversational.py
```

### Manual Testing
```bash
# Health check
curl http://localhost:8001/health

# Test query
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "casual shoes for men", "n": 5}' | python -m json.tool
```

---

## 🔜 Next Steps

1. ✅ Basic conversational service (DONE)
2. 🔄 Test with real OpenAI API
3. 📊 Migrate to Vector DB (Weaviate/Qdrant)
4. 🎨 Add to frontend UI (chat interface)
5. 📈 Add conversation history persistence

---

## 💡 Tips

**Save Costs:**
- Use gpt-4o-mini for development
- Cache common queries
- Batch explanations (1 call for multiple products)
- Reduce MAX_TOKENS if responses are too long

**Improve Quality:**
- Add user's past interactions to context
- Use conversation history for multi-turn
- Fine-tune prompts for better parsing

---

**Status:** Ready to test!
**Port:** 8001
**Model:** gpt-4o-mini (cost-efficient)
**API Key:** ✅ Configured and secured
