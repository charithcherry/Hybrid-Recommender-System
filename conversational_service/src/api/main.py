"""Conversational Recommendation Service - Natural Language Interface."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

app = FastAPI(
    title="Conversational Recommendation Service",
    description="Natural language interface for product search and recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ConversationalQuery(BaseModel):
    """Natural language query request."""
    query: str = Field(..., description="Natural language query (e.g., 'red dress for summer wedding')")
    user_id: Optional[int] = Field(None, description="Optional user ID for personalization")
    n: int = Field(10, description="Number of recommendations")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn")


class ProductResult(BaseModel):
    """Product recommendation result."""
    item_id: int
    title: str
    description: str
    score: float
    explanation: Optional[str] = None  # Why this was recommended
    metadata: Optional[Dict] = None


class ConversationalResponse(BaseModel):
    """Conversational recommendation response."""
    query: str
    understood_intent: Dict  # What the LLM understood from query
    recommendations: List[ProductResult]
    explanation: str  # Overall explanation
    conversation_id: str
    retrieval_time_ms: float
    timestamp: str


class AppState:
    """Application state for conversational service."""
    def __init__(self):
        self.gemini_api_key = None  # Gemini API key (FREE tier!)
        self.faiss_retriever = None
        self.hybrid_retriever = None  # Hybrid FAISS + Qdrant
        self.products_df = None
        self.conversation_history: Dict[str, List[Dict]] = {}  # {conv_id: [messages]}

state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    print("Starting Conversational Recommendation Service...")

    try:
        # Initialize Gemini (FREE tier!)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            state.gemini_api_key = gemini_key
            model = os.getenv('LLM_MODEL', 'gemini-2.5-flash')
            print(f"Gemini API initialized (model: {model})")
            print("Using Gemini FREE tier (1500 requests/day)!")
        else:
            print("WARNING: GEMINI_API_KEY not found in .env!")

        # Load Hybrid Retriever (FAISS + Qdrant)
        sys.path.insert(0, "../ml_service/src")
        from embeddings.vector_db.hybrid_retriever import HybridRetriever

        qdrant_path = "../ml_service/qdrant_data"

        try:
            state.hybrid_retriever = HybridRetriever(
                qdrant_storage_path=qdrant_path
            )
            print(f"Hybrid Retriever loaded (FAISS + Qdrant)")

            # Also keep FAISS retriever for backward compatibility
            from src.retrieval.faiss_retriever import FAISSRetriever
            embeddings_path = Path(os.getenv("EMBEDDINGS_PATH", "../ml_service/data/embeddings/product_text_embeddings.npy"))
            products_path = Path(os.getenv("PRODUCTS_PATH", "../ml_service/data/processed/products_processed.csv"))

            if embeddings_path.exists() and products_path.exists():
                state.faiss_retriever = FAISSRetriever(
                    embeddings_path=str(embeddings_path),
                    products_path=str(products_path)
                )
                print(f"FAISS retriever also available ({state.faiss_retriever.num_items} items)")
        except Exception as e:
            print(f"WARNING: Hybrid retriever failed: {e}")
            print("Falling back to FAISS only...")
            # Fallback to FAISS only
            state.hybrid_retriever = None

        print("Startup complete! Service ready on port 8001")

    except Exception as e:
        print(f"Error during startup: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "name": "Conversational Recommendation Service",
        "version": "1.0.0",
        "description": "Natural language interface for product recommendations",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_provider": "gemini",
        "gemini_configured": state.gemini_api_key is not None,
        "faiss_loaded": state.faiss_retriever is not None,
        "hybrid_retriever": state.hybrid_retriever is not None,
        "retrieval_mode": "hybrid" if state.hybrid_retriever else "faiss_only",
        "timestamp": time.time()
    }


@app.post("/chat", response_model=ConversationalResponse)
async def conversational_search(query: ConversationalQuery):
    """
    Conversational recommendation endpoint.

    Examples:
    - "I need a red dress for a summer wedding under $200"
    - "Show me casual watches for men"
    - "What backpacks would go well with my style?"
    """
    start_time = time.time()

    try:
        # Step 1: Parse query with Gemini LLM
        from src.llm.query_parser_gemini import parse_user_query

        intent = await parse_user_query(
            query.query,
            gemini_api_key=state.gemini_api_key
        )

        print(f"Parsed intent: {intent}")

        # Step 2: Retrieve candidates using Hybrid Retriever (FAISS + Qdrant)
        # Encode query with CLIP first
        if state.faiss_retriever:
            # Use CLIP from FAISS retriever
            query_text = intent.get('search_query', query.query)
            query_vector = state.faiss_retriever.clip_encoder.encode(query_text)

            # Use hybrid retriever if available
            if state.hybrid_retriever:
                print(f"[Hybrid] Using hybrid retriever (smart FAISS/Qdrant selection)")
                candidates = state.hybrid_retriever.search(
                    query_vector=query_vector,
                    filters=intent.get('filters', {}),
                    limit=query.n * 2
                )
            else:
                # Fallback to FAISS only
                print("[Fallback] Using FAISS retriever only")
                candidates = state.faiss_retriever.search(
                    query_text=query_text,
                    filters=intent.get('filters', {}),
                    n=query.n * 2
                )
        else:
            candidates = []

        # Step 3: Apply LLM-parsed filters
        from src.retrieval.filter_engine import apply_llm_filters

        filtered_candidates = apply_llm_filters(
            candidates,
            intent.get('filters', {}),
            top_k=query.n
        )

        # Step 4: Generate explanations for top recommendations
        from src.llm.explanation_generator import generate_simple_explanation

        recommendations = []
        for product in filtered_candidates[:query.n]:
            # Add simple rule-based explanation (no LLM API cost)
            explanation = generate_simple_explanation(product, intent.get('filters', {}))
            product['explanation'] = explanation
            recommendations.append(product)

        # Step 5: Generate overall response explanation
        overall_explanation = f"I found {len(recommendations)} items matching your request"
        if intent.get('filters'):
            filter_desc = ", ".join([f"{k}: {v}" for k, v in intent['filters'].items()])
            overall_explanation += f" with filters: {filter_desc}"

        retrieval_time = (time.time() - start_time) * 1000

        # Generate conversation ID if not provided
        conv_id = query.conversation_id or f"conv_{int(time.time())}"

        return ConversationalResponse(
            query=query.query,
            understood_intent=intent,
            recommendations=recommendations,
            explanation=overall_explanation,
            conversation_id=conv_id,
            retrieval_time_ms=retrieval_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    except Exception as e:
        print(f"Error in conversational search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history."""
    if conversation_id in state.conversation_history:
        return {"conversation_id": conversation_id, "messages": state.conversation_history[conversation_id]}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVICE_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
