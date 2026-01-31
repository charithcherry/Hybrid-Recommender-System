"""FastAPI service for multimodal recommender system."""

import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_loader import get_config


# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_id: int = Field(..., description="User ID to get recommendations for")
    n: int = Field(10, ge=1, le=100, description="Number of recommendations")
    filter_seen: bool = Field(True, description="Filter already seen items")


class ItemInfo(BaseModel):
    """Item information."""
    item_id: int
    title: str = ""
    description: str = ""
    score: float


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: int
    recommendations: List[ItemInfo]
    retrieval_time_ms: float
    model_version: str = "v1.0"
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


class StatsResponse(BaseModel):
    """System statistics response."""
    total_users: int
    total_items: int
    total_interactions: int
    model_type: str
    uptime_seconds: float


# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Recommender System API",
    description="Pinterest-style recommender with collaborative filtering and multimodal embeddings",
    version="1.0.0"
)

# Global state
class AppState:
    """Application state container."""
    def __init__(self):
        self.candidate_generator = None
        self.reranker = None
        self.products_df = None
        self.train_df = None
        self.config = None
        self.start_time = time.time()
        self.request_count = 0

state = AppState()


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("Loading models...")

    try:
        state.config = get_config()

        # Load products data
        products_path = Path("data/processed/products_processed.csv")
        if products_path.exists():
            state.products_df = pd.read_csv(products_path)
            print(f"Loaded {len(state.products_df)} products")

        # Load training data for filtering
        train_path = Path("data/processed/interactions_train.csv")
        if train_path.exists():
            state.train_df = pd.read_csv(train_path)
            print(f"Loaded {len(state.train_df)} training interactions")

        # Load candidate generator
        try:
            from src.models.candidate_generation import CandidateGenerator

            cg_path = Path("models/candidate_generation/candidate_generator.pkl")
            if cg_path.exists():
                state.candidate_generator = CandidateGenerator.load(str(cg_path))

                # Load CF model
                cf_path = Path("models/collaborative_filtering/matrix_factorization.pkl")
                if cf_path.exists():
                    state.candidate_generator.load_cf_model(str(cf_path))

                print("Candidate generator loaded")
            else:
                print("Candidate generator not found")

        except Exception as e:
            print(f"Error loading candidate generator: {e}")

        # Load re-ranker
        try:
            from src.models.reranking import ReRanker

            reranker_path = Path("models/reranking/reranker_best.pt")
            if reranker_path.exists():
                state.reranker = ReRanker.load(reranker_path)
                print("Re-ranker loaded")
            else:
                print("Re-ranker not found")

        except Exception as e:
            print(f"Error loading re-ranker: {e}")

        print("Startup complete!")

    except Exception as e:
        print(f"Error during startup: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multimodal Recommender System",
        "version": "1.0.0",
        "description": "Pinterest-style recommender with two-stage architecture",
        "endpoints": {
            "recommendations": "/recommend",
            "health": "/health",
            "stats": "/stats"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if state.candidate_generator is not None else "degraded",
        model_loaded=state.candidate_generator is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    n_users = 0
    n_items = 0
    n_interactions = 0

    if state.candidate_generator and state.candidate_generator.user_mapping:
        n_users = len(state.candidate_generator.user_mapping['to_idx'])
        n_items = len(state.candidate_generator.item_mapping['to_idx'])

    if state.train_df is not None:
        n_interactions = len(state.train_df)

    return StatsResponse(
        total_users=n_users,
        total_items=n_items,
        total_interactions=n_interactions,
        model_type="two_stage_pipeline",
        uptime_seconds=time.time() - state.start_time
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations for a user.

    Args:
        request: Recommendation request with user_id and parameters

    Returns:
        List of recommended items with scores
    """
    start_time = time.time()
    state.request_count += 1

    # Check if models are loaded
    if state.candidate_generator is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Stage 1: Candidate Generation
        candidates = state.candidate_generator.generate_hybrid_candidates(
            request.user_id,
            state.train_df,
            n=100
        )

        candidate_items = [item_id for item_id, _, _ in candidates]

        # Stage 2: Re-ranking
        if state.reranker:
            ranked = state.reranker.rerank(request.user_id, candidate_items)
            recommendations = ranked[:request.n]
        else:
            recommendations = [(item_id, score) for item_id, score, _ in candidates[:request.n]]

        # Enrich with product information
        enriched_recommendations = []

        for item_id, score in recommendations:
            item_info = ItemInfo(
                item_id=item_id,
                score=float(score)
            )

            # Add product details if available
            if state.products_df is not None:
                product_row = state.products_df[state.products_df['id'] == item_id]
                if len(product_row) > 0:
                    item_info.title = str(product_row.iloc[0].get('title', ''))
                    item_info.description = str(product_row.iloc[0].get('description', ''))[:200]

            enriched_recommendations.append(item_info)

        # Calculate retrieval time
        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=enriched_recommendations,
            retrieval_time_ms=retrieval_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations_get(
    user_id: int,
    n: int = Query(10, ge=1, le=100, description="Number of recommendations")
):
    """Get recommendations via GET request (alternative endpoint).

    Args:
        user_id: User ID
        n: Number of recommendations

    Returns:
        List of recommended items
    """
    request = RecommendationRequest(user_id=user_id, n=n)
    return await get_recommendations(request)


@app.get("/items/{item_id}", response_model=dict)
async def get_item_info(item_id: int):
    """Get information about a specific item.

    Args:
        item_id: Item ID

    Returns:
        Item information
    """
    if state.products_df is None:
        raise HTTPException(status_code=503, detail="Product data not loaded")

    product = state.products_df[state.products_df['id'] == item_id]

    if len(product) == 0:
        raise HTTPException(status_code=404, detail="Item not found")

    product_dict = product.iloc[0].to_dict()

    return {
        "item_id": item_id,
        "title": product_dict.get('title', ''),
        "description": product_dict.get('description', ''),
        "category": product_dict.get('category', ''),
        "style": product_dict.get('style', '')
    }


@app.get("/similar/{item_id}", response_model=List[ItemInfo])
async def get_similar_items(
    item_id: int,
    n: int = Query(10, ge=1, le=50, description="Number of similar items")
):
    """Get items similar to a given item.

    Args:
        item_id: Item ID
        n: Number of similar items

    Returns:
        List of similar items
    """
    if state.candidate_generator is None or state.candidate_generator.cf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        similar_items = state.candidate_generator.cf_model.recommend_similar_items(
            item_id, n=n
        )

        enriched_items = []
        for similar_id, score in similar_items:
            item_info = ItemInfo(item_id=similar_id, score=float(score))

            if state.products_df is not None:
                product = state.products_df[state.products_df['id'] == similar_id]
                if len(product) > 0:
                    item_info.title = str(product.iloc[0].get('title', ''))
                    item_info.description = str(product.iloc[0].get('description', ''))[:200]

            enriched_items.append(item_info)

        return enriched_items

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar items: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    host = config.get('api.host', '0.0.0.0')
    port = config.get('api.port', 8000)

    print(f"Starting API server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
