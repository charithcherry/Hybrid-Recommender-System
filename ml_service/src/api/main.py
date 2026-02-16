"""FastAPI service for multimodal recommender system."""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Literal, Tuple
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import time
import json
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
    interaction_count: Optional[int] = None


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


# New models for user management and interactions
class UserRegistration(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)
    email: Optional[str] = None


class UserLogin(BaseModel):
    """User login request."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response."""
    success: bool
    user_id: Optional[int] = None
    username: Optional[str] = None
    message: str


class UserProfile(BaseModel):
    """User profile response."""
    user_id: int
    username: str
    created_at: str
    total_interactions: int


class InteractionRequest(BaseModel):
    """User interaction tracking request."""
    user_id: int
    item_id: int
    interaction_type: Literal["view", "like", "dislike", "save", "buy"]


class InteractionResponse(BaseModel):
    """Interaction tracking response."""
    success: bool
    interaction_id: Optional[int] = None
    timestamp: str
    message: str
    action: str  # 'added', 'removed', 'updated'
    current_interaction: Optional[str] = None  # Current active interaction type


class UserInteraction(BaseModel):
    """User interaction details."""
    interaction_id: int
    item_id: int
    interaction_type: str
    rating: int
    timestamp: str


class PaginatedProductsResponse(BaseModel):
    """Paginated products response."""
    products: List[dict]
    total: int
    page: int
    limit: int
    total_pages: int


class FilterValuesResponse(BaseModel):
    """Available filter values."""
    genders: List[str]
    masterCategories: List[str]
    subCategories: List[str]
    articleTypes: List[str]
    baseColours: List[str]
    seasons: List[str]
    usages: List[str]


class SplitRecommendationResponse(BaseModel):
    """Split recommendation response with CF and content-based sections."""
    user_id: int
    cf_recommendations: List[ItemInfo]
    content_recommendations: List[ItemInfo]
    retrieval_time_ms: float
    timestamp: str


class InteractionStateResponse(BaseModel):
    """User's current interaction states for items."""
    user_id: int
    interaction_states: Dict[int, str]  # {item_id: interaction_type}


# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Recommender System API",
    description="Pinterest-style recommender with collaborative filtering and multimodal embeddings",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

        # Full CLIP embeddings for all 44K products
        self.full_embeddings = None
        self.products_id_mapping = None  # {item_id: index in products_df}

        # New: User management and interaction tracking
        self.users: Dict[int, dict] = {}  # {user_id: {username, created_at, ...}}
        self.interactions: List[dict] = []  # [{interaction_id, user_id, item_id, type, rating, timestamp}]
        self.interaction_states: Dict[int, Dict[int, str]] = {}  # {user_id: {item_id: interaction_type}}
        self.item_interaction_counts: Dict[int, int] = {}  # {item_id: count} - real-time interaction counts
        self.category_specialist_index: Dict[str, List[Tuple[int, float]]] = {}  # {category: [(user_id, specialization_score)]}
        self.next_user_id: int = 1000  # Start after training users (0-999)
        self.next_interaction_id: int = 0

state = AppState()


# Interaction rating mapping
INTERACTION_RATINGS = {
    'view': 1,
    'like': 3,
    'dislike': 1,
    'save': 4,
    'buy': 5
}

# Related categories for cross-category discovery (70% same, 20% related, 10% diverse)
RELATED_CATEGORIES = {
    'Watches': ['Belts', 'Cufflinks', 'Ties', 'Wallets', 'Sunglasses', 'Jewellery'],
    'Shoes': ['Socks', 'Flip Flops', 'Sandals', 'Shoe Accessories'],
    'Casual Shoes': ['Socks', 'Flip Flops', 'Sandals', 'Sports Shoes'],
    'Sports Shoes': ['Socks', 'Track Pants', 'Shorts', 'Tshirts'],
    'Formal Shoes': ['Socks', 'Formal Trousers', 'Shirts', 'Ties', 'Belts'],
    'Shirts': ['Ties', 'Belts', 'Jackets', 'Sweaters', 'Blazers'],
    'Tshirts': ['Jeans', 'Shorts', 'Track Pants', 'Casual Shoes'],
    'Jeans': ['Belts', 'Casual Shoes', 'Tshirts', 'Shirts'],
    'Backpacks': ['Wallets', 'Belts', 'Accessories', 'Caps'],
    'Handbags': ['Wallets', 'Sunglasses', 'Watches', 'Jewellery'],
    'Jackets': ['Shirts', 'Sweaters', 'Jeans', 'Casual Shoes'],
    'Sweaters': ['Shirts', 'Jeans', 'Jackets', 'Casual Shoes'],
    'Caps': ['Tshirts', 'Sunglasses', 'Backpacks', 'Casual Shoes'],
    'Sunglasses': ['Watches', 'Caps', 'Handbags', 'Wallets']
}

# JSON file paths for persistence
USERS_DB_PATH = Path("data/users_db.json")
INTERACTIONS_DB_PATH = Path("data/interactions_db.json")
INTERACTION_STATES_PATH = Path("data/interaction_states_db.json")


def save_users_to_json():
    """Save users to JSON file."""
    try:
        with open(USERS_DB_PATH, 'w') as f:
            json.dump({
                'users': state.users,
                'next_user_id': state.next_user_id
            }, f, indent=2)
    except Exception as e:
        print(f"Error saving users: {e}")


def save_interactions_to_json():
    """Save interactions to JSON file."""
    try:
        with open(INTERACTIONS_DB_PATH, 'w') as f:
            json.dump({
                'interactions': state.interactions,
                'next_interaction_id': state.next_interaction_id
            }, f, indent=2)
    except Exception as e:
        print(f"Error saving interactions: {e}")


def load_users_from_json():
    """Load users from JSON file."""
    try:
        if USERS_DB_PATH.exists():
            with open(USERS_DB_PATH, 'r') as f:
                data = json.load(f)
                # Convert string keys back to integers
                state.users = {int(k): v for k, v in data.get('users', {}).items()}
                state.next_user_id = data.get('next_user_id', 1000)
                print(f"Loaded {len(state.users)} users from JSON database")
        else:
            print("No existing user database found - starting fresh")
    except Exception as e:
        print(f"Error loading users: {e}")


def load_interactions_from_json():
    """Load interactions from JSON file."""
    try:
        if INTERACTIONS_DB_PATH.exists():
            with open(INTERACTIONS_DB_PATH, 'r') as f:
                data = json.load(f)
                state.interactions = data.get('interactions', [])
                state.next_interaction_id = data.get('next_interaction_id', 0)
                print(f"Loaded {len(state.interactions)} interactions from JSON database")
        else:
            print("No existing interactions database found - starting fresh")
    except Exception as e:
        print(f"Error loading interactions: {e}")


def save_interaction_states_to_json():
    """Save interaction states to JSON file."""
    try:
        # Convert to string keys for JSON serialization
        json_states = {
            str(user_id): {str(item_id): itype for item_id, itype in items.items()}
            for user_id, items in state.interaction_states.items()
        }
        with open(INTERACTION_STATES_PATH, 'w') as f:
            json.dump(json_states, f, indent=2)
    except Exception as e:
        print(f"Error saving interaction states: {e}")


def load_interaction_states_from_json():
    """Load interaction states from JSON file."""
    try:
        if INTERACTION_STATES_PATH.exists():
            with open(INTERACTION_STATES_PATH, 'r') as f:
                json_states = json.load(f)
                # Convert string keys back to integers
                state.interaction_states = {
                    int(user_id): {int(item_id): itype for item_id, itype in items.items()}
                    for user_id, items in json_states.items()
                }
                print(f"Loaded interaction states for {len(state.interaction_states)} users")
        else:
            print("No existing interaction states database found - starting fresh")
    except Exception as e:
        print(f"Error loading interaction states: {e}")


def build_category_specialist_index(
    train_df: pd.DataFrame,
    products_df: pd.DataFrame,
    min_specialization: float = 0.30,
    min_interactions: int = 5,
    max_specialists: int = 50
) -> Dict[str, List[Tuple[int, float]]]:
    """Build specialist index at startup. O(users * interactions) once.

    Args:
        train_df: Training interactions DataFrame
        products_df: Products DataFrame with categories
        min_specialization: Minimum % of items in category to be specialist (default 30%)
        min_interactions: Minimum interactions required to be considered
        max_specialists: Maximum specialists to keep per category

    Returns:
        Dict mapping category to list of (user_id, specialization_score) tuples
    """
    from collections import defaultdict

    specialist_index = defaultdict(list)

    for user_id in range(1000):  # Training users 0-999
        # Get liked items (rating >= 3)
        user_interactions = train_df[
            (train_df['user_id'] == user_id) &
            (train_df['rating'] >= 3)
        ]

        if len(user_interactions) < min_interactions:
            continue

        # Join with products to get categories
        item_ids = user_interactions['item_id'].tolist()
        products = products_df[products_df['id'].isin(item_ids)]

        if len(products) == 0:
            continue

        # Count items per category
        category_counts = products['articleType'].value_counts()
        total_items = len(products)

        # Add to index for categories where user is specialist (>= 30%)
        for category, count in category_counts.items():
            specialization = count / total_items
            if specialization >= min_specialization:
                specialist_index[category].append((user_id, specialization))

    # Sort by specialization score, keep top N
    for category in specialist_index:
        specialist_index[category] = sorted(
            specialist_index[category],
            key=lambda x: x[1],
            reverse=True
        )[:max_specialists]

    return dict(specialist_index)


def get_category_specialists(category: str, top_k: int = 20) -> List[Tuple[int, float]]:
    """O(1) lookup of category specialists.

    Args:
        category: Product category (articleType)
        top_k: Number of top specialists to return

    Returns:
        List of (user_id, specialization_score) tuples, sorted by score descending
    """
    if category not in state.category_specialist_index:
        return []
    return state.category_specialist_index[category][:top_k]


def detect_user_interest_clusters(
    user_interactions: List[dict],
    products_df: pd.DataFrame,
    min_percentage: float = 0.20,
    min_items: int = 3,
    max_clusters: int = 3
) -> List[Tuple[str, float, List[int]]]:
    """Detect multiple interest clusters from user interactions.

    Args:
        user_interactions: List of interaction dicts with item_id and rating
        products_df: Products DataFrame
        min_percentage: Minimum % of items to form a cluster (default 20%)
        min_items: Minimum items to form a cluster (default 3)
        max_clusters: Maximum number of clusters to return (default 3)

    Returns:
        List of (category, weight, item_ids) tuples, sorted by weight descending
    """
    # Get liked items (rating >= 3)
    liked_items = [i for i in user_interactions if i.get('rating', 0) >= 3]
    if len(liked_items) == 0:
        return []

    item_ids = [i['item_id'] for i in liked_items]
    products = products_df[products_df['id'].isin(item_ids)]

    if len(products) == 0:
        return []

    # Count items per category (articleType)
    category_counts = products['articleType'].value_counts()
    total_items = len(products)

    # Identify significant clusters (>= 20% OR >= 3 items)
    clusters = []
    for category, count in category_counts.items():
        percentage = count / total_items
        if percentage >= min_percentage or count >= min_items:
            category_items = products[products['articleType'] == category]['id'].tolist()
            clusters.append((category, percentage, category_items))

    # Sort by weight descending, take top N
    clusters = sorted(clusters, key=lambda x: x[1], reverse=True)[:max_clusters]

    # Normalize weights to sum to 1.0
    total_weight = sum(w for _, w, _ in clusters)
    if total_weight > 0:
        clusters = [(cat, w/total_weight, items) for cat, w, items in clusters]

    return clusters


def generate_multi_interest_cf_candidates(
    clusters: List[Tuple[str, float, List[int]]],
    n: int = 50,
    diversity_mix: Tuple[float, float, float] = (0.7, 0.2, 0.1)
) -> List[Tuple[int, float]]:
    """Generate CF candidates from multiple interest clusters.

    Args:
        clusters: List of (category, weight, item_ids) tuples
        n: Total number of recommendations
        diversity_mix: (same_category, related_category, diverse_category) ratios

    Returns:
        List of (item_id, score) tuples, sorted by score descending
    """
    all_candidates = {}  # {item_id: score}
    user_liked_items = set()
    for _, _, items in clusters:
        user_liked_items.update(items)

    # Allocate recommendation budget proportionally
    for category, weight, cluster_items in clusters:
        budget = int(n * weight)
        if budget == 0:
            continue

        # Get specialists for this category (using Phase 1 optimization!)
        specialists = get_category_specialists(category, top_k=15)
        if len(specialists) == 0:
            continue

        # Collect items liked by specialists
        cluster_candidates = {}  # {item_id: score}

        for specialist_id, specialist_score in specialists:
            specialist_items = state.train_df[
                (state.train_df['user_id'] == specialist_id) &
                (state.train_df['rating'] >= 3)
            ]['item_id'].tolist()

            for item_id in specialist_items:
                if item_id not in user_liked_items:
                    if item_id not in cluster_candidates:
                        cluster_candidates[item_id] = 0
                    cluster_candidates[item_id] += specialist_score

        # Apply category-aware filtering (70% same, 20% related, 10% diverse)
        same_cat = []
        related_cat = []
        diverse_cat = []

        for item_id, score in cluster_candidates.items():
            product = state.products_df[state.products_df['id'] == item_id]
            if len(product) == 0:
                continue

            item_category = product.iloc[0]['articleType']

            if item_category == category:
                same_cat.append((item_id, score))
            elif category in RELATED_CATEGORIES and item_category in RELATED_CATEGORIES[category]:
                related_cat.append((item_id, score))
            else:
                diverse_cat.append((item_id, score))

        # Sort by score
        same_cat = sorted(same_cat, key=lambda x: x[1], reverse=True)
        related_cat = sorted(related_cat, key=lambda x: x[1], reverse=True)
        diverse_cat = sorted(diverse_cat, key=lambda x: x[1], reverse=True)

        # Allocate budget (70% same, 20% related, 10% diverse)
        same_n = int(budget * diversity_mix[0])
        related_n = int(budget * diversity_mix[1])
        diverse_n = int(budget * diversity_mix[2])

        # Take top items from each category
        cluster_recs = same_cat[:same_n] + related_cat[:related_n] + diverse_cat[:diverse_n]

        # Add to final candidates with cluster weight
        for item_id, score in cluster_recs:
            final_score = score * weight  # Weight by cluster importance
            if item_id not in all_candidates or final_score > all_candidates[item_id]:
                all_candidates[item_id] = final_score

    # Sort by final score and return top n
    return sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)[:n]


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

        # Load full CLIP embeddings for content-based recommendations
        try:
            embeddings_path = Path("data/embeddings/product_text_embeddings.npy")
            if embeddings_path.exists():
                state.full_embeddings = np.load(embeddings_path)
                print(f"Loaded full CLIP embeddings: {state.full_embeddings.shape}")

                # Create mapping from item_id to index in products_df
                if state.products_df is not None:
                    state.products_id_mapping = {int(row['id']): idx for idx, row in state.products_df.iterrows()}
                    print(f"Created product ID mapping for {len(state.products_id_mapping)} items")
            else:
                print("CLIP embeddings not found")
        except Exception as e:
            print(f"Error loading full embeddings: {e}")

        # Load user data and interactions from JSON
        load_users_from_json()
        load_interactions_from_json()
        load_interaction_states_from_json()

        # Build category specialist index for fast CF lookups
        if state.train_df is not None and state.products_df is not None:
            state.category_specialist_index = build_category_specialist_index(
                state.train_df,
                state.products_df
            )
            print(f"Built category specialist index for {len(state.category_specialist_index)} categories")
        else:
            print("Warning: Could not build specialist index - missing data")

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
        # Check if this is a new user (ID >= 1000)
        if request.user_id >= 1000:
            # Get user's interactions
            user_interactions = [i for i in state.interactions if i['user_id'] == request.user_id]

            if len(user_interactions) == 0:
                # Pure cold-start - redirect to popular recommendations
                return await get_coldstart_recommendations(n=request.n)

            elif len(user_interactions) < 5:
                # Partial cold-start - use content-based on liked items
                liked_items = [i['item_id'] for i in user_interactions if i['rating'] >= 3]

                if len(liked_items) == 0:
                    # No liked items yet - use popular
                    return await get_coldstart_recommendations(n=request.n)

                # Generate content-based candidates from liked items
                candidates = state.candidate_generator.generate_content_candidates(
                    user_id=request.user_id,
                    n=request.n * 2,
                    interactions_df=state.train_df
                )

                # Blend with popular items
                popular_candidates = state.candidate_generator.generate_popular_candidates(
                    user_id=request.user_id,
                    n=request.n,
                    interactions_df=state.train_df
                )

                # Merge and deduplicate
                all_candidates = candidates + popular_candidates
                seen = set()
                unique_candidates = []
                for c in all_candidates:
                    if c[0] not in seen:
                        seen.add(c[0])
                        unique_candidates.append(c)

                # Scale scores to be in 2-5 range for better display
                recommendations = [(item_id, 2.0 + (score * 3.0)) for item_id, score, _ in unique_candidates[:request.n]]

            else:
                # Enough interactions - use full pipeline with temporary embedding
                # For now, use content + popular blend (full CF requires trained user embedding)
                candidates = state.candidate_generator.generate_hybrid_candidates(
                    request.user_id,
                    state.train_df,
                    n=100,
                    adaptive_weights=True  # Phase 3: Enable adaptive weighting
                )
                candidate_items = [item_id for item_id, _, _ in candidates]

                # Re-ranking (skip for new users as reranker expects trained user IDs)
                # Scale scores to be in 2-5 range
                recommendations = [(item_id, 2.0 + (score * 3.0)) for item_id, score, _ in candidates[:request.n]]

        else:
            # Existing user from training data - use full two-stage pipeline
            # Stage 1: Candidate Generation
            candidates = state.candidate_generator.generate_hybrid_candidates(
                request.user_id,
                state.train_df,
                n=100,
                adaptive_weights=True  # Phase 3: Enable adaptive weighting
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

        # Get item popularity for interaction counts
        item_popularity = state.train_df['item_id'].value_counts() if state.train_df is not None else {}

        for item_id, score in recommendations:
            item_info = ItemInfo(
                item_id=item_id,
                score=float(score)
            )

            # Add product details if available
            if state.products_df is not None:
                product_row = state.products_df[state.products_df['id'] == item_id]
                if len(product_row) > 0:
                    row = product_row.iloc[0]
                    # Get product name - use productDisplayName
                    try:
                        prod_name = row['productDisplayName']
                        # Check if it's valid
                        if pd.isna(prod_name) or not str(prod_name).strip() or str(prod_name).startswith('Product '):
                            prod_name = f'Fashion Item #{item_id}'
                        item_info.title = str(prod_name).strip()
                        item_info.description = str(prod_name).strip()
                    except:
                        item_info.title = f'Fashion Item #{item_id}'
                        item_info.description = f'Fashion Item #{item_id}'
                else:
                    # Product not found in catalog
                    item_info.title = f'Fashion Item #{item_id}'
                    item_info.description = f'Fashion Item #{item_id}'

            # Add interaction count (training data + real-time interactions)
            training_count = int(item_popularity[item_id]) if item_id in item_popularity else 0
            realtime_count = state.item_interaction_counts.get(item_id, 0)
            item_info.interaction_count = training_count + realtime_count

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


@app.get("/recommend/coldstart", response_model=RecommendationResponse)
async def get_coldstart_recommendations(
    n: int = Query(50, ge=1, le=100, description="Number of recommendations"),
    gender: Optional[str] = Query(None, description="Filter by gender"),
    masterCategory: Optional[str] = Query(None, description="Filter by master category"),
    subCategory: Optional[str] = Query(None, description="Filter by sub category"),
    articleType: Optional[str] = Query(None, description="Filter by article type"),
    baseColour: Optional[str] = Query(None, description="Filter by color"),
    season: Optional[str] = Query(None, description="Filter by season"),
    usage: Optional[str] = Query(None, description="Filter by usage")
):
    """Get cold-start recommendations for new users (popular items).

    Args:
        n: Number of recommendations (default 50 for top 50)
        Filter parameters: gender, masterCategory, subCategory, articleType, baseColour, season, usage

    Returns:
        List of popular/trending items with interaction counts
    """
    if state.train_df is None:
        raise HTTPException(status_code=503, detail="Training data not loaded")

    try:
        # Get popular items (most interactions)
        item_popularity = state.train_df['item_id'].value_counts()

        # Filter products if filters are provided
        filtered_products = state.products_df.copy() if state.products_df is not None else None

        if filtered_products is not None:
            if gender:
                filtered_products = filtered_products[filtered_products['gender'] == gender]
            if masterCategory:
                filtered_products = filtered_products[filtered_products['masterCategory'] == masterCategory]
            if subCategory:
                filtered_products = filtered_products[filtered_products['subCategory'] == subCategory]
            if articleType:
                filtered_products = filtered_products[filtered_products['articleType'] == articleType]
            if baseColour:
                filtered_products = filtered_products[filtered_products['baseColour'] == baseColour]
            if season:
                filtered_products = filtered_products[filtered_products['season'] == season]
            if usage:
                filtered_products = filtered_products[filtered_products['usage'] == usage]

            # Filter popular items to only include filtered products
            filtered_item_ids = set(filtered_products['id'].tolist())
            popular_items = [item_id for item_id in item_popularity.index if item_id in filtered_item_ids]
        else:
            popular_items = item_popularity.head(n * 2).index.tolist()

        # Normalize popularity scores to 2-5 range
        max_popularity = item_popularity.max()
        min_popularity = item_popularity.min()

        # Build recommendations
        recommendations = []

        for item_id in popular_items[:n * 2]:  # Get extra for diversity
            # Get product info
            if state.products_df is not None:
                product_row = state.products_df[state.products_df['id'] == item_id]

                if len(product_row) > 0:
                    row = product_row.iloc[0]
                    # Normalize popularity score to 2-5 range
                    raw_score = item_popularity[item_id]
                    normalized_score = 2.0 + ((raw_score - min_popularity) / (max_popularity - min_popularity + 1)) * 3.0

                    # Use productDisplayName directly
                    # Combine training data count with real-time interactions
                    training_count = int(raw_score)
                    realtime_count = state.item_interaction_counts.get(item_id, 0)
                    total_count = training_count + realtime_count

                    item_info = ItemInfo(
                        item_id=int(item_id),
                        title=str(row['productDisplayName']),
                        description=str(row['productDisplayName']),
                        score=float(normalized_score),
                        interaction_count=total_count  # Combined count
                    )
                    recommendations.append(item_info)

                    if len(recommendations) >= n:
                        break

        return RecommendationResponse(
            user_id=-1,  # No specific user for cold-start
            recommendations=recommendations,
            retrieval_time_ms=0,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating cold-start recommendations: {str(e)}")


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


@app.get("/recommend/{user_id}/split", response_model=SplitRecommendationResponse)
async def get_split_recommendations(
    user_id: int,
    n: int = Query(10, ge=1, le=50, description="Number of recommendations per section"),
    gender: Optional[str] = Query(None, description="Filter by gender"),
    masterCategory: Optional[str] = Query(None, description="Filter by master category"),
    subCategory: Optional[str] = Query(None, description="Filter by sub category"),
    articleType: Optional[str] = Query(None, description="Filter by article type"),
    baseColour: Optional[str] = Query(None, description="Filter by color"),
    season: Optional[str] = Query(None, description="Filter by season"),
    usage: Optional[str] = Query(None, description="Filter by usage")
):
    """Get recommendations split into CF and content-based sections.

    Args:
        user_id: User ID
        n: Number of recommendations per section
        Filter parameters: gender, masterCategory, subCategory, articleType, baseColour, season, usage

    Returns:
        Split recommendations with CF and content-based sections
    """
    start_time = time.time()

    if state.candidate_generator is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        cf_recommendations = []
        content_recommendations = []

        # Helper function to apply filters
        def apply_filters(item_ids):
            if state.products_df is None:
                return item_ids

            filtered_df = state.products_df[state.products_df['id'].isin(item_ids)]

            if gender:
                filtered_df = filtered_df[filtered_df['gender'] == gender]
            if masterCategory:
                filtered_df = filtered_df[filtered_df['masterCategory'] == masterCategory]
            if subCategory:
                filtered_df = filtered_df[filtered_df['subCategory'] == subCategory]
            if articleType:
                filtered_df = filtered_df[filtered_df['articleType'] == articleType]
            if baseColour:
                filtered_df = filtered_df[filtered_df['baseColour'] == baseColour]
            if season:
                filtered_df = filtered_df[filtered_df['season'] == season]
            if usage:
                filtered_df = filtered_df[filtered_df['usage'] == usage]

            return filtered_df['id'].tolist()

        # Check if this is a new user or trained user
        if user_id >= 1000:
            # New user - use content-based only
            user_interactions = [i for i in state.interactions if i['user_id'] == user_id]

            # New users: Use embedding-based user similarity for CF
            print(f"New user {user_id} - Using embedding-based user similarity for CF")

            if len(user_interactions) >= 5 and state.full_embeddings is not None:
                try:
                    # Step 1: Compute user embedding from liked items
                    liked_item_ids = [i['item_id'] for i in user_interactions if i['rating'] >= 3]

                    if len(liked_item_ids) > 0:
                        # Get embeddings for liked items
                        liked_embeddings = []
                        for item_id in liked_item_ids[:20]:  # Use up to 20 recent likes
                            if item_id < len(state.full_embeddings):
                                liked_embeddings.append(state.full_embeddings[item_id])

                        if len(liked_embeddings) > 0:
                            # Create user embedding (average of liked item embeddings)
                            user_embedding = np.mean(liked_embeddings, axis=0)
                            user_embedding = user_embedding / (np.linalg.norm(user_embedding) + 1e-8)

                            print(f"Created user embedding from {len(liked_embeddings)} liked items")

                            # Step 2: MULTI-INTEREST CLUSTERING (Phase 2)
                            # Detect multiple interest clusters (e.g., watches + backpacks + shoes)
                            print(f"Detecting interest clusters from {len(user_interactions)} interactions...")
                            clusters = detect_user_interest_clusters(user_interactions, state.products_df)

                            if len(clusters) > 0:
                                # Multi-interest user: recommend from ALL clusters proportionally
                                print(f"Found {len(clusters)} interest clusters:")
                                for cat, weight, items in clusters:
                                    print(f"  - {cat}: {weight*100:.1f}% ({len(items)} items)")

                                # Generate CF candidates using multi-interest approach
                                cf_candidates = generate_multi_interest_cf_candidates(clusters, n=n)
                                print(f"Generated {len(cf_candidates)} multi-interest CF recommendations")
                            else:
                                # Fallback: no significant clusters found
                                print("No significant interest clusters detected, skipping CF recommendations")
                                cf_candidates = []

                except Exception as e:
                    print(f"Error generating user-similarity CF: {e}")
                    import traceback
                    traceback.print_exc()
                    cf_recommendations = []
            else:
                # Not enough interactions - show popular
                print(f"Not enough interactions ({len(user_interactions)}) for user-based CF")
                cf_recommendations = []

            # Content-based recommendations (using CandidateGenerator with rebuilt FAISS)
            if len(user_interactions) > 0:
                try:
                    print(f"Attempting content-based for user {user_id} with {len(user_interactions)} interactions")

                    # Create temporary dataframe with new user's interactions
                    temp_interactions = pd.DataFrame(user_interactions)

                    # Combine with training data for content-based
                    combined_df = pd.concat([state.train_df, temp_interactions], ignore_index=True)

                    # Use CandidateGenerator's method (now has all 44K items in FAISS!)
                    candidates = state.candidate_generator.generate_content_candidates(
                        user_id, n=n * 2, interactions_df=combined_df
                    )
                    candidate_ids = [item_id for item_id, _, _ in candidates]
                    print(f"Generated {len(candidate_ids)} content candidates from FAISS")

                    # Apply filters
                    filtered_ids = apply_filters(candidate_ids)
                    print(f"After filters: {len(filtered_ids)} candidates")

                    for item_id in filtered_ids[:n]:
                        score = next((s for i, s, _ in candidates if i == item_id), 0.5)
                        content_recommendations.append((item_id, 2.0 + (score * 3.0)))

                    print(f"Final content recommendations: {len(content_recommendations)} items")

                    # FALLBACK: If not enough content-based results, use category-based matching
                    if len(content_recommendations) < n and state.products_df is not None:
                        print(f"Using category-based fallback (have {len(content_recommendations)}, need {n})")

                        # Get categories of liked items
                        liked_item_ids = [i['item_id'] for i in user_interactions if i['rating'] >= 3]
                        liked_products = state.products_df[state.products_df['id'].isin(liked_item_ids)]

                        if len(liked_products) > 0:
                            # Find most common categories
                            categories = liked_products[['gender', 'masterCategory', 'subCategory', 'articleType']].mode()

                            if len(categories) > 0:
                                # Find similar items by category
                                similar_items = state.products_df.copy()

                                # Match on article type (most specific)
                                if 'articleType' in categories.columns and not pd.isna(categories['articleType'].iloc[0]):
                                    article_type = categories['articleType'].iloc[0]
                                    similar_items = similar_items[similar_items['articleType'] == article_type]
                                    print(f"Matching articleType: {article_type}, found {len(similar_items)} items")

                                # Exclude already liked items
                                similar_items = similar_items[~similar_items['id'].isin(liked_item_ids)]
                                similar_items = similar_items[~similar_items['id'].isin([i for i, _, _ in content_recommendations])]

                                # Apply current filters
                                filtered_similar = apply_filters(similar_items['id'].tolist())

                                # Add to recommendations
                                needed = n - len(content_recommendations)
                                for item_id in filtered_similar[:needed]:
                                    content_recommendations.append((item_id, 3.5))  # Medium-high score

                                print(f"Added {min(needed, len(filtered_similar))} category-based recommendations")

                except Exception as e:
                    print(f"Error generating content recommendations for user {user_id}: {e}")
                    import traceback
                    traceback.print_exc()

        else:
            # Trained user - use full pipeline
            # CF recommendations
            try:
                cf_candidates = state.candidate_generator.generate_cf_candidates(
                    user_id, state.train_df, n=n * 2
                )
                candidate_ids = [item_id for item_id, _, _ in cf_candidates]
                filtered_ids = apply_filters(candidate_ids)

                # Re-rank CF candidates
                if state.reranker and len(filtered_ids) > 0:
                    ranked = state.reranker.rerank(user_id, filtered_ids)
                    cf_recommendations = ranked[:n]
                else:
                    for item_id in filtered_ids[:n]:
                        score = next((s for i, s, _ in cf_candidates if i == item_id), 3.0)
                        cf_recommendations.append((item_id, score))
            except Exception as e:
                print(f"Error generating CF recommendations: {e}")

            # Content-based recommendations
            try:
                content_candidates = state.candidate_generator.generate_content_candidates(
                    user_id, n=n * 2, interactions_df=state.train_df
                )
                candidate_ids = [item_id for item_id, _, _ in content_candidates]
                filtered_ids = apply_filters(candidate_ids)

                for item_id in filtered_ids[:n]:
                    score = next((s for i, s, _ in content_candidates if i == item_id), 0.5)
                    content_recommendations.append((item_id, 2.0 + (score * 3.0)))
            except Exception as e:
                print(f"Error generating content recommendations: {e}")

        # Enrich with product information
        def enrich_items(items):
            enriched = []
            # Get item popularity for interaction counts
            item_popularity = state.train_df['item_id'].value_counts() if state.train_df is not None else {}

            for item_id, score in items:
                item_info = ItemInfo(item_id=item_id, score=float(score))

                if state.products_df is not None:
                    product_row = state.products_df[state.products_df['id'] == item_id]
                    if len(product_row) > 0:
                        row = product_row.iloc[0]
                        prod_name = row.get('productDisplayName', f'Fashion Item #{item_id}')
                        item_info.title = str(prod_name)
                        item_info.description = str(prod_name)

                # Add interaction count (training data + real-time interactions)
                training_count = int(item_popularity[item_id]) if item_id in item_popularity else 0
                realtime_count = state.item_interaction_counts.get(item_id, 0)
                item_info.interaction_count = training_count + realtime_count

                enriched.append(item_info)
            return enriched

        cf_enriched = enrich_items(cf_recommendations)
        content_enriched = enrich_items(content_recommendations)

        retrieval_time = (time.time() - start_time) * 1000

        return SplitRecommendationResponse(
            user_id=user_id,
            cf_recommendations=cf_enriched,
            content_recommendations=content_enriched,
            retrieval_time_ms=retrieval_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        print(f"Error generating split recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.get("/items/all", response_model=PaginatedProductsResponse)
async def get_all_products(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    gender: Optional[str] = Query(None, description="Filter by gender"),
    masterCategory: Optional[str] = Query(None, description="Filter by master category"),
    subCategory: Optional[str] = Query(None, description="Filter by sub category"),
    articleType: Optional[str] = Query(None, description="Filter by article type"),
    baseColour: Optional[str] = Query(None, description="Filter by color"),
    season: Optional[str] = Query(None, description="Filter by season"),
    usage: Optional[str] = Query(None, description="Filter by usage")
):
    """Get all products with pagination and filters.

    Args:
        page: Page number (1-indexed)
        limit: Items per page
        Filter parameters: gender, masterCategory, subCategory, articleType, baseColour, season, usage

    Returns:
        Paginated list of products with total count
    """
    if state.products_df is None:
        raise HTTPException(status_code=503, detail="Product data not loaded")

    try:
        # Start with all products
        filtered_df = state.products_df.copy()

        # Apply filters
        if gender:
            filtered_df = filtered_df[filtered_df['gender'] == gender]
        if masterCategory:
            filtered_df = filtered_df[filtered_df['masterCategory'] == masterCategory]
        if subCategory:
            filtered_df = filtered_df[filtered_df['subCategory'] == subCategory]
        if articleType:
            filtered_df = filtered_df[filtered_df['articleType'] == articleType]
        if baseColour:
            filtered_df = filtered_df[filtered_df['baseColour'] == baseColour]
        if season:
            filtered_df = filtered_df[filtered_df['season'] == season]
        if usage:
            filtered_df = filtered_df[filtered_df['usage'] == usage]

        # Calculate pagination
        total = len(filtered_df)
        total_pages = (total + limit - 1) // limit  # Ceiling division
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit

        # Get page of results
        page_df = filtered_df.iloc[start_idx:end_idx]

        # Convert to list of dicts
        products = []
        for _, row in page_df.iterrows():
            product = {
                'item_id': int(row['id']),
                'title': str(row.get('productDisplayName', f"Fashion Item #{row['id']}")),
                'gender': str(row.get('gender', '')),
                'masterCategory': str(row.get('masterCategory', '')),
                'subCategory': str(row.get('subCategory', '')),
                'articleType': str(row.get('articleType', '')),
                'baseColour': str(row.get('baseColour', '')),
                'season': str(row.get('season', '')),
                'usage': str(row.get('usage', '')),
                'year': float(row.get('year', 0)) if pd.notna(row.get('year')) else None
            }
            products.append(product)

        return PaginatedProductsResponse(
            products=products,
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching products: {str(e)}")



@app.get("/items/filter-values", response_model=FilterValuesResponse)
async def get_filter_values():
    """Get all available filter values from the dataset.

    Returns:
        Lists of unique values for each filter category
    """
    if state.products_df is None:
        raise HTTPException(status_code=503, detail="Product data not loaded")

    try:
        return FilterValuesResponse(
            genders=sorted([str(x) for x in state.products_df['gender'].dropna().unique() if str(x) != 'nan']),
            masterCategories=sorted([str(x) for x in state.products_df['masterCategory'].dropna().unique() if str(x) != 'nan']),
            subCategories=sorted([str(x) for x in state.products_df['subCategory'].dropna().unique() if str(x) != 'nan']),
            articleTypes=sorted([str(x) for x in state.products_df['articleType'].dropna().unique() if str(x) != 'nan']),
            baseColours=sorted([str(x) for x in state.products_df['baseColour'].dropna().unique() if str(x) != 'nan']),
            seasons=sorted([str(x) for x in state.products_df['season'].dropna().unique() if str(x) != 'nan']),
            usages=sorted([str(x) for x in state.products_df['usage'].dropna().unique() if str(x) != 'nan'])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching filter values: {str(e)}")


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
        "title": product_dict.get('productDisplayName', f'Fashion Item #{item_id}') if product_dict.get('productDisplayName') else f'Fashion Item #{item_id}',
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
                    row = product.iloc[0]
                    # Get product name
                    prod_name = row.get('productDisplayName', '')
                    item_info.title = str(prod_name) if prod_name and not pd.isna(prod_name) else f'Fashion Item #{similar_id}'
                    item_info.description = str(row.get('description', ''))[:200]

            enriched_items.append(item_info)

        return enriched_items

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar items: {str(e)}")


# ============================================================================
# New Endpoints: User Management and Interaction Tracking
# ============================================================================

@app.post("/users/register", response_model=UserProfile)
async def register_user(registration: UserRegistration):
    """Register a new user and assign a unique user ID.

    Args:
        registration: User registration data (username, optional email)

    Returns:
        User profile with assigned user_id
    """
    # Generate new user ID
    user_id = state.next_user_id
    state.next_user_id += 1

    # Check if username already exists
    existing_user = next((u for u in state.users.values() if u['username'] == registration.username), None)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken. Please choose a different username or login.")

    # Create user profile
    user_data = {
        'user_id': user_id,
        'username': registration.username,
        'password': registration.password,  # In production, hash this!
        'email': registration.email,
        'created_at': datetime.now().isoformat(),
        'total_interactions': 0
    }

    # Store in hashmap
    state.users[user_id] = user_data

    # Save to JSON file
    save_users_to_json()

    print(f"Registered new user: {registration.username} (ID: {user_id})")

    return UserProfile(
        user_id=user_id,
        username=registration.username,
        created_at=user_data['created_at'],
        total_interactions=0
    )


@app.post("/users/login", response_model=LoginResponse)
async def login_user(login: UserLogin):
    """Login existing user with username and password.

    Args:
        login: Login credentials (username, password)

    Returns:
        Login response with user_id if successful
    """
    # Find user by username
    user = next((u for u in state.users.values() if u['username'] == login.username), None)

    if not user:
        return LoginResponse(
            success=False,
            message="Username not found. Please register first."
        )

    # Check password
    if user['password'] != login.password:
        return LoginResponse(
            success=False,
            message="Incorrect password. Please try again."
        )

    # Successful login
    print(f"User logged in: {login.username} (ID: {user['user_id']})")

    return LoginResponse(
        success=True,
        user_id=user['user_id'],
        username=user['username'],
        message="Login successful!"
    )


@app.get("/users/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: int):
    """Get user profile by ID.

    Args:
        user_id: User ID

    Returns:
        User profile information
    """
    if user_id not in state.users:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = state.users[user_id]

    # Count user interactions
    interaction_count = sum(1 for i in state.interactions if i['user_id'] == user_id)

    return UserProfile(
        user_id=user_id,
        username=user_data['username'],
        created_at=user_data['created_at'],
        total_interactions=interaction_count
    )


@app.post("/interactions", response_model=InteractionResponse)
async def track_interaction(interaction: InteractionRequest):
    """Track a user interaction with an item (Facebook-style toggle behavior).

    Args:
        interaction: Interaction details (user_id, item_id, type)

    Returns:
        Interaction confirmation with timestamp and action taken

    Behavior:
    - Click same interaction type → removes it (toggle off)
    - Click different interaction type → replaces previous one
    - Only one active interaction per user-item pair
    """
    # Validate user exists (for new users)
    if interaction.user_id >= 1000 and interaction.user_id not in state.users:
        raise HTTPException(status_code=404, detail="User not found. Please register first.")

    # Initialize user's interaction states if not exists
    if interaction.user_id not in state.interaction_states:
        state.interaction_states[interaction.user_id] = {}

    # Get current interaction state for this item
    current_interaction = state.interaction_states[interaction.user_id].get(interaction.item_id)

    timestamp = datetime.now().isoformat()
    action = ""
    interaction_id = None
    message = ""

    # Check if clicking the same interaction (toggle off)
    if current_interaction == interaction.interaction_type:
        # Remove the interaction
        del state.interaction_states[interaction.user_id][interaction.item_id]
        action = "removed"
        message = f"Interaction '{interaction.interaction_type}' removed"
        current_interaction = None

        # Update user interaction count
        if interaction.user_id in state.users:
            state.users[interaction.user_id]['total_interactions'] = max(
                0,
                state.users[interaction.user_id]['total_interactions'] - 1
            )

        # Update item-level interaction count (decrease)
        if interaction.item_id in state.item_interaction_counts:
            state.item_interaction_counts[interaction.item_id] = max(
                0,
                state.item_interaction_counts[interaction.item_id] - 1
            )

    else:
        # Add or update interaction
        # Map interaction type to rating
        rating = INTERACTION_RATINGS.get(interaction.interaction_type, 1)

        # Create interaction record
        interaction_id = state.next_interaction_id
        state.next_interaction_id += 1

        interaction_data = {
            'interaction_id': interaction_id,
            'user_id': interaction.user_id,
            'item_id': interaction.item_id,
            'interaction_type': interaction.interaction_type,
            'rating': rating,
            'timestamp': timestamp
        }

        # Store interaction
        state.interactions.append(interaction_data)

        # Update interaction state
        state.interaction_states[interaction.user_id][interaction.item_id] = interaction.interaction_type

        if current_interaction:
            action = "updated"
            message = f"Interaction changed from '{current_interaction}' to '{interaction.interaction_type}'"
            # Switching interaction type - item count stays the same
        else:
            action = "added"
            message = f"Interaction '{interaction.interaction_type}' recorded successfully"

            # Update user interaction count only for new interactions
            if interaction.user_id in state.users:
                state.users[interaction.user_id]['total_interactions'] += 1

            # Update item-level interaction count (increase for new interactions only)
            if interaction.item_id not in state.item_interaction_counts:
                state.item_interaction_counts[interaction.item_id] = 0
            state.item_interaction_counts[interaction.item_id] += 1

        current_interaction = interaction.interaction_type

    # Save all changes
    if interaction.user_id in state.users:
        save_users_to_json()

    save_interactions_to_json()
    save_interaction_states_to_json()

    print(f"[{action.upper()}] {message} for user {interaction.user_id}, item {interaction.item_id}")

    return InteractionResponse(
        success=True,
        interaction_id=interaction_id,
        timestamp=timestamp,
        message=message,
        action=action,
        current_interaction=current_interaction
    )


@app.get("/users/{user_id}/interactions", response_model=List[UserInteraction])
async def get_user_interactions(user_id: int):
    """Get all interactions for a specific user.

    Args:
        user_id: User ID

    Returns:
        List of user interactions
    """
    # Filter interactions for this user
    user_interactions = [
        UserInteraction(
            interaction_id=i['interaction_id'],
            item_id=i['item_id'],
            interaction_type=i['interaction_type'],
            rating=i['rating'],
            timestamp=i['timestamp']
        )
        for i in state.interactions
        if i['user_id'] == user_id
    ]

    return user_interactions


@app.get("/users/{user_id}/interaction-states", response_model=InteractionStateResponse)
async def get_user_interaction_states(user_id: int):
    """Get current interaction states for a user (what they've interacted with).

    Args:
        user_id: User ID

    Returns:
        Dictionary of item_id -> interaction_type
    """
    interaction_states = state.interaction_states.get(user_id, {})

    return InteractionStateResponse(
        user_id=user_id,
        interaction_states=interaction_states
    )




