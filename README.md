# Hybrid Recommender System 🚧

> **Status**: Work in Progress | Active Development

A production-ready two-stage recommendation system combining collaborative filtering (ALS) with multimodal embeddings (CLIP) for personalized content ranking.

## Architecture

### Two-Stage Recommendation Pipeline

1. **Candidate Generation (Retrieval)**
   - Fast retrieval of top-N candidates from large item pool
   - Uses collaborative filtering and vector similarity search
   - Optimized for recall

2. **Re-Ranking**
   - Precise ranking of candidate items
   - Combines multimodal features with user preferences
   - Optimized for precision and relevance

## Features

- **Multimodal Embeddings**: CLIP-based text and image embeddings
- **Collaborative Filtering**: Matrix Factorization (ALS) + Neural CF
- **Experiment Tracking**: MLflow integration
- **Model Serving**: FastAPI with Docker deployment
- **Evaluation Metrics**: Precision@K, Recall@K, NDCG

## Project Structure

```
MultiModelRC/
├── data/                       # Data storage
│   ├── raw/                   # Raw dataset files
│   ├── processed/             # Processed interaction data
│   └── embeddings/            # Pre-computed embeddings
├── models/                     # Saved models
│   ├── collaborative_filtering/
│   ├── candidate_generation/
│   └── reranking/
├── src/                        # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── embeddings/            # Embedding extraction
│   ├── models/                # Model implementations
│   ├── evaluation/            # Evaluation metrics
│   └── api/                   # FastAPI service
├── notebooks/                  # Jupyter notebooks
├── experiments/                # Experiment configs
├── mlruns/                     # MLflow tracking
├── docker/                     # Docker configurations
├── tests/                      # Unit tests
└── config/                     # Configuration files
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
python src/data/download_data.py
```

## Usage

### Training Models

```bash
# Train collaborative filtering models
python src/models/train_cf.py

# Train multimodal embedding model
python src/embeddings/extract_embeddings.py

# Train candidate generation
python src/models/train_candidate_generation.py

# Train re-ranker
python src/models/train_reranking.py
```

### Run Experiments with MLflow

```bash
mlflow ui
# Visit http://localhost:5000
```

### Evaluation

```bash
python src/evaluation/evaluate.py --model candidate_generation
python src/evaluation/evaluate.py --model reranking
```

### Start API Server

```bash
uvicorn src.api.main:app --reload
# Visit http://localhost:8000/docs
```

### Docker Deployment

```bash
docker-compose up --build
```

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG**: Normalized Discounted Cumulative Gain for ranking quality

## Components

### 1. Data Pipeline
- Image preprocessing and augmentation
- Text cleaning and tokenization
- Interaction matrix construction

### 2. Embedding Models
- CLIP for aligned multimodal embeddings
- Pre-computed embeddings for fast inference

### 3. Collaborative Filtering
- Matrix Factorization (ALS)
- Neural Collaborative Filtering

### 4. Candidate Generation
- Vector similarity search (FAISS/Annoy)
- Hybrid retrieval (CF + content-based)

### 5. Re-Ranking
- Multi-feature neural ranker
- User context integration

## Performance

Target metrics for small prototype:
- Candidate Generation: <50ms for 100 candidates
- Re-Ranking: <100ms for top 20 items
- Precision@10: >0.3
- NDCG@10: >0.5

---

## 🚧 Development Roadmap

### ✅ Completed
- [x] Project structure and environment setup
- [x] Data pipeline (download, preprocessing)
- [x] CLIP text embeddings extraction
- [x] Matrix Factorization (ALS) implementation
- [x] Candidate generation (collaborative filtering)
- [x] Basic evaluation metrics (Precision, Recall, NDCG)

### 🔄 In Progress
- [ ] Neural Collaborative Filtering model
- [ ] Re-ranking neural network
- [ ] FastAPI deployment
- [ ] MLflow experiment tracking
- [ ] Visualization dashboard

### 📋 Planned
- [ ] CLIP image embeddings integration
- [ ] Advanced re-ranking features
- [ ] A/B testing framework
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)
- [ ] Performance optimization
- [ ] Comprehensive unit tests

---

## 🤝 Contributing

This is an active research project. Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or collaboration opportunities, please open an issue.

## License

MIT License
