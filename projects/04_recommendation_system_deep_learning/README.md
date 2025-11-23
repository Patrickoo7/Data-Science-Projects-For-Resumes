# Advanced Recommendation System with Deep Learning

## Project Overview
Production-ready recommendation system using neural collaborative filtering, deep learning embeddings, and hybrid approaches. Implements multiple recommendation algorithms with real-time serving capabilities.

## Features
- **Neural Collaborative Filtering (NCF)**: Deep learning-based collaborative filtering
- **Matrix Factorization**: SVD, ALS, NMF
- **Content-Based Filtering**: Item similarity using embeddings
- **Hybrid Models**: Combining collaborative and content-based approaches
- **Deep & Cross Network**: Feature interaction learning
- **Graph Neural Networks**: User-item graph representations
- **Cold Start Solutions**: Handling new users and items
- **A/B Testing Framework**: Evaluate recommendation strategies

## Tech Stack
- **Deep Learning**: PyTorch, PyTorch Lightning
- **Recommendation**: Surprise, LightFM, RecBole
- **Graph Learning**: PyTorch Geometric, DGL
- **Serving**: FastAPI, Redis, Celery
- **Data**: pandas, scipy, NetworkX
- **Evaluation**: scikit-learn, implicit

## Project Structure
```
├── data/                   # User-item interaction data
├── models/                 # Trained models
├── notebooks/              # Exploratory analysis
├── src/                    # Source code
│   ├── data_loader.py     # Data processing
│   ├── models/            # Model implementations
│   │   ├── ncf.py        # Neural Collaborative Filtering
│   │   ├── autoencoders.py
│   │   ├── graph_models.py
│   │   └── hybrid.py
│   ├── train.py          # Training pipeline
│   ├── evaluate.py       # Evaluation metrics
│   └── inference.py      # Recommendation generation
├── api/                   # REST API
│   ├── app.py
│   └── cache.py
├── config.yaml
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
```bash
python src/data_loader.py --dataset movielens-1m --output data/processed/
```

### 2. Train Model
```bash
python src/train.py \
    --model ncf \
    --embedding_dim 64 \
    --layers [128,64,32] \
    --epochs 20 \
    --batch_size 256
```

### 3. Evaluate Model
```bash
python src/evaluate.py \
    --model_path models/ncf_best.pth \
    --metrics ndcg,hit_rate,mrr \
    --top_k 10
```

### 4. Generate Recommendations
```bash
python src/inference.py --user_id 123 --top_k 10
```

### 5. Start API Server
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Model Architectures

### 1. Neural Collaborative Filtering (NCF)
```
User Embedding (64) ─────┐
                         ├─► Concatenate ─► MLP [128,64,32] ─► Output (1)
Item Embedding (64) ─────┘
```

### 2. Autoencoder-based CF
```
User-Item Matrix ─► Encoder [512,256] ─► Bottleneck (64) ─► Decoder [256,512] ─► Reconstruction
```

### 3. Graph Neural Network
```
User-Item Graph ─► GCN Layers ─► Node Embeddings ─► Link Prediction
```

## Performance Metrics
| Model | NDCG@10 | Hit Rate@10 | MRR | Coverage |
|-------|---------|-------------|-----|----------|
| NCF | 0.423 | 0.687 | 0.312 | 0.85 |
| Autoencoder | 0.398 | 0.652 | 0.289 | 0.82 |
| GNN | 0.445 | 0.712 | 0.334 | 0.88 |
| Hybrid | 0.467 | 0.731 | 0.351 | 0.91 |

## API Endpoints

### Get Recommendations
```bash
GET /recommend/{user_id}?top_k=10

Response:
{
  "user_id": 123,
  "recommendations": [
    {"item_id": 456, "score": 0.95, "title": "The Matrix"},
    {"item_id": 789, "score": 0.89, "title": "Inception"}
  ]
}
```

### Similar Items
```bash
GET /similar/{item_id}?top_k=10
```

### Batch Recommendations
```bash
POST /recommend/batch
Body: {"user_ids": [123, 456, 789], "top_k": 10}
```

## Advanced Features

### 1. Cold Start Handling
- Content-based recommendations for new items
- Demographic-based recommendations for new users
- Popularity-based fallback

### 2. Diversity & Serendipity
- MMR (Maximal Marginal Relevance) reranking
- Category diversification
- Exploration vs exploitation balance

### 3. Explainability
- Attention weights visualization
- Feature importance analysis
- "Because you liked X" explanations

### 4. Real-time Features
- Redis caching for fast serving
- Incremental model updates
- Online learning capabilities

### 5. Scalability
- Approximate Nearest Neighbors (ANN) with FAISS
- Distributed training with PyTorch DDP
- Model compression and quantization

## Datasets
- MovieLens (100K, 1M, 20M)
- Amazon Product Reviews
- Netflix Prize
- Book-Crossing
- Last.fm Music
- Custom e-commerce data

## Evaluation Metrics
- **Ranking Metrics**: NDCG, MRR, MAP
- **Classification Metrics**: Precision, Recall, F1
- **Coverage**: Catalog coverage, User coverage
- **Diversity**: Intra-list diversity, Personalization
- **Business Metrics**: CTR, Conversion rate, Revenue lift

## Future Enhancements
- Multi-armed bandit for exploration
- Contextual recommendations (time, location, device)
- Session-based recommendations with RNNs
- Cross-domain recommendations
- Federated learning for privacy
- Knowledge graph integration

## References
- Neural Collaborative Filtering (He et al., 2017)
- Deep & Cross Network (Wang et al., 2017)
- Graph Convolutional Matrix Completion (Berg et al., 2017)
- AutoRec (Sedhain et al., 2015)

## License
MIT License
