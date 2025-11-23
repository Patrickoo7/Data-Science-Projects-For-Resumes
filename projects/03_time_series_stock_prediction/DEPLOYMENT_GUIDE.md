# 🚀 StockPredict Deployment Guide

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Database Setup](#database-setup)
6. [Model Management](#model-management)
7. [Monitoring Setup](#monitoring-setup)
8. [Troubleshooting](#troubleshooting)

## Local Development

### Backend Setup

```bash
# Navigate to project
cd projects/03_time_series_stock_prediction

# Create virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp ../.env.example ../.env
# Edit .env with your local configuration

# Run database migrations (if using local PostgreSQL)
alembic upgrade head

# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access API: http://localhost:8000
Swagger Docs: http://localhost:8000/docs

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
pip install -r requirements.txt

# Start Streamlit dashboard
streamlit run app.py
```

Access Dashboard: http://localhost:8501

## Docker Deployment

### Quick Start (All Services)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Individual Services

```bash
# Start only backend + database
docker-compose up -d db redis backend

# Start only frontend
docker-compose up -d frontend

# Rebuild specific service
docker-compose up -d --build backend
```

### Service URLs
- **Backend API**: http://localhost:8001
- **Frontend Dashboard**: http://localhost:8501
- **PostgreSQL**: localhost:5433
- **Redis**: localhost:6380
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3002

## Production Deployment

### Prerequisites

1. **Server Requirements**
   - Ubuntu 20.04+ or similar Linux distribution
   - 16GB+ RAM (for model inference)
   - 100GB+ disk space
   - Docker & Docker Compose installed
   - GPU support (optional but recommended)

2. **Domain & SSL**
   - Domain name pointed to your server
   - SSL certificate (Let's Encrypt recommended)

### Step 1: Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose -y

# Add user to docker group
sudo usermod -aG docker $USER

# Install NVIDIA Docker (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Step 2: Clone Repository

```bash
# Clone repository
git clone https://github.com/yourusername/stockpredict.git
cd stockpredict/projects/03_time_series_stock_prediction
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

Important production settings:
```env
DEBUG=False
ENVIRONMENT=production
SECRET_KEY=<generate-secure-random-key-using-secrets.token_urlsafe(32)>
DATABASE_URL=postgresql://stockpredict:STRONG_PASSWORD@db:5432/stockpredict
BACKEND_CORS_ORIGINS=["https://yourdomain.com"]
DEVICE=cuda  # If GPU available, otherwise cpu
```

### Step 4: Deploy with Docker Compose

```bash
# Pull latest images
docker-compose pull

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

### Step 5: Initialize Database

```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Create admin user (optional)
docker-compose exec backend python scripts/create_admin.py
```

### Step 6: Load Pre-trained Models

```bash
# Create models directory
mkdir -p models

# Download pre-trained LSTM model (example)
# You can train your own model or use pre-trained weights
docker-compose exec backend python scripts/download_models.py

# Verify model loading
docker-compose exec backend python -c "from services.forecasting_service import forecaster; forecaster.load_model()"
```

### Step 7: Configure Nginx (Reverse Proxy)

```nginx
# /etc/nginx/sites-available/stockpredict
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/stockpredict /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 8: SSL with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

## Environment Configuration

### Critical Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SECRET_KEY` | JWT secret (MUST be changed) | Random 32+ char string |
| `DATABASE_URL` | PostgreSQL connection | `postgresql://user:pass@host:port/db` |
| `REDIS_URL` | Redis connection | `redis://host:port/0` |
| `MODEL_TYPE` | Model architecture | `lstm`, `gru`, `transformer` |
| `DEVICE` | Compute device | `cuda` or `cpu` |
| `ALLOWED_TICKERS` | Supported stock symbols | `["AAPL","GOOGL","MSFT"]` |

### Security Best Practices

1. **Generate Secure SECRET_KEY**
```python
import secrets
print(secrets.token_urlsafe(32))
```

2. **Use Strong Database Passwords**
```bash
openssl rand -base64 32
```

3. **Restrict CORS Origins**
```env
BACKEND_CORS_ORIGINS=["https://yourdomain.com"]
```

4. **Enable API Key Authentication**
```env
REQUIRE_API_KEY=True
API_KEY_EXPIRE_DAYS=365
```

## Database Setup

### PostgreSQL Connection

```bash
# Connect to database
docker-compose exec db psql -U stockpredict -d stockpredict

# Backup database
docker-compose exec db pg_dump -U stockpredict stockpredict > backup.sql

# Restore database
docker-compose exec -T db psql -U stockpredict stockpredict < backup.sql
```

### Migrations

```bash
# Create new migration
docker-compose exec backend alembic revision -m "description"

# Apply migrations
docker-compose exec backend alembic upgrade head

# Rollback migration
docker-compose exec backend alembic downgrade -1

# View migration history
docker-compose exec backend alembic history
```

### Database Optimization

```sql
-- Create indexes for better query performance
CREATE INDEX idx_predictions_ticker ON predictions(ticker);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
CREATE INDEX idx_stock_data_ticker_date ON stock_data(ticker, date);

-- Vacuum and analyze
VACUUM ANALYZE;
```

## Model Management

### Training Custom Models

```bash
# Train new LSTM model
docker-compose exec backend python scripts/train_model.py \
  --model-type lstm \
  --ticker AAPL \
  --epochs 100 \
  --batch-size 32

# Evaluate model
docker-compose exec backend python scripts/evaluate_model.py \
  --model-path models/lstm_stock_predictor.pth \
  --ticker AAPL
```

### Model Versioning

```bash
# Save model with version
docker-compose exec backend python -c "
from services.forecasting_service import forecaster
forecaster.save_model('models/lstm_v1.0.0.pth')
"

# Load specific model version
# Update .env: MODEL_PATH=./models/lstm_v1.0.0.pth
docker-compose restart backend
```

### Model Performance Monitoring

```python
# Monitor prediction accuracy
from services.forecasting_service import forecaster

# Get model metrics
metrics = forecaster.get_model_info()
print(f"MAE: {metrics['mae']}")
print(f"RMSE: {metrics['rmse']}")
print(f"MAPE: {metrics['mape']}")
```

## Monitoring Setup

### Prometheus

Access: http://localhost:9091

Key metrics:
- `api_requests_total` - Total API requests
- `api_latency_seconds` - API latency
- `prediction_count` - Prediction count
- `model_inference_time` - Model inference time
- `cache_hit_rate` - Cache hit rate
- `stock_data_fetch_errors` - Data fetch errors

### Grafana

Access: http://localhost:3002
Default credentials: admin/admin

Import dashboards:
1. FastAPI metrics dashboard
2. PostgreSQL dashboard
3. Redis dashboard
4. Custom StockPredict metrics

### Application Logging

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend

# View last 100 lines
docker-compose logs --tail=100 backend

# Export logs
docker-compose logs > logs.txt
```

### Custom Metrics

```python
# In your code, track custom metrics
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions made', ['ticker'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@prediction_latency.time()
async def predict(ticker: str):
    result = await forecaster.predict(ticker)
    prediction_counter.labels(ticker=ticker).inc()
    return result
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed

```bash
# Check database status
docker-compose ps db

# View database logs
docker-compose logs db

# Restart database
docker-compose restart db

# Check connectivity
docker-compose exec backend python -c "from core.database import engine; print(engine.connect())"
```

#### 2. Model Loading Error

```bash
# Check available memory
free -h

# Check GPU availability
docker-compose exec backend python -c "import torch; print(torch.cuda.is_available())"

# Check model files
docker-compose exec backend ls -lh /app/models

# Use CPU if GPU unavailable
# In .env: DEVICE=cpu
```

#### 3. Stock Data Fetch Error

```bash
# Test yfinance connection
docker-compose exec backend python -c "
import yfinance as yf
ticker = yf.Ticker('AAPL')
print(ticker.info)
"

# Check for rate limiting
# Add delays between requests in .env:
# DATA_FETCH_DELAY_SECONDS=1
```

#### 4. Frontend Not Loading

```bash
# Check frontend status
docker-compose ps frontend

# Rebuild frontend
docker-compose up -d --build frontend

# Check Streamlit logs
docker-compose logs frontend

# Test connection
curl http://localhost:8501/_stcore/health
```

#### 5. High Memory Usage

```bash
# Check resource usage
docker stats

# Reduce model batch size in .env
BATCH_SIZE=16

# Limit workers
# In Dockerfile: --workers 2

# Use model quantization (optional)
# In .env: USE_QUANTIZATION=True
```

#### 6. Prediction Accuracy Issues

```bash
# Retrain model with more data
docker-compose exec backend python scripts/train_model.py \
  --ticker AAPL \
  --start-date 2020-01-01 \
  --epochs 200

# Adjust hyperparameters
# In .env:
# SEQUENCE_LENGTH=120  # Increase lookback window
# LEARNING_RATE=0.0001  # Adjust learning rate
```

### Performance Tuning

1. **Enable GPU Acceleration**
```env
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0
```

2. **Optimize Database Connection Pool**
```env
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
```

3. **Enable Redis Caching**
```env
CACHE_EXPIRE_SECONDS=7200
ENABLE_CACHE=True
```

4. **Increase Workers**
```dockerfile
CMD ["uvicorn", "main:app", "--workers", "4", "--host", "0.0.0.0"]
```

5. **Use Batch Predictions**
```python
# Predict multiple tickers at once
tickers = ["AAPL", "GOOGL", "MSFT"]
results = await asyncio.gather(*[forecaster.predict(t) for t in tickers])
```

## Health Checks

```bash
# API Health
curl http://localhost:8001/health

# Database Health
docker-compose exec db pg_isready

# Redis Health
docker-compose exec redis redis-cli ping

# Model Health
curl http://localhost:8001/api/v1/forecast/model/info
```

## Backup Strategy

### Automated Backups

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker-compose exec -T db pg_dump -U stockpredict stockpredict > "backup_db_$DATE.sql"

# Backup models
tar -czf "backup_models_$DATE.tar.gz" models/

# Backup predictions (last 30 days)
docker-compose exec -T db psql -U stockpredict stockpredict << EOF
\copy (SELECT * FROM predictions WHERE created_at > NOW() - INTERVAL '30 days') TO '/tmp/predictions_$DATE.csv' CSV HEADER
EOF

# Upload to cloud storage (example)
# aws s3 cp backup_db_$DATE.sql s3://your-bucket/backups/
# aws s3 cp backup_models_$DATE.tar.gz s3://your-bucket/backups/

echo "Backup completed: $DATE"
```

Schedule with cron:
```bash
# Run daily at 2 AM
0 2 * * * /path/to/backup.sh
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.prod.yml
backend:
  deploy:
    replicas: 3
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Load Balancing

Use Nginx upstream:
```nginx
upstream backend {
    least_conn;
    server backend1:8000 weight=3;
    server backend2:8000 weight=2;
    server backend3:8000 weight=1;
}
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stockpredict-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stockpredict-backend
  template:
    metadata:
      labels:
        app: stockpredict-backend
    spec:
      containers:
      - name: backend
        image: yourusername/stockpredict-backend:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
```

## API Usage Examples

### Authentication

```bash
# Register user
curl -X POST http://localhost:8001/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "trader1",
    "password": "SecurePass123!",
    "full_name": "John Doe"
  }'

# Login
curl -X POST http://localhost:8001/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePass123!"

# Create API key
curl -X POST http://localhost:8001/api/v1/auth/api-keys \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Key",
    "expires_in_days": 365
  }'
```

### Stock Predictions

```bash
# Get prediction
curl -X POST http://localhost:8001/api/v1/forecast/predict \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "forecast_days": 30,
    "use_cache": true
  }'

# Get historical data
curl http://localhost:8001/api/v1/forecast/historical/AAPL \
  -H "X-API-Key: YOUR_API_KEY"

# Get supported tickers
curl http://localhost:8001/api/v1/forecast/tickers/supported \
  -H "X-API-Key: YOUR_API_KEY"
```

### Backtesting

```bash
# Run backtest
curl -X POST http://localhost:8001/api/v1/backtest/run \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 10000,
    "strategy": "long_only"
  }'
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/stockpredict/issues
- Documentation: ./docs/
- Email: support@stockpredict.com

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

---

**Happy Trading! 📈🚀**
