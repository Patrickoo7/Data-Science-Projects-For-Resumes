# 🚀 SentiAI Deployment Guide

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Database Setup](#database-setup)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)

## Local Development

### Backend Setup

```bash
# Navigate to project
cd projects/02_nlp_sentiment_analysis_transformers

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
npm install

# Start development server
npm start
```

Access UI: http://localhost:3000

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
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:3000
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

## Production Deployment

### Prerequisites

1. **Server Requirements**
   - Ubuntu 20.04+ or similar Linux distribution
   - 8GB+ RAM
   - 50GB+ disk space
   - Docker & Docker Compose installed

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
```

### Step 2: Clone Repository

```bash
# Clone repository
git clone https://github.com/yourusername/sentiai.git
cd sentiai/projects/02_nlp_sentiment_analysis_transformers
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
SECRET_KEY=<generate-secure-random-key>
DATABASE_URL=postgresql://sentiai:STRONG_PASSWORD@db:5432/sentiai
BACKEND_CORS_ORIGINS=["https://yourdomain.com"]
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

### Step 6: Configure Nginx (Reverse Proxy)

```nginx
# /etc/nginx/sites-available/sentiai
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/sentiai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 7: SSL with Let's Encrypt

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
| `MODEL_NAME` | Hugging Face model | `bert-base-uncased` |
| `DEVICE` | Compute device | `cuda` or `cpu` |

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

## Database Setup

### PostgreSQL Connection

```bash
# Connect to database
docker-compose exec db psql -U sentiai -d sentiai

# Backup database
docker-compose exec db pg_dump -U sentiai sentiai > backup.sql

# Restore database
docker-compose exec -T db psql -U sentiai sentiai < backup.sql
```

### Migrations

```bash
# Create new migration
docker-compose exec backend alembic revision -m "description"

# Apply migrations
docker-compose exec backend alembic upgrade head

# Rollback migration
docker-compose exec backend alembic downgrade -1
```

## Monitoring Setup

### Prometheus

Access: http://localhost:9090

Key metrics:
- `api_requests_total` - Total API requests
- `api_latency_seconds` - API latency
- `prediction_count` - Prediction count
- `cache_hit_rate` - Cache hit rate

### Grafana

Access: http://localhost:3001
Default credentials: admin/admin

Import dashboards:
1. FastAPI metrics dashboard
2. PostgreSQL dashboard
3. Redis dashboard
4. Custom SentiAI metrics

### Logging

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend

# Export logs
docker-compose logs > logs.txt
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
```

#### 2. Model Loading Error

```bash
# Check available memory
free -h

# Check model files
docker-compose exec backend ls -lh /app/models

# Use CPU if GPU unavailable
# In .env: DEVICE=cpu
```

#### 3. Frontend Not Loading

```bash
# Check frontend status
docker-compose ps frontend

# Rebuild frontend
docker-compose up -d --build frontend

# Check nginx logs
docker-compose logs nginx
```

#### 4. High Memory Usage

```bash
# Check resource usage
docker stats

# Reduce batch size in .env
BATCH_SIZE=16

# Limit workers
# In Dockerfile: --workers 2
```

### Performance Tuning

1. **Enable GPU Acceleration**
```env
DEVICE=cuda
```

2. **Increase Workers**
```dockerfile
CMD ["uvicorn", "main:app", "--workers", "4"]
```

3. **Optimize Cache**
```env
CACHE_EXPIRE_SECONDS=7200
```

4. **Database Connection Pool**
```env
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
```

## Health Checks

```bash
# API Health
curl http://localhost:8000/health

# Database Health
docker-compose exec db pg_isready

# Redis Health
docker-compose exec redis redis-cli ping
```

## Backup Strategy

### Automated Backups

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker-compose exec -T db pg_dump -U sentiai sentiai > "backup_db_$DATE.sql"

# Backup models
tar -czf "backup_models_$DATE.tar.gz" models/

# Upload to cloud storage (example)
# aws s3 cp backup_db_$DATE.sql s3://your-bucket/backups/
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
        cpus: '2'
        memory: 4G
```

### Load Balancing

Use Nginx upstream:
```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/sentiai/issues
- Documentation: ./docs/
- Email: support@sentiai.com

---

**Happy Deploying! 🚀**
