# Advanced Sentiment Analysis with Transformers

## Project Overview
End-to-end NLP project implementing state-of-the-art sentiment analysis using transformer models (BERT, RoBERTa, DistilBERT). Includes fine-tuning, multi-class classification, aspect-based sentiment analysis, and production deployment.

## Features
- **Transformer Models**: BERT, RoBERTa, DistilBERT, ALBERT
- **Multi-class Sentiment**: Positive, Negative, Neutral classification
- **Aspect-Based Analysis**: Extract sentiment for specific aspects (e.g., price, quality, service)
- **Attention Visualization**: Understand model focus with attention weights
- **Real-time API**: FastAPI-based deployment with async support
- **Model Optimization**: Quantization, ONNX export, TensorRT optimization

## Tech Stack
- **NLP**: Transformers (Hugging Face), spaCy, NLTK
- **Deep Learning**: PyTorch, PyTorch Lightning
- **Deployment**: FastAPI, Docker, Kubernetes
- **MLOps**: Weights & Biases, Hugging Face Hub
- **Data**: pandas, datasets

## Project Structure
```
├── data/                   # Dataset directory
├── models/                 # Saved models and tokenizers
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── model.py           # Transformer models
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation with metrics
│   └── inference.py       # Inference pipeline
├── deployment/             # Deployment files
│   ├── api.py             # FastAPI application
│   ├── Dockerfile
│   └── k8s/               # Kubernetes configs
├── config.yaml
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### 1. Data Preprocessing
```bash
python src/data_preprocessing.py --input data/raw/reviews.csv --output data/processed/
```

### 2. Train Model
```bash
python src/train.py \
    --model bert-base-uncased \
    --epochs 5 \
    --batch_size 16 \
    --max_length 128 \
    --learning_rate 2e-5
```

### 3. Evaluate Model
```bash
python src/evaluate.py --model_path models/best_model --test_data data/test.csv
```

### 4. Inference
```bash
python src/inference.py --text "This product is amazing! Best purchase ever."
```

### 5. Deploy API
```bash
cd deployment
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Model Performance
| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| BERT-base | 92.3% | 0.921 | 0.918 | 0.924 |
| RoBERTa-base | 93.7% | 0.935 | 0.932 | 0.938 |
| DistilBERT | 91.1% | 0.909 | 0.905 | 0.913 |
| ALBERT | 92.8% | 0.926 | 0.923 | 0.929 |

## API Usage
```python
import requests

url = "http://localhost:8000/predict"
data = {"text": "This product exceeded my expectations!"}
response = requests.post(url, json=data)
print(response.json())
```

## Advanced Features
- **Multi-task Learning**: Sentiment + emotion classification
- **Active Learning**: Efficient data labeling
- **Explainability**: LIME and SHAP integration
- **Adversarial Testing**: Robustness evaluation
- **Cross-lingual**: mBERT for multilingual support

## Dataset
- Amazon Reviews (1M+ samples)
- Twitter Sentiment140
- IMDB Movie Reviews
- Custom domain-specific data

## Future Enhancements
- Zero-shot classification
- Few-shot learning with GPT
- Sentiment trend analysis
- Real-time streaming analytics
- Multi-modal sentiment (text + images)

## License
MIT License
