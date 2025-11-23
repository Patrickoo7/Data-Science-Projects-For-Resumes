"""
FastAPI deployment for sentiment analysis.
Supports real-time inference with async support.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
import sys
from pathlib import Path
from typing import List, Dict
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from model import SentimentTransformer

app = FastAPI(
    title="Sentiment Analysis API",
    description="Advanced sentiment analysis using transformer models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_NAME = 'bert-base-uncased'
NUM_CLASSES = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 128

# Sentiment labels
SENTIMENT_LABELS = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Load model and tokenizer
print("Loading model...")
model = SentimentTransformer(model_name=MODEL_NAME, num_classes=NUM_CLASSES)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

try:
    checkpoint = torch.load('../models/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
except:
    print("Using untrained model for demonstration")

model.to(DEVICE)
model.eval()


class TextInput(BaseModel):
    """Single text input for prediction."""
    text: str


class BatchTextInput(BaseModel):
    """Batch text input for predictions."""
    texts: List[str]


class SentimentResponse(BaseModel):
    """Sentiment prediction response."""
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]


@app.get("/")
async def root():
    """API home page."""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "/predict": "POST - Single text prediction",
            "/predict_batch": "POST - Batch predictions",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "num_classes": NUM_CLASSES
    }


@app.post("/predict", response_model=SentimentResponse)
async def predict(input_data: TextInput):
    """Predict sentiment for a single text."""
    try:
        text = input_data.text

        # Tokenize
        encoding = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Prepare response
        probs_dict = {
            SENTIMENT_LABELS[i]: float(probabilities[0][i])
            for i in range(NUM_CLASSES)
        }

        return SentimentResponse(
            text=text,
            sentiment=SENTIMENT_LABELS[predicted_class],
            confidence=round(confidence, 4),
            probabilities=probs_dict
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(input_data: BatchTextInput):
    """Predict sentiment for multiple texts."""
    try:
        texts = input_data.texts
        results = []

        for text in texts:
            encoding = tokenizer(
                text,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)

            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            results.append({
                "text": text,
                "sentiment": SENTIMENT_LABELS[predicted_class],
                "confidence": round(confidence, 4)
            })

        return {
            "success": True,
            "num_texts": len(texts),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def model_info():
    """Get model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "max_length": MAX_LENGTH,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(DEVICE),
        "sentiment_labels": SENTIMENT_LABELS
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
