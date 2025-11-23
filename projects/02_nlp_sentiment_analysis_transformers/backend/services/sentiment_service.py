"""
Sentiment analysis service.
Handles model loading, inference, and caching.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from typing import Dict, List
import hashlib
import json
from functools import lru_cache

from ..core.config import settings
from ..utils.cache import cache_manager


class SentimentAnalyzer:
    """Sentiment analysis service with caching."""

    def __init__(self):
        self.device = torch.device(
            settings.DEVICE if torch.cuda.is_available() else 'cpu'
        )
        self.model = None
        self.tokenizer = None
        self.sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.model_version = settings.APP_VERSION

    def load_model(self):
        """Load model and tokenizer."""
        if self.model is not None:
            return

        print(f"Loading model: {settings.MODEL_NAME}")
        print(f"Device: {self.device}")

        try:
            # Try to load fine-tuned model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.MODEL_PATH,
                num_labels=3
            )
            self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
            print("Loaded fine-tuned model")
        except:
            # Fall back to pre-trained model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.MODEL_NAME,
                num_labels=3
            )
            self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
            print("Loaded pre-trained model (no fine-tuning found)")

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"sentiment:{hashlib.md5(text.encode()).hexdigest()}"

    async def predict(self, text: str, use_cache: bool = True) -> Dict:
        """
        Predict sentiment for a single text.

        Args:
            text: Input text
            use_cache: Whether to use caching

        Returns:
            Dict with sentiment, confidence, and probabilities
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text)
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Start timing
        start_time = time.time()

        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=settings.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Processing time
        processing_time = (time.time() - start_time) * 1000  # ms

        # Prepare result
        result = {
            'text': text,
            'sentiment': self.sentiment_labels[predicted_class],
            'confidence': round(confidence, 4),
            'probabilities': {
                'negative': round(probabilities[0][0].item(), 4),
                'neutral': round(probabilities[0][1].item(), 4),
                'positive': round(probabilities[0][2].item(), 4)
            },
            'model_version': self.model_version,
            'processing_time_ms': round(processing_time, 2)
        }

        # Cache result
        if use_cache:
            await cache_manager.set(
                cache_key,
                json.dumps(result),
                expire=settings.CACHE_EXPIRE_SECONDS
            )

        return result

    async def predict_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Predict sentiment for multiple texts.

        Args:
            texts: List of input texts
            use_cache: Whether to use caching

        Returns:
            List of prediction results
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached_result = await cache_manager.get(cache_key)
                if cached_result:
                    results.append(json.loads(cached_result))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    results.append(None)  # Placeholder
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            results = [None] * len(texts)

        # Process uncached texts
        if uncached_texts:
            start_time = time.time()

            # Tokenize batch
            inputs = self.tokenizer(
                uncached_texts,
                max_length=settings.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            # Batch inference
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)

            processing_time = (time.time() - start_time) * 1000

            # Process results
            for i, (text, idx) in enumerate(zip(uncached_texts, uncached_indices)):
                predicted_class = torch.argmax(probabilities[i]).item()
                confidence = probabilities[i][predicted_class].item()

                result = {
                    'text': text,
                    'sentiment': self.sentiment_labels[predicted_class],
                    'confidence': round(confidence, 4),
                    'probabilities': {
                        'negative': round(probabilities[i][0].item(), 4),
                        'neutral': round(probabilities[i][1].item(), 4),
                        'positive': round(probabilities[i][2].item(), 4)
                    },
                    'model_version': self.model_version,
                    'processing_time_ms': round(processing_time / len(uncached_texts), 2)
                }

                results[idx] = result

                # Cache result
                if use_cache:
                    cache_key = self._get_cache_key(text)
                    await cache_manager.set(
                        cache_key,
                        json.dumps(result),
                        expire=settings.CACHE_EXPIRE_SECONDS
                    )

        return results

    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.model is None:
            self.load_model()

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            'model_name': settings.MODEL_NAME,
            'model_version': self.model_version,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'max_length': settings.MAX_LENGTH,
            'sentiment_labels': self.sentiment_labels
        }


# Global analyzer instance
analyzer = SentimentAnalyzer()
