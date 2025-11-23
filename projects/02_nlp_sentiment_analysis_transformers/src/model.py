"""
Transformer-based sentiment analysis models.
Supports BERT, RoBERTa, DistilBERT, and custom architectures.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    BertModel,
    RobertaModel,
    DistilBertModel,
    AlbertModel
)
from typing import Optional, Dict


class SentimentTransformer(nn.Module):
    """
    Transformer-based sentiment classification model.
    Supports multiple pre-trained models with custom classification heads.
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 3,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        super(SentimentTransformer, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        # Freeze transformer weights if specified
        if freeze_bert:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Classification head
        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        return_attention=False
    ):
        """Forward pass with optional attention weight return."""

        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=return_attention
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Classification
        logits = self.classifier(pooled_output)

        if return_attention:
            return logits, outputs.attentions
        return logits


class MultiTaskSentimentModel(nn.Module):
    """
    Multi-task model for sentiment and emotion classification.
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_sentiment_classes: int = 3,
        num_emotion_classes: int = 6,
        dropout: float = 0.3
    ):
        super(MultiTaskSentimentModel, self).__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        hidden_size = self.config.hidden_size

        # Sentiment classification head
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_sentiment_classes)
        )

        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_emotion_classes)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]

        sentiment_logits = self.sentiment_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)

        return sentiment_logits, emotion_logits


class AspectBasedSentimentModel(nn.Module):
    """
    Aspect-based sentiment analysis model.
    Extracts sentiment for specific aspects mentioned in text.
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 3,
        num_aspects: int = 5
    ):
        super(AspectBasedSentimentModel, self).__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        hidden_size = self.config.hidden_size

        # Aspect detection
        self.aspect_detector = nn.Linear(hidden_size, num_aspects)

        # Aspect-specific sentiment classifier
        self.aspect_sentiment = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Sequence output for aspect detection
        sequence_output = outputs.last_hidden_state

        # Pooled output for overall sentiment
        pooled_output = sequence_output[:, 0, :]

        # Detect aspects
        aspect_logits = self.aspect_detector(pooled_output)

        # Get sentiment for each aspect
        aspect_sentiments = [
            classifier(pooled_output) for classifier in self.aspect_sentiment
        ]

        return aspect_logits, aspect_sentiments


class SentimentEnsemble(nn.Module):
    """Ensemble of multiple transformer models."""

    def __init__(self, models: list, weights: Optional[list] = None):
        super(SentimentEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights else [1.0 / len(models)] * len(models)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = []
        for model, weight in zip(self.models, self.weights):
            logits = model(input_ids, attention_mask, token_type_ids)
            outputs.append(logits * weight)

        ensemble_output = torch.stack(outputs).sum(dim=0)
        return ensemble_output


def load_model_and_tokenizer(model_name: str, num_classes: int = 3):
    """Load pre-trained model and tokenizer."""
    model = SentimentTransformer(model_name=model_name, num_classes=num_classes)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def save_model(model, tokenizer, path: str):
    """Save model and tokenizer."""
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    # Test model creation
    model = SentimentTransformer('bert-base-uncased', num_classes=3)
    print(f"Model: {model.__class__.__name__}")

    # Test forward pass
    dummy_input = torch.randint(0, 1000, (2, 128))
    dummy_mask = torch.ones(2, 128)
    output = model(dummy_input, dummy_mask)
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
