"""
Vision AI service for image classification.
Handles model loading, inference, and caching.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time
import hashlib
import json
from typing import Dict, List, Tuple
import io

from ..core.config import settings


class ImageClassifier:
    """Image classification service with caching."""

    def __init__(self):
        self.device = torch.device(
            settings.DEVICE if torch.cuda.is_available() else 'cpu'
        )
        self.model = None
        self.transform = None
        self.model_version = settings.APP_VERSION
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> List[str]:
        """Load ImageNet class names."""
        # Simplified - in production, load from file
        return [f"class_{i}" for i in range(settings.NUM_CLASSES)]

    def load_model(self):
        """Load pre-trained model."""
        if self.model is not None:
            return

        print(f"Loading model: {settings.MODEL_NAME}")
        print(f"Device: {self.device}")

        try:
            # Try to load custom fine-tuned model
            if settings.MODEL_NAME == "resnet50":
                self.model = models.resnet50(pretrained=False)
                self.model.fc = nn.Linear(self.model.fc.in_features, settings.NUM_CLASSES)
                checkpoint = torch.load(settings.MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded fine-tuned model")
        except:
            # Fall back to pre-trained ImageNet model
            if settings.MODEL_NAME == "resnet50":
                self.model = models.resnet50(pretrained=True)
            elif settings.MODEL_NAME == "efficientnet_b0":
                self.model = models.efficientnet_b0(pretrained=True)
            elif settings.MODEL_NAME == "vit_b_16":
                self.model = models.vit_b_16(pretrained=True)
            else:
                self.model = models.resnet50(pretrained=True)
            print("Loaded pre-trained ImageNet model")

        self.model.to(self.device)
        self.model.eval()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(settings.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("Model loaded successfully!")

    async def predict(self, image_bytes: bytes, top_k: int = 5) -> Dict:
        """
        Predict image class.

        Args:
            image_bytes: Image file bytes
            top_k: Number of top predictions to return

        Returns:
            Dict with predictions and metadata
        """
        if self.model is None:
            self.load_model()

        start_time = time.time()

        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size

        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)

        processing_time = (time.time() - start_time) * 1000

        # Prepare predictions
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                'class': self.class_names[idx.item()],
                'class_id': idx.item(),
                'confidence': round(prob.item(), 4)
            })

        result = {
            'predictions': predictions,
            'top_class': predictions[0]['class'],
            'top_confidence': predictions[0]['confidence'],
            'model_version': self.model_version,
            'model_name': settings.MODEL_NAME,
            'image_size': original_size,
            'processing_time_ms': round(processing_time, 2)
        }

        return result

    async def predict_batch(
        self,
        images_bytes: List[bytes],
        top_k: int = 5
    ) -> List[Dict]:
        """Batch prediction for multiple images."""
        if self.model is None:
            self.load_model()

        results = []
        start_time = time.time()

        # Preprocess all images
        image_tensors = []
        for img_bytes in images_bytes:
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            image_tensor = self.transform(image)
            image_tensors.append(image_tensor)

        # Stack into batch
        batch = torch.stack(image_tensors).to(self.device)

        # Batch inference
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)

        processing_time = (time.time() - start_time) * 1000

        # Process results
        for i, probs in enumerate(probabilities):
            top_probs, top_indices = torch.topk(probs, top_k)

            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    'class': self.class_names[idx.item()],
                    'class_id': idx.item(),
                    'confidence': round(prob.item(), 4)
                })

            results.append({
                'predictions': predictions,
                'top_class': predictions[0]['class'],
                'top_confidence': predictions[0]['confidence'],
                'processing_time_ms': round(processing_time / len(images_bytes), 2)
            })

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
            'num_classes': settings.NUM_CLASSES,
            'image_size': settings.IMAGE_SIZE
        }


# Global classifier instance
classifier = ImageClassifier()
