"""
Model architectures with transfer learning support.
Implements ResNet, EfficientNet, Vision Transformer, and ensemble methods.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Optional, List


class TransferLearningModel(nn.Module):
    """Base transfer learning model with customizable architecture."""

    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(TransferLearningModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Load pre-trained model
        if model_name in ['resnet50', 'resnet101', 'resnet152']:
            self.backbone = self._create_resnet(model_name, pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_name.startswith('efficientnet'):
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif model_name.startswith('vit'):
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        else:
            raise ValueError(f"Model {model_name} not supported")

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def _create_resnet(self, model_name: str, pretrained: bool):
        """Create ResNet model."""
        if model_name == 'resnet50':
            return models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            return models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            return models.resnet152(pretrained=pretrained)

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def get_features(self, x):
        """Extract features without classification."""
        return self.backbone(x)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved accuracy."""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights else [1.0 / len(models)] * len(models)

    def forward(self, x):
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            outputs.append(output * weight)

        # Average predictions
        ensemble_output = torch.stack(outputs).sum(dim=0)
        return ensemble_output


class MultiScaleModel(nn.Module):
    """Multi-scale feature extraction for improved performance."""

    def __init__(self, base_model: nn.Module, scales: List[int] = [224, 256, 288]):
        super(MultiScaleModel, self).__init__()
        self.base_model = base_model
        self.scales = scales

    def forward(self, x):
        outputs = []
        original_size = x.shape[-2:]

        for scale in self.scales:
            if scale != original_size[0]:
                x_scaled = nn.functional.interpolate(
                    x, size=(scale, scale), mode='bilinear', align_corners=False
                )
            else:
                x_scaled = x

            output = self.base_model(x_scaled)
            outputs.append(output)

        # Average predictions across scales
        return torch.stack(outputs).mean(dim=0)


def create_model(
    model_name: str = 'resnet50',
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """Factory function to create model."""
    return TransferLearningModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )


def load_model(checkpoint_path: str, model: nn.Module, device: str = 'cuda'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def save_model(model: nn.Module, optimizer, epoch: int, loss: float, path: str):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    # Test model creation
    model = create_model('resnet50', num_classes=10, pretrained=True)
    print(f"Model created: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
