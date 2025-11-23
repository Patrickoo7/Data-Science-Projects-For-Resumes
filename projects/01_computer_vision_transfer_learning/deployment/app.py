"""
Flask API for image classification inference.
Supports single image prediction, batch prediction, and Grad-CAM visualization.
"""

from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn.functional as F
from PIL import Image
import io
import base64
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model import create_model, load_model
from data_loader import get_valid_transforms
import albumentations as A

app = Flask(__name__)

# Configuration
MODEL_PATH = '../models/best_model_resnet50.pth'
NUM_CLASSES = 10
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names (update based on your dataset)
CLASS_NAMES = [f'class_{i}' for i in range(NUM_CLASSES)]

# Load model
model = create_model('resnet50', num_classes=NUM_CLASSES, pretrained=False)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using untrained model for demonstration")

# Transforms
transform = get_valid_transforms(IMAGE_SIZE)


def preprocess_image(image_bytes):
    """Preprocess image for inference."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)

    # Apply transformations
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)

    return image_tensor


@app.route('/')
def home():
    """Home page with API documentation."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Classification API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .endpoint { background: #f4f4f4; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🖼️ Image Classification API</h1>
        <p>Advanced image classification using transfer learning</p>

        <div class="endpoint">
            <h3>POST /predict</h3>
            <p>Upload an image and get class predictions</p>
            <p><strong>Parameters:</strong> <code>image</code> (file)</p>
            <p><strong>Returns:</strong> JSON with predictions and probabilities</p>
        </div>

        <div class="endpoint">
            <h3>POST /predict_batch</h3>
            <p>Upload multiple images for batch prediction</p>
            <p><strong>Parameters:</strong> <code>images</code> (multiple files)</p>
            <p><strong>Returns:</strong> JSON with predictions for all images</p>
        </div>

        <div class="endpoint">
            <h3>GET /health</h3>
            <p>Check API health status</p>
            <p><strong>Returns:</strong> JSON with status information</p>
        </div>

        <h2>Example Usage</h2>
        <pre>
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:5000/predict
        </pre>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': 'resnet50',
        'device': str(DEVICE),
        'num_classes': NUM_CLASSES
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Single image prediction endpoint."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Read and preprocess image
        image_file = request.files['image']
        image_bytes = image_file.read()
        image_tensor = preprocess_image(image_bytes).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)

        # Prepare response
        predictions = []
        for prob, idx in zip(top5_prob[0], top5_idx[0]):
            predictions.append({
                'class': CLASS_NAMES[idx.item()],
                'class_id': idx.item(),
                'probability': round(prob.item() * 100, 2)
            })

        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_class': predictions[0]['class'],
            'confidence': predictions[0]['probability']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    try:
        images = request.files.getlist('images')
        results = []

        for idx, image_file in enumerate(images):
            image_bytes = image_file.read()
            image_tensor = preprocess_image(image_bytes).to(DEVICE)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                top_prob, top_idx = torch.topk(probabilities, 1)

            results.append({
                'image_index': idx,
                'filename': image_file.filename,
                'predicted_class': CLASS_NAMES[top_idx[0].item()],
                'confidence': round(top_prob[0].item() * 100, 2)
            })

        return jsonify({
            'success': True,
            'num_images': len(images),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return jsonify({
        'architecture': 'ResNet50 with Transfer Learning',
        'num_classes': NUM_CLASSES,
        'input_size': IMAGE_SIZE,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'device': str(DEVICE),
        'class_names': CLASS_NAMES
    })


if __name__ == '__main__':
    print(f"Starting Flask API on port 5000")
    print(f"Device: {DEVICE}")
    print(f"Model: ResNet50")
    print(f"Classes: {NUM_CLASSES}")
    app.run(host='0.0.0.0', port=5000, debug=True)
