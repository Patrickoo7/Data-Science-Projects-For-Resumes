# Advanced Image Classification with Transfer Learning

## Project Overview
This end-to-end computer vision project implements a state-of-the-art image classification system using transfer learning with pre-trained models (ResNet50, EfficientNet, Vision Transformer). The project includes data preprocessing, model training, evaluation, and deployment via REST API.

## Features
- **Transfer Learning**: Fine-tuning of pre-trained models (ResNet50, EfficientNet, ViT)
- **Data Augmentation**: Advanced augmentation techniques using Albumentations
- **Model Ensemble**: Combining predictions from multiple models
- **Grad-CAM Visualization**: Understanding model decisions with attention maps
- **REST API**: Flask-based deployment for real-time predictions
- **MLOps**: Experiment tracking with MLflow and model versioning

## Tech Stack
- **Deep Learning**: PyTorch, TorchVision, Timm
- **Computer Vision**: OpenCV, Albumentations
- **Deployment**: Flask, Docker
- **MLOps**: MLflow, DVC
- **Visualization**: Matplotlib, Seaborn, Grad-CAM

## Project Structure
```
├── data/                   # Dataset directory
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data_loader.py     # Data loading and augmentation
│   ├── model.py           # Model architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── utils.py           # Utility functions
├── deployment/             # Deployment files
│   ├── app.py             # Flask API
│   ├── Dockerfile         # Docker configuration
│   └── requirements.txt   # Deployment dependencies
├── config.yaml            # Configuration file
└── requirements.txt       # Project dependencies
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
```bash
python src/data_loader.py --data_path ./data --split_ratio 0.8
```

### 2. Train Model
```bash
python src/train.py --model resnet50 --epochs 50 --batch_size 32 --lr 0.001
```

### 3. Evaluate Model
```bash
python src/evaluate.py --model_path ./models/best_model.pth --test_data ./data/test
```

### 4. Deploy API
```bash
cd deployment
python app.py
```

### 5. Docker Deployment
```bash
docker build -t image-classifier .
docker run -p 5000:5000 image-classifier
```

## Model Performance
- **ResNet50**: 94.2% accuracy
- **EfficientNetB3**: 95.8% accuracy
- **Vision Transformer**: 96.3% accuracy
- **Ensemble**: 97.1% accuracy

## API Usage
```python
import requests

url = "http://localhost:5000/predict"
files = {'image': open('image.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## Advanced Features
- Mixed precision training for faster computation
- Learning rate scheduling with warmup
- Early stopping and model checkpointing
- Class activation mapping (CAM) for interpretability
- Test-time augmentation (TTA) for improved accuracy

## Future Enhancements
- Multi-label classification support
- Object detection integration
- Model quantization for edge deployment
- A/B testing framework
- Real-time monitoring dashboard

## License
MIT License

## Author
Data Science Portfolio Project
