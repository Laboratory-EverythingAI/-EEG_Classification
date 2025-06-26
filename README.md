# EEG-Based Motor Imagery Classification

## Overview
This repository contains the implementation of an advanced deep learning framework for EEG-based motor imagery classification. The model achieves state-of-the-art performance (97.25% accuracy) using a CNN-LSTM architecture with spatio-temporal attention mechanisms.

## Key Features
- Multi-class motor imagery classification (left hand, right hand, feet, tongue)
- Advanced preprocessing pipeline with bandpass filtering and CSP
- Hybrid CNN-LSTM architecture with attention mechanism
- Comprehensive ablation studies
- Multiple baseline model implementations


## Repository Structure
```
EEG_Classification/
├── model/
│   ├── attention.py        # Attention mechanism implementations
│   ├── cnn_lstm.py        # Main model architecture
│   ├── baseline.py        # Baseline models (RF, SVM, LSTM)
│   └── utils.py           # Helper functions
├── setups/
│   ├── config.py          # Configuration parameters
│   ├── dataloader.py      # Data loading and preprocessing
│   └── transforms.py      # Data transformation functions
├── result/
│   ├── ablation/          # Ablation study results
│   ├── figures/           # Generated figures
│   └── models/            # Saved model checkpoints
├── train.py               # Training script
├── test.py                # Testing and evaluation script
└── requirements.txt       # Dependencies
```

## Installation
```bash
# Clone the repository
git clone https://github.com/Laboratory-EverythingAI/EEG_Classification.git
cd EEG_Classification

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy
- scikit-learn
- pandas
- matplotlib
- seaborn

## Usage
### Training
```bash
python train.py --config setups/config.py
```

### Testing
```bash
python test.py --model_path result/models/best_model.pth
```

## Model Architecture
```
Input (EEG Signals)
    │
    ▼
Preprocessing
    │
    ▼
CNN Layers
    │
    ▼
LSTM Layers
    │
    ▼
Attention Mechanism
    │
    ▼
Classification
```

## Performance
| Model | Accuracy (%) | F1-Score (%) | Training Time (s) |
|-------|-------------|--------------|------------------|
| CNN-LSTM-Attention | 97.25 ± 0.78 | 97.21 ± 0.83 | 523.6 ± 21.5 |
| CNN-LSTM | 93.58 ± 1.12 | 93.39 ± 1.23 | 478.2 ± 18.9 |
| Random Forest | 92.52 ± 1.23 | 92.19 ± 1.34 | 45.3 ± 2.7 |

## Ablation Studies
The repository includes comprehensive ablation studies on:
- Different attention mechanisms
- Network components
- Preprocessing techniques

Detailed results can be found in `result/ablation/`.
