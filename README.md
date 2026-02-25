<<<<<<< HEAD
# Image-classification
Image Classification with TensorFlow-Hub using manually collected dataset
=======
# Jaguar Car Model Classification

A deep learning project for classifying Jaguar car models using transfer learning with MobileNetV2 and InceptionV3 architectures.

## Project Structure

```
task1/
├── Jaguar/                          # Dataset directory
│   ├── E-PACE/                      # Class-specific image folders
│   ├── F-PACE/
│   ├── Jaguar XE/
│   ├── Jaguar XF/
│   └── Jaguar XJ/
├── models/                          # Trained models
│   ├── mobilenetv2_best.keras       # Best MobileNetV2 model (highest val accuracy)
│   ├── mobilenetv2_jaguar_final.keras
│   ├── inceptionv3_best.keras       # Best InceptionV3 model (highest val accuracy)
│   └── inceptionv3_jaguar_final.keras
├── plots/                           # Visualizations and plots
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── mobilenetv2_training_history.png
│   ├── mobilenetv2_confusion_matrix.png
│   ├── mobilenetv2_predictions.png
│   ├── inceptionv3_training_history.png
│   ├── inceptionv3_confusion_matrix.png
│   ├── inceptionv3_predictions.png
│   └── model_comparison.png
├── data/                            # Dataset splits and results
│   ├── train_split.csv              # Training set split
│   ├── val_split.csv                # Validation set split
│   ├── test_split.csv               # Test set split
│   └── model_comparison.csv         # Performance comparison results
├── jaguar_classification.ipynb      # Main Jupyter notebook
└── clean_dataset.py                 # Dataset cleaning utility

```

## Dataset

The dataset contains images of 5 Jaguar car models:
- E-PACE
- F-PACE
- Jaguar XE
- Jaguar XF
- Jaguar XJ

**Dataset Split:**
- Training: 70%
- Validation: 15%
- Test: 15%

## Models

### MobileNetV2
- Lightweight CNN architecture optimized for mobile and embedded devices
- Input size: 224x224
- Two-stage training approach:
  1. Train top layers with frozen base
  2. Fine-tune with partially unfrozen layers

### InceptionV3
- Powerful CNN architecture with inception modules
- Input size: 299x299
- Two-stage training approach:
  1. Train top layers with frozen base
  2. Fine-tune with partially unfrozen layers

## Training Strategy

Both models use a **2-stage training approach**:

1. **Stage 1 - Feature Extraction (Frozen Base)**
   - Pre-trained weights from ImageNet are frozen
   - Only the classification head is trained
   - Learning rate: 0.001
   - Epochs: 10

2. **Stage 2 - Fine-Tuning (Unfrozen Layers)**
   - Later layers of the base model are unfrozen
   - Fine-tune on the Jaguar dataset
   - Lower learning rate: 0.0001
   - Epochs: 15

### Data Augmentation

Applied to training set only:
- Rotation: ±30°
- Width/Height shift: 25%
- Shear: 20%
- Zoom: 25%
- Horizontal flip
- Brightness: 70-130%

### Class Weights

Balanced class weights are computed to handle class imbalance in the dataset.

## Results

Performance metrics are available in `data/model_comparison.csv` and visualized in `plots/model_comparison.png`.

## Usage

### Running the Notebook

1. Ensure all dependencies are installed:
   ```bash
   pip install tensorflow keras numpy pandas matplotlib seaborn pillow scikit-learn
   ```

2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook jaguar_classification.ipynb
   ```

### Cleaning the Dataset

Before training, clean the dataset to remove corrupted images:

```bash
python clean_dataset.py Jaguar
```

This script removes:
- macOS metadata files (._* files)
- Corrupted images that cannot be opened

### Loading Trained Models

```python
from tensorflow import keras

# Load the best model
model = keras.models.load_model('models/mobilenetv2_best.keras')
# or
model = keras.models.load_model('models/inceptionv3_best.keras')
```

## Key Features

- **Organized Output Structure**: Models, plots, and data are saved in dedicated directories
- **Reproducible Splits**: Train/val/test splits are saved as CSV files
- **Comprehensive Visualizations**: Training history, confusion matrices, and sample predictions
- **Model Comparison**: Side-by-side comparison of both architectures
- **Best Model Checkpointing**: Automatically saves the best performing model during training

## Files Description

### Models Directory (`models/`)
- `*_best.keras`: Best model checkpoints based on validation accuracy
- `*_final.keras`: Final models after complete training

### Plots Directory (`plots/`)
- `class_distribution.png`: Dataset class distribution visualization
- `sample_images.png`: Sample images from each class
- `*_training_history.png`: Training and validation accuracy/loss curves
- `*_confusion_matrix.png`: Confusion matrices on test set
- `*_predictions.png`: Sample predictions with confidence scores
- `model_comparison.png`: Bar chart comparing model performances

### Data Directory (`data/`)
- `train_split.csv`: Filenames and labels for training set
- `val_split.csv`: Filenames and labels for validation set
- `test_split.csv`: Filenames and labels for test set
- `model_comparison.csv`: Performance metrics comparison table

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow
- scikit-learn

## Notes

- The notebook uses GPU acceleration if available
- Random seeds are set for reproducibility (seed=42)
- Models use .keras format (recommended over legacy .h5 format)
>>>>>>> eff07f7 (initial commit)
