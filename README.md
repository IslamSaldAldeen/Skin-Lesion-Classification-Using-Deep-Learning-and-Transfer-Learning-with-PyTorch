# HAM10000 Skin Lesion Classification with PyTorch

Academic computer vision project for 7-class skin lesion classification using the HAM10000 dataset. The system receives a dermoscopic skin lesion image and predicts one of seven classes.

> **Important medical disclaimer:** This project is an academic prototype only. It is not clinically validated and must not replace dermatologists or professional diagnosis.

## Dataset

Dataset: **Skin Cancer MNIST: HAM10000** from Kaggle.

Expected files:

```text
data/HAM10000_metadata.csv
data/HAM10000_images_part_1/
data/HAM10000_images_part_2/
```

The project works with the original image folders and metadata file, not only the 28x28 CSV files.

## Classes

| Label | Medical name | General meaning |
|---|---|---|
| akiec | Actinic keratoses / Bowen disease | Pre-cancerous or superficial malignant category |
| bcc | Basal cell carcinoma | Malignant |
| bkl | Benign keratosis-like lesions | Benign |
| df | Dermatofibroma | Benign |
| mel | Melanoma | Malignant and medically important |
| nv | Melanocytic nevi | Usually benign |
| vasc | Vascular lesions | Usually benign |

Label mapping:

```python
{'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
```

## Repository Structure

```text
skin-lesion-classification/
|-- data/
|   |-- README.md
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_training.ipynb
|   |-- 03_evaluation.ipynb
|-- src/
|   |-- dataset.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   |-- gradcam.py
|   |-- utils.py
|   |-- transforms.py
|   |-- metrics.py
|   |-- plots.py
|   |-- run_experiments.py
|-- models/
|-- outputs/
|-- reports/
|-- app.py
|-- requirements.txt
|-- README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## 1. Run EDA

```bash
python -m src.eda
```

This creates:

```text
outputs/class_distribution.png
outputs/sample_images.png
```

## 2. Train One Model

Baseline weighted ResNet50 head-only training:

```bash
python -m src.train --experiment-name exp2_resnet50_weighted_head --architecture resnet50 --freeze head --weighted-loss --epochs 8 --lr 1e-3
```

Outputs:

```text
models/best_exp2_resnet50_weighted_head.pth
outputs/history_exp2_resnet50_weighted_head.csv
outputs/training_curves_exp2_resnet50_weighted_head.png
```

## 3. Run Full Advanced Experiments

```bash
python -m src.run_experiments
```

Experiment plan:

| Experiment | Model | Imbalance handling | Fine-tuning | Purpose |
|---|---|---|---|---|
| Exp 1 | ResNet50 | None | Head only | Baseline |
| Exp 2 | ResNet50 | Class weights | Head only | Handle imbalance |
| Exp 3 | ResNet50 | Class weights | Last block | Improve features |
| Exp 4 | ResNet50 | Class weights | Whole model | Best model candidate |
| Exp 5 | EfficientNet-B0 | Class weights | Fine-tuning | Model comparison |

## 4. Evaluate

```bash
python -m src.evaluate --model-path models/best_exp2_resnet50_weighted_head.pth --architecture resnet50
```

Outputs:

```text
outputs/classification_report.txt
outputs/confusion_matrix.png
outputs/test_metrics.json
outputs/test_predictions.csv
```

## 5. Run Streamlit Demo

```bash
streamlit run app.py
```

The demo supports:

- image upload
- predicted class
- confidence score
- top-3 predictions
- optional Grad-CAM heatmap

## Notebooks

- `01_eda.ipynb`: metadata loading, missing values, class distribution, sample images.
- `02_training.ipynb`: split, DataLoaders, model setup, and experiment commands.
- `03_evaluation.ipynb`: classification report, confusion matrix, predictions, Grad-CAM.

## Evaluation Focus

Because HAM10000 is imbalanced, accuracy alone is not enough. The project tracks:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Weighted F1-score
- Classification report
- Confusion matrix

Special attention should be given to recall for medically important classes such as `mel`, `bcc`, and `akiec`.

## Limitations

- The dataset is imbalanced.
- Some classes have very few images.
- The model is not clinically validated.
- Images are dermoscopic and may not generalize to normal phone camera images.
- Some lesion classes are visually similar, especially `mel`, `nv`, and `bkl`.
- This system must not replace doctors or professional diagnosis.

## Future Work

- Use larger and more diverse datasets.
- Improve melanoma recall.
- Try Vision Transformer and stronger EfficientNet versions.
- Use lesion segmentation before classification.
- Add ensemble learning.
- Add dermatologist-validated testing.
- Deploy as a web or mobile application.
- Improve Grad-CAM explanations inside the user interface.

## Team Members

Add your team members here.
