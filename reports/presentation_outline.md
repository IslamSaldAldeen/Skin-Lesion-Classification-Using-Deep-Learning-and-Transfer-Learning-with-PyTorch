# Presentation Outline - 6 to 8 Minutes

## Slide 1 - Title and Team Members
HAM10000 Skin Lesion Classification Using PyTorch.

## Slide 2 - Problem and Motivation
Skin lesion classification is important because some lesions, especially melanoma, can be medically serious. Our goal is to build an academic decision-support prototype, not a real diagnosis tool.

## Slide 3 - Dataset Overview
HAM10000 contains 10,015 dermoscopic images across seven classes. We use the metadata file and original image folders.

## Slide 4 - Class Distribution
Show `class_distribution.png`. Explain imbalance: `nv` is dominant, while `df` and `vasc` are minority classes.

## Slide 5 - Preprocessing and Augmentation
Images are RGB, 224×224, tensors, ImageNet normalization. Training augmentation only: flips, rotation, resized crop, light color jitter.

## Slide 6 - Model Architecture
Main model: pretrained ResNet50 with final layer replaced by 7 outputs. EfficientNet-B0 is included for comparison.

## Slide 7 - Training Setup and Experiments
Explain the five experiments: baseline, weighted loss, last-block fine-tuning, full fine-tuning, EfficientNet comparison.

## Slide 8 - Results
Show metrics table after training. Emphasize macro F1 because the dataset is imbalanced.

## Slide 9 - Confusion Matrix and Error Analysis
Show confusion matrix. Discuss classes that are confused, especially `mel`, `nv`, and `bkl`.

## Slide 10 - Grad-CAM Explainability
Show Grad-CAM example. Explain that highlighted regions show what influenced the model prediction.

## Slide 11 - Demo
Open Streamlit app, upload image, show predicted class, confidence, top-3 predictions, and Grad-CAM.

## Slide 12 - Limitations and Future Work
Mention imbalance, no clinical validation, dermoscopic-only data, and future work like segmentation, ensembles, and dermatologist validation.
