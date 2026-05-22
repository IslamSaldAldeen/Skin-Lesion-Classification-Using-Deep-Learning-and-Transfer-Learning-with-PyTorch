# HAM10000 Skin Lesion Classification Project Report

## 1. Title
HAM10000 Skin Lesion Classification Using PyTorch Transfer Learning

## 2. Introduction
This project builds an academic decision-support prototype for classifying dermoscopic skin lesion images into seven HAM10000 diagnostic categories. The system is not clinically validated and should not replace dermatologists or professional medical diagnosis.

## 3. Problem Statement
The task is multi-class image classification. The input is a dermoscopic skin lesion image, and the output is one of seven classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`.

## 4. Dataset Description
Dataset: Skin Cancer MNIST: HAM10000. The project uses the metadata file and original image folders:

- `HAM10000_metadata.csv`
- `HAM10000_images_part_1/`
- `HAM10000_images_part_2/`

Important metadata columns include `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, and `localization`.

## 5. Exploratory Data Analysis
EDA should include dataset shape, first rows, missing values, class distribution, and sample images from every class. The class distribution is imbalanced, with `nv` usually dominant and `df`/`vasc` much smaller.

## 6. Preprocessing
Images are converted to RGB, resized to 224×224, converted to PyTorch tensors, and normalized using ImageNet mean and standard deviation.

## 7. Data Augmentation
Training images use horizontal flip, vertical flip, random rotation, random resized crop, and light color jitter. Validation and test images do not use augmentation.

## 8. Model Architecture
The main architecture is pretrained ResNet50. The final classification layer is replaced with a 7-class output layer. EfficientNet-B0 is included as an additional comparison model.

## 9. Training Setup
The project uses CrossEntropyLoss and Weighted CrossEntropyLoss for imbalance handling. AdamW is used as the main optimizer. The best model is saved based on validation macro F1-score.

## 10. Experiments
| Experiment | Model | Imbalance Handling | Fine-Tuning | Purpose |
|---|---|---|---|---|
| Exp 1 | ResNet50 | None | Head only | Baseline |
| Exp 2 | ResNet50 | Class weights | Head only | Handle imbalance |
| Exp 3 | ResNet50 | Class weights | Last block | Improve features |
| Exp 4 | ResNet50 | Class weights | Whole model | Best candidate |
| Exp 5 | EfficientNet-B0 | Class weights | Fine-tuning | Model comparison |

## 11. Results
Add final values after training:

| Experiment | Accuracy | Macro F1 | Weighted F1 | Notes |
|---|---:|---:|---:|---|
| Exp 1 |  |  |  |  |
| Exp 2 |  |  |  |  |
| Exp 3 |  |  |  |  |
| Exp 4 |  |  |  |  |
| Exp 5 |  |  |  |  |

## 12. Confusion Matrix Analysis
Discuss which classes are confused most often. Pay special attention to melanoma (`mel`) confusion with `nv` and `bkl`.

## 13. Error Analysis
Mention best-performing and worst-performing classes, the effect of imbalance, and why missing melanoma cases can be more serious than classifying a benign lesion incorrectly.

## 14. Grad-CAM / Explainability
Grad-CAM was used to highlight the regions that contributed most to the predicted skin lesion class. This improves interpretability, which is important for medical image projects.

## 15. Limitations
- The dataset is imbalanced.
- Some classes have very few images.
- The model is not clinically validated.
- Dermoscopic images may not generalize to normal phone camera images.
- Some lesion classes are visually similar.
- The system should not replace doctors.

## 16. Future Work
- Use larger and more diverse datasets.
- Improve melanoma recall.
- Try Vision Transformers and stronger EfficientNet models.
- Use segmentation before classification.
- Add ensemble learning.
- Add dermatologist-validated testing.
- Deploy as a web or mobile application.

## 17. Conclusion
The project demonstrates a complete deep learning pipeline for medical image classification, including EDA, preprocessing, transfer learning, imbalance handling, evaluation, Grad-CAM explainability, and a Streamlit demo.

## 18. References
- HAM10000 dataset on Kaggle.
- PyTorch documentation.
- Torchvision pretrained models.
- Grad-CAM explainability method.
