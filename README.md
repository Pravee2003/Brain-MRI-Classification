# Brain MRI Classification Using a Deep Convolutional Vision Transformer (DC-ViT) Framework with Explainable AI

## Description
This repository implements a hybrid deep learning framework for automated brain tumor classification from MRI scans. The proposed architecture integrates convolutional feature extraction and transformer-based contextual modeling with a boosting-based classifier. Specifically, MobileViT is used to capture local spatial features and fine-grained tumor textures, while the Swin-S Transformer models hierarchical global dependencies using shifted-window self-attention. The fused feature embeddings are passed to an XGBoost classifier, which learns nonlinear decision boundaries in a high-dimensional feature space. Hyperparameter optimization is performed using the Optuna framework with the Tree-structured Parzen Estimator (TPE) algorithm to ensure stable convergence and improved generalization. The framework also incorporates SHAP-based explainability to provide both global and local interpretation of model predictions.

## Dataset Information
The experiments are conducted using the Brain Tumor Segmentation (BraTS) benchmark dataset, a widely used dataset for brain tumor analysis. The dataset contains multi-modal MRI scans including T1, T1-weighted contrast-enhanced (T1ce), T2, and FLAIR sequences, along with expert-annotated tumor segmentation labels.

- Original Format: 3D volumetric NIfTI (.nii.gz)
- Preprocessing:
  - Conversion of 3D volumes into 2D axial slices
  - Intensity normalization
  - Spatial resizing to fixed input dimensions
- Data Split:
  - Training set for model learning
  - Validation set for hyperparameter tuning
  - Test set for unbiased performance evaluation

Dataset Link:
https://sageb.io/RaTSO5

## Code Information
The implementation is modular and developed using Python within a Jupyter Notebook environment. The pipeline consists of:

- Data loading and preprocessing modules
- Feature extraction using CNN, MobileViT, and Swin Transformer backbones
- Feature-level fusion through concatenation of deep embeddings
- Classification using XGBoost (gradient boosting decision trees)
- Hyperparameter optimization using Optuna (TPE sampler)
- Model evaluation using standard classification metrics
- Explainability using SHAP (SHapley Additive exPlanations)

## Usage Instructions
1. Download the BraTS dataset from the provided link and organize it into the required directory structure.
2. Install all dependencies listed in the requirements section.
3. Execute notebooks sequentially:
   - Data loading and preprocessing
   - Feature extraction (CNN / MobileViT / Swin)
   - Hybrid feature fusion and XGBoost training
   - Hyperparameter optimization using Optuna
   - Model evaluation and SHAP-based interpretation
4. Ensure GPU support if available for faster training, although CPU-based execution is also supported.

## Requirements
The implementation requires the following libraries:

- numpy
- pandas
- matplotlib
- opencv-python
- nibabel
- scikit-learn
- torch
- torchvision
- timm
- xgboost
- optuna
- shap
- tqdm

Python version: 3.8 or above

## Methodology
The proposed framework follows a hybrid feature fusion strategy combining convolutional and transformer-based representations. MRI volumes are first converted into 2D slices and preprocessed through normalization and resizing. Feature extraction is performed using MobileViT (local feature representation) and Swin-S Transformer (hierarchical global context modeling). The extracted features are flattened and concatenated to form a unified feature vector. This fused representation is then input to an XGBoost classifier, which learns complex nonlinear relationships through ensemble boosting. Hyperparameter optimization is performed using the Optuna framework with the Tree-structured Parzen Estimator (TPE), which models the probability distribution of promising hyperparameter configurations and maximizes expected improvement. Model performance is evaluated using accuracy, precision, recall, F1-score, and AUROC. SHAP is applied to provide global feature importance, local prediction explanations, and visualization of tumor-relevant regions.

