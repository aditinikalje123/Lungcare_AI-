# LungCare AI – Chest X-Ray Classification

## Project Overview

LungCare AI is a deep learning-based application that classifies chest X-ray images into two categories:

- **NORMAL** – No pneumonia detected by the model.
- **PNEUMONIA** – Features associated with pneumonia detected by the model.

The project uses a custom Convolutional Neural Network (CNN) developed using PyTorch and provides a web interface through FastAPI for uploading X-ray images and viewing predictions.

This project is intended for educational and research purposes and is not a substitute for professional medical diagnosis.

## Problem Statement

Pneumonia is a respiratory infection that causes inflammation in the air sacs of the lungs. Chest X-ray images are commonly used as part of the clinical process to help identify pneumonia.

The objective of this project is to build a deep learning model that analyzes chest X-ray images and classifies them as NORMAL or PNEUMONIA.

## Proposed Solution

The proposed solution uses computer vision and deep learning techniques.

A custom CNN architecture is implemented using PyTorch. The model is trained locally using chest X-ray images and the best-performing model weights are saved as `xray_model.pth`.

A FastAPI application loads the trained model, accepts uploaded X-ray images, preprocesses them, and returns the predicted class along with prediction scores.

## Technology Stack

- Python 3.10.11
- PyTorch
- Torchvision
- FastAPI
- Uvicorn
- Pillow
- HTML
- CSS
- JavaScript

## Model Architecture

The project uses a custom Convolutional Neural Network consisting of:

- Five convolutional layers
- Batch Normalization
- ReLU activation functions
- Max Pooling layers
- Adaptive Average Pooling
- Dropout layers
- Fully connected classification layers

The model accepts RGB images resized to 224 × 224 pixels and produces predictions for two classes: NORMAL and PNEUMONIA.

## Data Preprocessing

The following preprocessing steps are applied:

1. Resize images to 224 × 224 pixels.
2. Convert images to RGB format.
3. Convert images into PyTorch tensors.
4. Normalize images using ImageNet mean and standard deviation.
5. Apply data augmentation during training, including random rotation and color jitter.

## Model Training

The model is trained locally using `train_local.py`.

The training process includes:

- Training and validation dataset splitting.
- Weighted Cross Entropy Loss.
- AdamW optimizer.
- ReduceLROnPlateau learning rate scheduler.
- Early stopping.
- Saving the best model weights.

The trained model is saved as:

`xray_model.pth`

## Project Structure

```text
LungCare-AI/
│
├── app.py
├── index.html
├── requirements.txt
├── train_local.py
├── train.py
├── xray_model.pth
├── README.md
│
├── flowcharts/
│
├── notebook/
│
├── scripts/
│
└── xray/
    ├── components/
    ├── constant/
    ├── entity/
    ├── ml/
    │   └── model/
    │       └── arch.py
    └── pipeline/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/aditinikalje123/Lungcare_AI-.git
```

### 2. Navigate to the project directory

```bash
cd Lungcare_AI-
```

### 3. Create a virtual environment

```bash
python -m venv lungenv
```

### 4. Activate the environment

For Windows PowerShell:

```powershell
.\lungenv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI application using:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

FastAPI interactive documentation:

```text
http://127.0.0.1:8000/docs
```

Health check endpoint:

```text
http://127.0.0.1:8000/health
```

## Training the Model

To train the CNN model locally, run:

```bash
python train_local.py
```

The best model weights are saved to `xray_model.pth`.

## Prediction Workflow

1. Open the LungCare AI web application.
2. Upload a chest X-ray image.
3. The application preprocesses the uploaded image.
4. The trained CNN model performs inference.
5. The application displays the predicted class and prediction scores.

## Disclaimer

LungCare AI is an educational deep learning project. Its predictions are not medical diagnoses and should not be used as the sole basis for medical decisions. Always consult a qualified healthcare professional.

## Author

**Aditi Nikalje**

GitHub: https://github.com/aditinikalje123