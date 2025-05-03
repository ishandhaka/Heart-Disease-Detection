# Heart Disease Detection System

This project implements a machine learning system to detect heart disease based on various medical attributes.

## Dataset

The dataset (`dataset.csv`) contains the following features:
- age: Age in years
- sex: Sex (1 = male, 0 = female)
- chest pain type: Type of chest pain
- resting bp s: Resting blood pressure (mm Hg)
- cholesterol: Serum cholesterol (mg/dl)
- fasting blood sugar: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- resting ecg: Resting electrocardiographic results
- max heart rate: Maximum heart rate achieved
- exercise angina: Exercise-induced angina (1 = yes, 0 = no)
- oldpeak: ST depression induced by exercise relative to rest
- ST slope: Slope of the peak exercise ST segment
- target: Heart disease diagnosis (1 = disease, 0 = no disease)

## Project Files

- `explore_data.py`: Script to explore and visualize the dataset
- `heart_disease_model.py`: Script to train and evaluate multiple machine learning models
- `predict.py`: Interactive script to make predictions using the trained model
- `dataset.csv`: Heart disease dataset
- Various output plots (generated when running the scripts)

## Installation

1. Make sure you have Python 3.6+ installed
2. Install required packages:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

### Step 1: Explore the dataset
```
python explore_data.py
```
This will generate exploratory plots and display dataset information.

### Step 2: Train models
```
python heart_disease_model.py
```
This will train multiple machine learning models, evaluate their performance, and save the best model.

### Step 3: Make predictions
```
python predict.py
```
This will prompt you to enter patient information and provide a heart disease prediction.

## Model Performance

The system evaluates several machine learning models:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The best performing model is automatically saved and used for predictions. 