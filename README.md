# German Credit Scoring Project

A machine learning project for credit risk assessment using the StatLog German Credit dataset. This project includes model training, evaluation, and a FastAPI-based REST API for credit scoring predictions.

## Project Overview

This project implements multiple machine learning models to predict credit risk (good/bad) based on applicant characteristics. The trained Random Forest model is deployed as a REST API for real-time credit scoring predictions.

## Files & Structure

- **`statlog-summary.ipynb`** - Data exploration and analysis of the German credit dataset
- **`logistic-regression.ipynb`** - Logistic regression model training and evaluation
- **`random-forrest.ipynb`** - Random Forest model training, hyperparameter tuning, and evaluation
- **`api.py`** - FastAPI application serving credit scoring predictions
- **`api_models.py`** - Pydantic models for API request/response validation
- **`data/`** - Contains the German credit dataset
- **`models/`** - Trained model artifacts (German Credit Random Forest model)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- **fastapi** - Web framework for building the REST API
- **uvicorn** - ASGI server for running FastAPI
- **pydantic** - Data validation using Python type hints
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning library
- **joblib** - Model serialization

## Usage

### Training the Model

**Must run first:** Execute the Random Forest notebook to train and export the model:
```bash
jupyter notebook random-forrest.ipynb
```

This notebook trains the Random Forest classifier and saves the model as `models/german_credit_rf.joblib`, which is required for the API to function.

### Running the API

Start the FastAPI server:
```bash
python api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

- **GET `/`** - Welcome message and API documentation
- **POST `/predict`** - Single credit prediction
- **POST `/predict-batch`** - Batch credit predictions

### Example Request

```json
{
  "status": "A11",
  "duration": 12,
  "credit_history": "A34",
  "purpose": "A43",
  "credit_amount": 2100,
  "savings": "A65",
  "employment": "A75",
  "installment_rate": 2,
  "personal_status_sex": "A93",
  "other_debtors": "A101",
  "residence_since": 3,
  "property": "A121",
  "age": 35,
  "other_installment_plans": "A143",
  "housing": "A151",
  "existing_credits": 1,
  "job": "A173",
  "liable_maintenance_people": 1,
  "telephone": "A191",
  "foreign_worker": "A201"
}
```

## Model

The project uses a **Random Forest** classifier trained on the StatLog German Credit dataset. The model predicts whether an applicant should be granted credit based on 20 features including:
- Credit amount and duration
- Employment status and income
- Age and personal information
- Credit history and existing credits

## Dataset

The project uses the **StatLog (German Credit Data)** dataset from the UCI Machine Learning Repository, containing 1000 credit application records with 20 attributes.
