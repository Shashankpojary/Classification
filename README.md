# Project 2 — Data Classification Using AI

## Objective
Load a small dataset, split it into training and testing sets, train a supervised classification algorithm, and evaluate its predictions.

## Dataset
The built-in Iris dataset from scikit-learn (150 samples, 4 numeric features, 3 flower classes). Internet access is not needed to download the dataset.

## Requirements
- Python 3.8 or later
- scikit-learn

## Setup and run
1. Extract this ZIP.
2. Open a terminal in this folder.
3. (Recommended) Create a virtual environment:
   - Windows: `python -m venv .venv`
   - PowerShell: `.venv\Scripts\Activate.ps1`
4. Install dependency: `python -m pip install -r requirements.txt`
5. Run: `python main.py`

## What the program does
1. Loads the Iris dataset.
2. Prints dataset size, features, and class names.
3. Splits data into 80% training and 20% testing.
4. Trains a Decision Tree classifier.
5. Prints accuracy, classification report, and confusion matrix.
6. Predicts the class for one new sample.

## Note
Accuracy can vary with dataset, split, and model settings. This is a beginner-friendly example aligned with the project brief; it is not a production system.
