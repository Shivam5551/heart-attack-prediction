# Heart Attack Prediction Project

This project uses machine learning to predict the likelihood of a heart attack based on medical data. It includes a Jupyter notebook for data analysis, model training, and evaluation, as well as the necessary requirements and a trained model file.

## Project Structure
- `ml-model/heart-attact-prediction.ipynb`: Main notebook containing all code for data loading, preprocessing, visualization, model training, and evaluation.
- `ml-model/requirements.txt`: List of required Python packages.
- `ml-model/heart_attack_model.pkl`: Trained Random Forest model for heart attack prediction.

## Features
- Loads and analyzes a heart attack dataset from Kaggle.
- Visualizes data using Seaborn.
- Splits data into training and test sets.
- Trains a Random Forest classifier.
- Evaluates model performance (accuracy, confusion matrix, classification report).
- Saves and loads the trained model for future predictions.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/Shivam5551/heart-attack-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd heart-attack-prediction/ml-model
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the notebook:
   ```bash
   jupyter notebook heart-attact-prediction.ipynb
   ```

## Usage
- Run the notebook cells step by step to download the dataset, preprocess data, train the model, and evaluate results.
- The trained model is saved as `heart_attack_model.pkl` and can be loaded for predictions.

## Requirements
See `requirements.txt` for all dependencies.

