# Machine_learning
This machine learning process file was created for miuul bootcamp final project

---

# Machine Learning Process

This repository demonstrates a comprehensive end-to-end machine learning workflow in Python. The main script, `Machine_learning_process.py`, covers all essential steps for building a robust classification model, from data preparation to model evaluation and saving.

## Overview

The script implements the following steps:

1. **Data Preparation**
   - Loads a CSV file as input data.
   - Standardizes numerical features using `StandardScaler`.
   - Encodes the target variable (assumed to be `"disease"`) with `LabelEncoder`.

2. **Data Splitting and Balancing**
   - Splits the dataset into training and test sets while preserving class distribution.
   - Addresses class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

3. **Baseline Model Evaluation**
   - Evaluates several classification algorithms:
     - Decision Tree (CART)
     - Random Forest (RF)
     - AdaBoost
     - GradientBoosting (GBM)
     - XGBoost
   - Uses 3-fold cross-validation and reports a chosen metric (e.g., ROC AUC).

4. **Hyperparameter Optimization**
   - Performs grid search hyperparameter tuning for the Random Forest model.
   - Compares model performance before and after tuning.
   - Stores the best model for further evaluation.

5. **Model Evaluation on Test Data**
   - Fits the optimized model on the entire training set.
   - Evaluates the model on the test set, outputting:
     - Classification report (precision, recall, F1-score for each class)
     - Confusion matrix

6. **Feature Importance**
   - Computes and displays feature importances as determined by the Random Forest model.

7. **Prediction Example and Model Saving**
   - Generates a prediction for a randomly selected sample.
   - Saves the trained model to disk using `joblib`.

8. **Label Mapping**
   - Prints a mapping of encoded class labels to their original names for interpretability.

## File Structure

- `Machine_learning_process.py` — Main script implementing the full pipeline.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- imbalanced-learn
- xgboost
- joblib

Install requirements with:
```bash
pip install pandas scikit-learn imbalanced-learn xgboost joblib
```

## Usage

1. **Prepare your dataset:**  
   The script expects a CSV file where:
   - The target column is named `"disease"`.
   - All other columns are treated as features.

2. **Edit the data path:**  
   Update the file path in the main section:
   ```python
   X, y, le = data_prep("path/to/your/data.csv")
   ```

3. **Run the script:**
   ```bash
   python Machine_learning_process.py
   ```

4. **Results:**  
   - The script will print class distribution, baseline and optimized model performance, classification reports, feature importances, and a sample prediction.
   - The best Random Forest model will be saved as a `.pkl` file.

## Functions Overview

- `data_prep(filepath)`: Loads and preprocesses the data.
- `split_and_smote(X, y, test_size, random_state)`: Splits the data and applies SMOTE.
- `base_models(X, y, scoring)`: Evaluates baseline classifiers.
- `hyperparameter_optimization(X, y, cv, scoring)`: Tunes hyperparameters (RF example).
- `evaluate_on_test(model, X_test, y_test, label_encoder)`: Evaluates and prints test results.

## Customization

- **Model and Parameters:**  
  You can easily add or swap models in the `base_models` and `hyperparameter_optimization` sections.
- **Evaluation Metrics:**  
  The scoring metric for cross-validation can be changed (e.g., `"roc_auc_ovr"`, `"accuracy"`).
- **Saving and Loading Models:**  
  The script uses `joblib` for model persistence.

