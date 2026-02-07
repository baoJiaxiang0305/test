"""
Insurance Claims Prediction Model using ExtraTreesRegressor

This script demonstrates a machine learning model for predicting insurance claims (Incurred).
The model uses RandomizedSearchCV to tune hyperparameters of an ExtraTreesRegressor.
"""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import make_regression

# Generate synthetic insurance claims data
# In reality, this would be loaded from a CSV or database
# Insurance claims typically have a right-skewed distribution
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)

# Make y positive and add skewness to simulate insurance claims
y = np.abs(y) + np.random.exponential(scale=100, size=len(y))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Log transform the target to handle skewness
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Define base estimator with squared_error criterion
# Fixed: Changed from "absolute_error" to "squared_error" to align with R² evaluation
base = ExtraTreesRegressor(
    n_estimators=100,
    criterion="squared_error",  # Predicts conditional mean (aligns with R² and RMSE)
    random_state=42,
    n_jobs=-1
)

# Define parameter distribution for RandomizedSearchCV
# Fixed: Reduced min_samples_leaf values to capture more detail and reduce underfitting
param_dist = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],  # Smaller values to capture more detail
    'max_features': ['sqrt', 'log2', None]
}

# Perform randomized search
print("Performing RandomizedSearchCV...")
random_search = RandomizedSearchCV(
    base,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train_log)

# Get best model
best_model = random_search.best_estimator_

# Make predictions
y_pred_log = best_model.predict(X_test)

# Transform back from log space
y_pred = np.expm1(y_pred_log)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nBest parameters: {random_search.best_params_}")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# Fixed: Using criterion="squared_error" now aligns with R² and RMSE evaluation.
# The model predicts the conditional mean, which is the optimal predictor for squared error loss.
# Combined with smaller min_samples_leaf values, the model can better capture complex relationships.
