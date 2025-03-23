import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from datetime import datetime
from xgboost import XGBClassifier
import joblib

# Load the training and test datasets
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
test_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

# Separate target and features
Y = data['meal']
X = data.drop('meal', axis=1)

# Drop identifier column
X = X.drop('id', axis=1)

# Convert 'DateTime' to numeric timestamp
X['DateTime'] = pd.to_datetime(X['DateTime'])
X['DateTime_numeric'] = X['DateTime'].astype('int64') / 10**9
X = X.drop('DateTime', axis=1)

# Fill missing values
X = X.ffill()

# Encode categorical features
cat_cols = X.select_dtypes(include='object').columns
X[cat_cols] = X[cat_cols].apply(lambda col: col.astype('category').cat.codes)

# Feature engineering: extract hour from timestamp
X['hour_of_day'] = X['DateTime_numeric'].apply(lambda ts: datetime.fromtimestamp(ts).hour)

# Split for internal validation
x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42)

# GridSearchCV for model tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid, scoring='accuracy', cv=5)
grid_search.fit(x, y)
modelFit = grid_search.best_estimator_  # Best model

# Optional: Train model again (for explicit reference)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(x, y)

# Evaluate accuracy on validation set
y_pred = model.predict(xt)
accuracy = accuracy_score(yt, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# --- Preprocess test data ---

# Drop ID and convert datetime
test_data = test_data.drop('id', axis=1)
test_data['DateTime'] = pd.to_datetime(test_data['DateTime'])
test_data['DateTime_numeric'] = test_data['DateTime'].astype('int64') / 10**9
test_data = test_data.drop('DateTime', axis=1)

# Fill missing values
test_data = test_data.ffill()

# Encode categoricals
cat_cols_test = test_data.select_dtypes(include='object').columns
test_data[cat_cols_test] = test_data[cat_cols_test].apply(lambda col: col.astype('category').cat.codes)

# Feature engineering for test data
test_data['hour_of_day'] = test_data['DateTime_numeric'].apply(lambda ts: datetime.fromtimestamp(ts).hour)

# Align columns: add missing ones and reorder
missing_cols = set(x.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[x.columns]

# --- Make predictions ---

# Predict probabilities of class 1 (meal)
pred_proba = modelFit.predict_proba(test_data)[:, 1]

# Convert to binary labels using threshold of 0.5
pred_label = (pred_proba >= 0.5).astype(int)

# Save the trained model
joblib.dump(modelFit, 'meal_forecast_model.pkl')

# Output
print(f"\nGenerated {len(pred_proba)} probability predictions.")
print("Sample probabilities:", pred_proba[:10])
print("Sample binary labels:", pred_label[:10])
