import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
test_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

Y = data['meal']
X = data.drop('meal', axis=1)

# Convert 'id' and 'DateTime' to numerical representations
# Assuming 'id' is a unique identifier, drop it
X = X.drop('id', axis=1)

# Convert 'DateTime' to datetime objects and then to numerical features
X['DateTime'] = pd.to_datetime(X['DateTime'])
X['DateTime_numeric'] = X['DateTime'].astype(int) / 10**9  # Convert to Unix timestamp
X = X.drop('DateTime', axis=1)  # Drop original 'DateTime' column

# Fill missing values
X = X.ffill()

# Encode categorical columns
cat_cols = X.select_dtypes(include='object').columns
X[cat_cols] = X[cat_cols].apply(lambda col: col.astype('category').cat.codes)

# Feature Engineering: Create hour_of_day feature
X['hour_of_day'] = X['DateTime_numeric'].apply(lambda ts: datetime.fromtimestamp(ts).hour)

x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42)

# Model Tuning using GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(XGBClassifier(), param_grid, scoring='accuracy', cv=5)
grid_search.fit(x, y)

modelFit = grid_search.best_estimator_  # Use the best model from grid search

# Model
model = XGBClassifier()
model.fit(x, y)

y_pred = model.predict(xt)
accuracy = accuracy_score(yt, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

#Apply cleaning to test_data
test_data = test_data.drop('id', axis=1)
test_data['DateTime'] = pd.to_datetime(test_data['DateTime'])
test_data['DateTime_numeric'] = test_data['DateTime'].astype('int64') / 10**9
test_data = test_data.drop('DateTime', axis=1)

# Fill missing values
test_data = test_data.ffill()

cat_cols_test = test_data.select_dtypes(include='object').columns
test_data[cat_cols_test] = test_data[cat_cols_test].apply(lambda col: col.astype('category').cat.codes)

test_data['hour_of_day'] = test_data['DateTime_numeric'].apply(lambda ts: datetime.fromtimestamp(ts).hour)

# Save the fitted model and generate final predictions
modelFit = model.fit(x, y)  # Explicitly name the fitted model
joblib.dump(modelFit, 'meal_forecast_model.pkl')

# Predict on 1000-row test data
# Ensure test_data has the same columsn as the training data (x)
# Get missing colummns from training data
missing_cols = set(x.columns) - set(test_data.columns)

# Add missing columns to test_data with 0
for col in missing_cols:
    test_data[col] = 0

# Reorder columns in test_data to match training data
test_data = test_data[x.columns]

# Restrict test data to 744 rows
test_data = test_data.head(744)

# Predict on the test data
# Return 0 or 1 predictions (not boolean T/F)
# Return integers
pred = modelFit.predict_proba(test_data)[:, 1]

print(f"Generated {len(pred)} predictions.")
print("Sample predictions:", pred[:10])
