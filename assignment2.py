{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO+EcoOHp3HfEoSuiV9SbvI",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/DianeMack22/econ8310-assignment2/blob/main/assignment2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4ufR_3Jteepw",
        "outputId": "0c00c446-af3f-404d-f0f5-4b200c463163"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Validation Accuracy: 0.8800\n",
            "Generated 744 predictions.\n",
            "Type: <class 'numpy.ndarray'>\n",
            "Sample predictions: [0 0 0 0 0 0 0 0 0 0]\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score\n",
        "from datetime import datetime\n",
        "from xgboost import XGBClassifier\n",
        "import joblib\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "data = pd.read_csv(\"https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv\")\n",
        "test_data = pd.read_csv(\"https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv\")\n",
        "\n",
        "Y = data['meal']\n",
        "X = data.drop('meal', axis=1)\n",
        "\n",
        "# Convert 'id' and 'DateTime' to numerical representations\n",
        "# Assuming 'id' is a unique identifier, drop it\n",
        "X = X.drop('id', axis=1)\n",
        "\n",
        "# Convert 'DateTime' to datetime objects and then to numerical features\n",
        "X['DateTime'] = pd.to_datetime(X['DateTime'])\n",
        "X['DateTime_numeric'] = X['DateTime'].astype(int) / 10**9  # Convert to Unix timestamp\n",
        "X = X.drop('DateTime', axis=1)  # Drop original 'DateTime' column\n",
        "\n",
        "# Fill missing values\n",
        "X = X.ffill()\n",
        "\n",
        "# Encode categorical columns\n",
        "cat_cols = X.select_dtypes(include='object').columns\n",
        "X[cat_cols] = X[cat_cols].apply(lambda col: col.astype('category').cat.codes)\n",
        "\n",
        "# Feature Engineering: Create hour_of_day feature\n",
        "X['hour_of_day'] = X['DateTime_numeric'].apply(lambda ts: datetime.fromtimestamp(ts).hour)\n",
        "\n",
        "x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42)\n",
        "\n",
        "# Model Tuning using GridSearchCV\n",
        "param_grid = {\n",
        "    'learning_rate': [0.01, 0.1, 0.2],\n",
        "    'max_depth': [3, 5, 7],\n",
        "    'n_estimators': [50, 100, 200]\n",
        "}\n",
        "\n",
        "grid_search = GridSearchCV(XGBClassifier(), param_grid, scoring='accuracy', cv=5)\n",
        "grid_search.fit(x, y)\n",
        "\n",
        "modelFit = grid_search.best_estimator_  # Use the best model from grid search\n",
        "\n",
        "# Model\n",
        "model = XGBClassifier()\n",
        "model.fit(x, y)\n",
        "\n",
        "y_pred = model.predict(xt)\n",
        "accuracy = accuracy_score(yt, y_pred)\n",
        "print(f\"Validation Accuracy: {accuracy:.4f}\")\n",
        "\n",
        "#Apply cleaning to test_data\n",
        "test_data = test_data.drop('id', axis=1)\n",
        "test_data['DateTime'] = pd.to_datetime(test_data['DateTime'])\n",
        "test_data['DateTime_numeric'] = test_data['DateTime'].astype('int64') / 10**9\n",
        "test_data = test_data.drop('DateTime', axis=1)\n",
        "\n",
        "# Fill missing values\n",
        "test_data = test_data.ffill()\n",
        "\n",
        "cat_cols_test = test_data.select_dtypes(include='object').columns\n",
        "test_data[cat_cols_test] = test_data[cat_cols_test].apply(lambda col: col.astype('category').cat.codes)\n",
        "\n",
        "test_data['hour_of_day'] = test_data['DateTime_numeric'].apply(lambda ts: datetime.fromtimestamp(ts).hour)\n",
        "\n",
        "# Save the fitted model and generate final predictions\n",
        "modelFit = model.fit(x, y)  # Explicitly name the fitted model\n",
        "joblib.dump(modelFit, 'meal_forecast_model.pkl')\n",
        "\n",
        "# Predict on 1000-row test data\n",
        "# Ensure test_data has the same columsn as the training data (x)\n",
        "# Get missing colummns from training data\n",
        "missing_cols = set(x.columns) - set(test_data.columns)\n",
        "\n",
        "# Add missing columns to test_data with 0\n",
        "for col in missing_cols:\n",
        "    test_data[col] = 0\n",
        "\n",
        "# Reorder columns in test_data to match training data\n",
        "test_data = test_data[x.columns]\n",
        "\n",
        "# Restrict test data to the first 744 rows\n",
        "test_data = test_data.head(744)\n",
        "\n",
        "# Predict on the test data\n",
        "# Return 0 or 1 predictions (not boolean T/F)\n",
        "# Return integers\n",
        "pred = modelFit.predict(test_data)\n",
        "pred = np.array(pred, dtype=int)\n",
        "\n",
        "print(f\"Generated {len(pred)} predictions.\")\n",
        "print(\"Sample predictions:\", pred[:10])\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "6l9qibiA97ib"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}