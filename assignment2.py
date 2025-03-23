{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPzkgX1jYhaXF2mrFH/zvPL",
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
      "execution_count": 9,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4ufR_3Jteepw",
        "outputId": "c4607237-e06d-4a31-85e4-b6809d089bee"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Validation Accuracy: 0.8800\n",
            "Generated 1000 predictions.\n",
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
        "x, xt, y, yt = train_test_split(X, Y, test_size=0.1, random_state=42)\n",
        "\n",
        "from xgboost import XGBClassifier\n",
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
        "# Save the fitted model and generate final predictions\n",
        "import joblib\n",
        "\n",
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
        "# Predict on the test data\n",
        "pred = modelFit.predict(test_data)\n",
        "\n",
        "# Return 0 or 1 predictions (not boolean T/F)\n",
        "pred = pred.astype(int)\n",
        "\n",
        "print(f\"Generated {len(pred)} predictions.\")\n",
        "print(\"Sample predictions:\", pred[:10])\n"
      ]
    }
  ]
}