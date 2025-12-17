# Boston Housing Price Prediction 🏠

A Python implementation of a Simple Linear Regression model to predict house prices using the classic Boston Housing dataset. This project demonstrates the end-to-end machine learning workflow from data ingestion to model evaluation.

## 📌 Project Overview

This project uses the `scikit-learn` library to build a regression model. It focuses on the relationship between the **average number of rooms per dwelling (RM)** and the **median value of homes (MEDV)**. 

The script performs data splitting, model training, and evaluates performance using standard regression metrics.

## 🛠️ Tech Stack

* **Language:** Python
* **Data Analysis:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Visualization:** `seaborn`, `matplotlib`

## 📊 Workflow

1.  **Data Loading:** Imports the Boston Housing CSV file.
2.  **Feature Selection:** Isolates `RM` as the independent variable and `MEDV` as the target.
3.  **Data Split:** Uses a 80/20 split for training and testing to ensure model generalization.
4.  **Model Training:** Fits a `LinearRegression` model to the training set.
5.  **Evaluation:** Outputs the error metrics to quantify prediction accuracy.


## 📈 Evaluation Metrics

The model's performance is measured using:

| Metric | Formula | Description |
| :--- | :--- | :--- |
| **MSE** | $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ | Average of the squares of the errors. |
| **RMSE** | $RMSE = \sqrt{MSE}$ | Error magnitude in the original units (thousands of dollars). |
