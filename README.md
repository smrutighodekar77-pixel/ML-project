# Student Performance Prediction

## Overview

This project is an end-to-end Machine Learning application that predicts a student's mathematics score based on demographic and academic information. The project includes data preprocessing, exploratory data analysis (EDA), model training, model evaluation, and a Flask web application for making predictions.

## Dataset

The dataset contains the following features:

* Gender
* Race/Ethnicity
* Parental Level of Education
* Lunch
* Test Preparation Course
* Reading Score
* Writing Score

Target Variable:

* Math Score

## Project Workflow

* Data Collection
* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Encoding and Scaling
* Train-Test Split
* Model Training
* Model Evaluation
* Prediction using Flask Application

## Machine Learning Models Used

* Linear Regression
* Ridge Regression
* Lasso Regression
* K-Nearest Neighbors Regressor
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor
* CatBoost Regressor
* AdaBoost Regressor

## Evaluation Metrics

The models were evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* CatBoost
* Flask

## Project Structure

```text
ML-project/
│
├── .ebextensions/
├── artifacts/
├── notebook/
├── src/
├── templates/
├── .gitignore
├── README.md
├── app.py
├── application.py
├── requirements.txt
└── setup.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/smrutighodekar77-pixel/ML-project.git
```

Move to the project directory:

```bash
cd ML-project
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

Then open your browser and visit:

```text
https://ml-project-python-310.up.railway.app/predictdata
```

## Author

Smruti Ghodekar
