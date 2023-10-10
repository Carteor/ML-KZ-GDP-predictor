Kazakhstan GDP Predictor
Overview
This machine learning project aims to predict Kazakhstan's Gross Domestic Product (GDP) based on several economic indicators using the XGBoost regression model. The model was trained on historical economic data from 1992 to 2012 and validated with data from subsequent years.

Indicators Used for Prediction
GDP (current US$)
Inflation, consumer prices (annual %)
Unemployment, total (% of total labor force) (modeled ILO estimate)
Current account balance (% of GDP)
General government final consumption expenditure (% of GDP)
Getting Started

Prerequisites
Python 3.x
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

Installation
Clone the repository:
git clone git@github.com:Carteor/ML-KZ-GDP-predictor.git

Install necessary Python packages:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
Run the Python script.

Dataset
The dataset used in this project originates from the World Bank and spans from 1992 to 2022. It includes various economic indicators for Kazakhstan, but only a select few were utilized for building the prediction model.

Usage
Data Preprocessing: The data is first cleaned and preprocessed, handling missing values and reshaping it to a suitable format for model training.
Model Training: Utilizing the XGBoost regressor model, the data from 1992-2012 is used to train the model.
Model Evaluation: The model is evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) with test data.
Prediction: The model is used to predict the GDP for years post-2012.
Visualization: Various plots represent the actual vs. predicted GDP, feature importance, and others.
Model Performance
The model's performance metrics (MSE, RMSE, MAE) on the test data are as follows:

Mean Squared Error (MSE): [3.157254421334165e+21] 
Root Mean Squared Error (RMSE): [56189451157.08254] 
Mean Absolute Error (MAE): [51858880931.18912]