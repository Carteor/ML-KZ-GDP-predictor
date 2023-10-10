import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

import joblib

file_path = 'API_KAZ_DS2_en_excel_v2_5882662.xls'

data = pd.read_excel(file_path, sheet_name="Data", skiprows=3)

columns_to_keep = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"] + [str(year) for year in range(1992, 2023)]
data = data[columns_to_keep]

indicators_of_interest = [
    "GDP (current US$)",
    "Inflation, consumer prices (annual %)",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)",
    "Current account balance (% of GDP)",
    "General government final consumption expenditure (% of GDP)"
]

filtered_data = data[data["Indicator Name"].isin(indicators_of_interest)].copy()

melted_data = pd.melt(filtered_data, id_vars=["Country Name", "Indicator Name", "Indicator Code"],
                      value_vars=[str(year) for year in range(1992,2023)],
                      var_name="Year", value_name="Value")

melted_data["Year"] = pd.to_datetime(melted_data["Year"], format="%Y")

missing_after_imputation = melted_data.isnull().sum()

print(melted_data.head())
print(missing_after_imputation)

plt.figure(figsize=(10, 6))
sns.heatmap(melted_data.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Data Vizualization After Melting")
plt.xlabel("Features")
plt.ylabel("Data Points")
plt.show()


reshaped_data = melted_data.pivot_table(values='Value', index=['Country Name', 'Year'], columns='Indicator Name').reset_index()

reshaped_data = reshaped_data.sort_values('Year')

features = reshaped_data.drop(columns=['GDP (current US$)', 'Country Name', 'Year'])
target = reshaped_data['GDP (current US$)']

train_mask = reshaped_data['Year'].dt.year <= 2012
test_mask = reshaped_data['Year'].dt.year > 2012

X_train, X_test = features[train_mask].values, features[test_mask].values
y_train, y_test = target[train_mask].values, target[test_mask].values


model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)

print(mse, rmse, mae)

feature_importance = model.feature_importances_

features_df = pd.DataFrame({
    'Features': features.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Features', data=features_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


predictions = model.predict(X_test)

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

comparison_df.head()

plt.figure(figsize=(10, 6))
plt.plot(comparison_df['Actual'], label='Actual')
plt.plot(comparison_df['Predicted'], label='Predicted', linestyle='dashed')
plt.legend()
plt.title('Actual vs Predicted GDP')
plt.xlabel('Year (if applicable)')
plt.ylabel('GDP (current US$)')
plt.show()

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test,predictions)

print(mse, rmse, mae)

filename = 'finalized_model.sav'
joblib.dump(model, filename)