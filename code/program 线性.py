import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('TkAgg')
data = pd.read_csv('merged_olympic_data.csv')

data['Total events'] = data['Total events'].fillna(0)
data['Total'] = data['Total'].fillna(0)
country_year_data = data[['Year', 'NOC', 'Total events', 'Total']].drop_duplicates(['Year', 'NOC'])
X = country_year_data[['Total events']]  # 特征
y = country_year_data['Total']  # 目标

# 分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_linreg = lin_reg.predict(X_test)

# 使用随机森林回归模型
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# 评估模型
print("Linear Regression - RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_linreg)))
print("Linear Regression - R^2:", r2_score(y_test, y_pred_linreg))

print("Random Forest - RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest - R^2:", r2_score(y_test, y_pred_rf))

# 输出特征重要性 回归系数

rf_feature_importances = rf_reg.feature_importances_

lin_reg_coefficients = lin_reg.coef_

rf_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_feature_importances
}).sort_values(by='Importance', ascending=False)

lin_reg_coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lin_reg_coefficients
}).sort_values(by='Coefficient', ascending=False)

rf_importance_df.to_excel('random_forest_feature_importance.xlsx', index=False)
lin_reg_coeff_df.to_excel('linear_regression_coefficients.xlsx', index=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance_df)

print("\nLinear Regression Coefficients:")
print(lin_reg_coeff_df)

output_data = pd.DataFrame({
    'Actual Total Medals': y_test,
    'Predicted Total (Linear Regression)': y_pred_linreg,
    'Predicted Total (Random Forest)': y_pred_rf
})
output_data.to_excel('model_predictions.xlsx', index=False)
print("\nPredictions have been saved to 'model_predictions.xlsx'.")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=country_year_data['Total events'], y=country_year_data['Total'])
plt.title('Total Events vs. Total Medals')
plt.xlabel('Total Events')
plt.ylabel('Total Medals')
plt.tight_layout()
plt.show()

