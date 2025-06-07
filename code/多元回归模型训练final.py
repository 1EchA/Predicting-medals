import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 设置为 TkAgg 后端

data = pd.read_csv('constructed_dataset_with_gold.csv')  # 替换为你最终构建的csv文件名

data.fillna(0, inplace=True)

# 选择自变量和因变量
features = ['Total_Events', 'Gender_Ratio', 'Is_Host', 'Historical_Medals', 'Total_Participants']
target = 'Gold'

X = data[features]
y = data[target]

# 随机划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,   # 20%数据做测试
    random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 预测及评估
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("训练集 MSE:", train_mse, "R²:", train_r2)
print("测试集  MSE:", test_mse,  "R²:", test_r2)

print("\n回归系数:", model.coef_)
print("截距(intercept):", model.intercept_)

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_test, y_test - y_pred_test, alpha=0.5, color='#E2CFC3')  # 淡金色
plt.axhline(0, color='#A3B18C', linestyle='--', linewidth=1.2)  # 抹茶色
plt.xlabel('Predicted', fontsize=12, color='#4C5866')  # 设置字体颜色
plt.ylabel('Residual', fontsize=12, color='#4C5866')
plt.title('Residual Plot (Test)', fontsize=14, color='#4C5866')
plt.grid(color='#D3D3D3', linestyle='--', linewidth=0.5, alpha=0.7)  # 添加网格线，低饱和度灰色
plt.tick_params(colors='#4C5866')  # 坐标轴刻度颜色

plt.tight_layout()
plt.show()

