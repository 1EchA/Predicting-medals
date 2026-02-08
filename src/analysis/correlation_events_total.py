import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np  # 添加这一行，导入NumPy库
matplotlib.use('TkAgg')

file_path = 'merged_olympic_data.csv'
data = pd.read_csv(file_path)

noc_list = [
    "USA", "CHN", "GBR", "GER", "AUS", "JPN", "ITA", "FRA", "CUB",
    "HUN", "CAN", "KOR", "ROU", "NED", "BRA", "AZE", "ESP",
    "UKR", "NZL", "UZB"
]

data_filtered = data[data['NOC'].isin(noc_list)]

print("Filtered data for top NOCs:")
print(data_filtered.head())
data_filtered.to_excel('filtered_data.xlsx', index=False)

def medal_to_points(medal):
    if medal == 'Gold':
        return 3
    elif medal == 'Silver':
        return 2
    elif medal == 'Bronze':
        return 1
    else:
        return 0

data_filtered['Medal_Points'] = data_filtered['Medal'].apply(medal_to_points)

data_filtered['Participant_Count'] = 1

columns_to_keep = ['NOC', 'Event', 'Medal_Points', 'Participant_Count', 'Total']
data_grouped = data_filtered.groupby(['NOC', 'Event'], as_index=False).agg({
    'Medal_Points': 'sum',
    'Participant_Count': 'sum',
    'Total': 'sum',
})

data_grouped = data_grouped[data_grouped['Participant_Count'] >= 10]

data_grouped['Winning_Rate'] = data_grouped['Medal_Points'] / data_grouped['Participant_Count']

print("Grouped data by NOC and Event with Winning Rate:")
print(data_grouped.head())
data_grouped.to_excel('grouped_data_with_winning_rate.xlsx', index=False)

total_medals_by_event = data_grouped.groupby('Event')['Total'].sum().reset_index()

print("Total Medals by Event:")
print(total_medals_by_event)

noc_event_pivot = data_grouped.pivot(index='Event', columns='NOC', values='Total').fillna(0)

print("Pivot table for heatmap with Total Medals:")
print(noc_event_pivot)
noc_event_pivot.to_excel('pivot_table_with_total_medals.xlsx')

plt.figure(figsize=(16, 12))
sns.heatmap(noc_event_pivot, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Relationship Between Events and Top NOCs (Based on Total Medals)')
plt.xlabel('NOC')
plt.ylabel('Event')
plt.tight_layout()
plt.show()

X = pd.get_dummies(data_grouped['Event'], drop_first=True)  # One-hot encode Event
y = data_grouped['Total']  # Total medals

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print(f"Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"Linear Regression R^2: {r2_score(y_test, y_pred)}")

event_importance = pd.DataFrame({
    'Event': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("Event Coefficients (Impact on Total Medals):")
print(event_importance)
event_importance.to_excel('event_coefficients.xlsx', index=False)
