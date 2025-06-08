import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
file_path = 'merged_olympic_data.csv'
data = pd.read_csv(file_path)

noc_list = [
    "USA", "CHN", "GBR", "GER", "AUS", "JPN", "ITA", "FRA", "CUB",
    "HUN", "CAN", "KOR", "ROU", "NED", "BRA", "AZE", "ESP",
    "UKR", "NZL", "UZB"
]

data_filtered = data[data['NOC'].isin(noc_list)]

data_filtered['Gold_Count'] = (data_filtered['Medal'] == 'Gold').astype(int)
data_filtered['Silver_Count'] = (data_filtered['Medal'] == 'Silver').astype(int)
data_filtered['Bronze_Count'] = (data_filtered['Medal'] == 'Bronze').astype(int)

data_grouped = data_filtered.groupby(['NOC', 'Event']).agg({
    'Gold_Count': 'sum',
    'Silver_Count': 'sum',
    'Bronze_Count': 'sum'
}).reset_index()

# 计算Winning Rate
data_grouped['Winning_Rate'] = (data_grouped['Gold_Count'] * 3 + data_grouped['Silver_Count'] * 2 + data_grouped[
    'Bronze_Count']) / (data_grouped['Gold_Count'] + data_grouped['Silver_Count'] + data_grouped['Bronze_Count'])

data_grouped = data_grouped.dropna(subset=['Winning_Rate'])

print("Grouped data with winning rate:")
print(data_grouped.head())

# 为K-means聚类准备数据，选择获奖率和项目作为特征
pivot_data = data_grouped.pivot_table(index='NOC', columns='Event', values='Winning_Rate', aggfunc='mean').fillna(0)

# 标准化
scaler = StandardScaler()
pivot_data_scaled = scaler.fit_transform(pivot_data)

# 使用肘部法来选择K值
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pivot_data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
pivot_data['Cluster'] = kmeans.fit_predict(pivot_data_scaled)

with pd.ExcelWriter('cluster_project_analysis.xlsx') as writer:
    for cluster in pivot_data['Cluster'].unique():
        cluster_data = data_grouped[data_grouped['NOC'].isin(pivot_data[pivot_data['Cluster'] == cluster].index)]
        cluster_data = cluster_data.merge(pivot_data[['Cluster']], left_on='NOC', right_index=True)

        cluster_data.to_excel(writer, sheet_name=f'Cluster_{cluster}', index=False)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=pivot_data.iloc[:, 0], y=pivot_data.iloc[:, 1], hue=pivot_data['Cluster'], palette='viridis', s=100)
plt.title('K-means Clustering of Countries Based on Winning Rate')
plt.xlabel('First Project Winning Rate')
plt.ylabel('Second Project Winning Rate')
plt.legend(title='Cluster', loc='upper right')
plt.show()

print("\nCountries in each cluster:")
print(pivot_data[['Cluster']].groupby('Cluster').apply(lambda x: x.index.tolist()))

data_filtered['Gold_Count'] = (data_filtered['Medal'] == 'Gold').astype(int)
data_filtered['Silver_Count'] = (data_filtered['Medal'] == 'Silver').astype(int)
data_filtered['Bronze_Count'] = (data_filtered['Medal'] == 'Bronze').astype(int)

data_grouped = data_filtered.groupby(['NOC', 'Event']).agg({
    'Gold_Count': 'sum',
    'Silver_Count': 'sum',
    'Bronze_Count': 'sum'
}).reset_index()

data_grouped['Total_Medals'] = data_grouped['Gold_Count'] + data_grouped['Silver_Count'] + data_grouped['Bronze_Count']
data_grouped['Winning_Rate'] = (data_grouped['Gold_Count'] * 3 + data_grouped['Silver_Count'] * 2 + data_grouped['Bronze_Count']) / data_grouped['Total_Medals']

data_grouped = data_grouped[data_grouped['Total_Medals'] > 0]

top_projects = []
for noc in data_grouped['NOC'].unique():
    country_data = data_grouped[data_grouped['NOC'] == noc]
    country_data_sorted = country_data.sort_values(by=['Total_Medals', 'Winning_Rate'], ascending=False)
    top_projects.append(country_data_sorted.head(2))

top_projects_df = pd.concat(top_projects)

top_projects_df.to_excel('top_projects_by_country.xlsx', index=False)

plt.figure(figsize=(12, 8))
sns.scatterplot(data=top_projects_df, x='Total_Medals', y='Winning_Rate', hue='NOC', palette='tab20', s=100)

plt.title('Top 2 Projects for Each Country Based on Medal Count and Winning Rate', fontsize=16)
plt.xlabel('Total Medals', fontsize=12)
plt.ylabel('Winning Rate', fontsize=12)
plt.legend(title='Country', loc='upper right')
plt.tight_layout()

plt.show()


data_grouped['Medal_Points'] = data_grouped['Gold_Count'] * 3 + data_grouped['Silver_Count'] * 2 + data_grouped['Bronze_Count']

top_3_nocs = ["USA", "GBR", "GER"]  # 选择前3个国家

top_n = 3  # 选择前3个项目

top_projects_by_medals = data_grouped.groupby('Event').agg({
    'Medal_Points': 'sum',
    'Winning_Rate': 'mean'
}).reset_index().sort_values(by='Medal_Points', ascending=False).head(top_n)

top_projects_by_rate = data_grouped.groupby('Event').agg({
    'Medal_Points': 'sum',
    'Winning_Rate': 'mean'
}).reset_index().sort_values(by='Winning_Rate', ascending=False).head(top_n)

# 可视化
filtered_data = data_grouped[(data_grouped['Event'].isin(top_projects_by_medals['Event'])) &
                             (data_grouped['NOC'].isin(top_3_nocs))]

print(filtered_data)

sns.set_palette("pastel")

plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
sns.barplot(x='Event', y='Medal_Points', hue='NOC', data=filtered_data)
plt.title('Medal Points by Event and Country (Top 3 Countries)')
plt.xlabel('Event')
plt.ylabel('Medal Points')

plt.subplot(1, 2, 2)
sns.barplot(x='Event', y='Winning_Rate', hue='NOC', data=filtered_data)
plt.title('Winning Rate by Event and Country (Top 3 Countries)')
plt.xlabel('Event')
plt.ylabel('Winning Rate')
plt.tight_layout()
plt.show()
