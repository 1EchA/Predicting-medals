import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
file_path = 'merged_olympic_data.csv'
data = pd.read_csv(file_path)

noc_list = [
    "USA", "CHN", "GBR", "GER", "AUS", "JPN", "ITA", "FRA", "CUB",
    "HUN", "CAN", "KOR", "ROU", "NED", "BRA", "AZE", "ESP",
    "UKR", "NZL", "UZB"
]

data_filtered = data[data['NOC'].isin(noc_list)]

data_grouped = data_filtered.groupby(['NOC', 'Event']).agg(
    Gold=('Medal', lambda x: (x == 'Gold').sum()),
    Silver=('Medal', lambda x: (x == 'Silver').sum()),
    Bronze=('Medal', lambda x: (x == 'Bronze').sum())
).reset_index()

print("Grouped data by NOC and Event with Medal Counts:")
print(data_grouped.head())

data_grouped['Winning_Rate'] = (data_grouped['Gold']*3 + data_grouped['Silver']*2 + data_grouped['Bronze']) / (data_grouped['Gold'] + data_grouped['Silver'] + data_grouped['Bronze'])

print("Grouped data with Winning Rate:")
print(data_grouped.head())

data_grouped = data_grouped[data_grouped['Winning_Rate'] > 0]

pivot_data = data_grouped.pivot_table(index='NOC', columns='Event', values='Winning_Rate', aggfunc='mean').fillna(0)
scaler = StandardScaler()
pivot_data_scaled = scaler.fit_transform(pivot_data)

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

plt.figure(figsize=(10, 8))
sns.scatterplot(x=pivot_data['Gold'], y=pivot_data['Silver'], hue=pivot_data['Cluster'], palette='viridis', s=100)
plt.title('K-means Clustering of Countries Based on Winning Rate')
plt.xlabel('Gold Medals')
plt.ylabel('Silver Medals')
plt.legend(title='Cluster', loc='upper right')
plt.show()

print(pivot_data[['Cluster']].groupby('Cluster').apply(lambda x: x.index.tolist()))
