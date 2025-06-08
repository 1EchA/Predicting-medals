import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

data = pd.read_csv('merged_olympic_data.csv')

data.fillna(0, inplace=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Gold', 'Silver', 'Bronze', 'Total', 'Total events']])

#  应用PCA降维至二维
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_features)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data['Cluster'], cmap='viridis', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.title('2D PCA of Medal Performance')
plt.show()