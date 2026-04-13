import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


# pandas → работа с таблицами (CSV)
# numpy → математика
# matplotlib → графики
# KMeans → алгоритм кластеризации
# StandardScaler → нормализация данных
# PCA → уменьшение размерности (для графика)
# joblib → сохранение модели

# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv('data/car_data.csv')
print(f"Dataset: {df.shape}")

# =========================
# 2. Выбор признаков (ВАЖНО)
# =========================
features = [
    'price',
    'horsepower',
    'enginesize',
    'curbweight',
    'citympg'
]

df = df[features]

# =========================
# 3. Масштабирование
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(df)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'features.pkl')

# =========================
# 4. Elbow метод
# =========================
inertias = []
K_range = range(2, 8)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

# =========================
# 5. Обучение (K=3)
# =========================
K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
df['Cluster'] = labels

joblib.dump(kmeans, 'kmeans_model.pkl')

# =========================
# 6. Названия кластеров (по цене)
# =========================
centroids = kmeans.cluster_centers_

price_index = features.index('price')
prices = centroids[:, price_index]

order = np.argsort(prices)

cluster_names = {}
cluster_colors = {}

name_map = ['Budget Cars', 'Mid Range Cars', 'Premium Cars']
color_map = ['#2ecc71', '#f39c12', '#e74c3c']

for rank, orig in enumerate(order):
    cluster_names[orig] = name_map[rank]
    cluster_colors[orig] = color_map[rank]

joblib.dump(cluster_names, 'cluster_names.pkl')

print("Model saved!")

# =========================
# 7. Elbow график
# =========================
plt.figure()
plt.plot(list(K_range), inertias, 'o-')
plt.axvline(x=3, linestyle='--')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.savefig("elbow.png")
plt.close()

# =========================
# 8. PCA визуализация
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

joblib.dump(pca, 'pca.pkl')

plt.figure()

for i in range(K):
    mask = labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cluster_names[i])

plt.legend()
plt.title("Car Clusters (PCA)")
plt.savefig("pca.png")
plt.close()

# =========================
# 9. Итог
# =========================
print("\n=== Cluster Summary ===")
for i in range(K):
    print(f"Cluster {i} — {cluster_names[i]}: {(labels==i).sum()} cars")

print("\nTraining complete!")