import pandas as pd
import joblib

# =========================
# 1. Загрузка модели
# =========================
# kmeans_model.pkl → сама обученная модель
# scaler.pkl → нормализация данных
# features.pkl → порядок колонок
# cluster_names.pkl → названия кластеров
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')
cluster_names = joblib.load('cluster_names.pkl')

# =========================
# 2. Ввод данных
# =========================
print("Введите данные автомобиля:\n")

data = {
    'price': float(input("price: ")),
    'horsepower': float(input("horsepower: ")),
    'enginesize': float(input("enginesize: ")),
    'curbweight': float(input("curbweight: ")),
    'citympg': float(input("citympg: "))
}

# =========================
# 3. Преобразование
# =========================
df = pd.DataFrame([data])

# порядок колонок (на всякий случай)
df = df[features]

X = scaler.transform(df)

# =========================
# 4. Предсказание
# =========================
cluster = kmeans.predict(X)[0]
cluster_name = cluster_names[cluster]

# =========================
# 5. Вывод
# =========================
print("\n=== RESULT ===")
print(f"Cluster ID: {cluster}")
print(f"Car Category: {cluster_name}")