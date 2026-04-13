# 🧠 Symptom Clustering ML Project (KMeans + Flask)

Этот проект использует алгоритм KMeans для кластеризации данных по симптомам и Flask для веб-интерфейса, где можно вводить данные и получать предсказанный кластер.

---

## 🚀 Возможности

- Обучение модели KMeans
- Сохранение модели (joblib)
- Предсказание кластера по введённым данным
- Расшифровка кластеров (названия)
- Простой веб-интерфейс (Flask)

---

## 📁 Структура проекта
project/
│
├── train.py # обучение модели
├── test.py # CLI тестирование
├── app.py # Flask сервер
│
├── kmeans_model.pkl # модель
├── features.pkl # список симптомов
├── cluster_names.pkl # названия кластеров
│
├── templates/
│ └── index.html # UI
│
├── static/
│ └── style.css # стили (опционально)
│
├── requirements.txt
└── README.md

---

## ⚙️ Установка

### 1. Создать окружение
```bash
python -m venv venv
2. Активировать
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
3. Установить зависимости
pip install -r requirements.txt
🏋️ Обучение модели
python train.py
🧪 Тест через терминал
python test.py
🌐 Запуск сайта
python app.py

Открыть в браузере:

http://127.0.0.1:5000

📊 Как работает модель
Вводятся симптомы (0 или 1)
KMeans определяет кластер
Кластер переводится в название:
Mild / Asymptomatic
Moderate Symptoms
Severe Symptoms
🧠 Пример входа
Fever: 1
Tiredness: 0
Dry-Cough: 1
...
📌 Требования
Python 3.8+
scikit-learn
pandas
numpy
flask
matplotlib
joblib
👨‍💻 Автор

ML Project for learning clustering + Flask integration


---

# 📦 requirements.txt

```txt
pandas
numpy
scikit-learn
matplotlib
joblib