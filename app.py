from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# data
df = pd.read_csv("data/car_data.csv")

# model
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
cluster_names = joblib.load("cluster_names.pkl")


@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":
        data = [
            float(request.form["price"]),
            float(request.form["horsepower"]),
            float(request.form["enginesize"]),
            float(request.form["curbweight"]),
            float(request.form["citympg"])
        ]

        X = scaler.transform([data])
        cluster = kmeans.predict(X)[0]

        result = {
            "id": cluster,
            "name": cluster_names[cluster]
        }

    table_data = df.head(50).to_dict(orient="records")

    return render_template(
        "index.html",
        table_data=table_data,
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)