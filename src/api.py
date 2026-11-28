import pickle
import pandas as pd

from fastapi import FastAPI

# Загружаем обученную модель
with open("artifacts/model.pkl", "rb") as file:
    model = pickle.load(file)


app = FastAPI()


# Получение предсказания
@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return {"prediction": float(prediction)}
