import os
import pickle
import boto3
import pandas as pd

from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Читаем настройки S3
BUCKET = os.getenv("S3_BUCKET")
DATA_KEY = os.getenv("S3_OBJECT_KEY")
ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL") or None

if not BUCKET or not DATA_KEY:
    raise ValueError("S3_BUCKET и S3_OBJECT_KEY должны быть заданы")


# Подключаемся к S3
s3 = boto3.client(
    "s3", endpoint_url=ENDPOINT_URL, region_name=os.getenv("AWS_DEFAULT_REGION") or None
)

response = s3.get_object(Bucket=BUCKET, Key=DATA_KEY)
df = pd.read_csv(BytesIO(response["Body"].read()))

print("Dataset shape:", df.shape)


# Подготавливаем данные
df = df.drop(columns=["listing_id", "snapshot_date"])

X = df.drop(columns=["price"])
y = df["price"]


# Делим данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Определяем типы признаков
numeric_features = X_train.select_dtypes(include="number").columns

categorical_features = X_train.select_dtypes(exclude="number").columns


# Создаём предобработку
preprocessor = ColumnTransformer(
    [
        ("num", SimpleImputer(strategy="median"), numeric_features),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)


# Создаём Pipeline
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ]
)


# Обучаем модель
pipeline.fit(X_train, y_train)


# Оцениваем модель
y_pred = pipeline.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R2:", r2_score(y_test, y_pred))


# Сохраняем модель
os.makedirs("artifacts", exist_ok=True)

with open("artifacts/model.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("Model saved successfully!")
