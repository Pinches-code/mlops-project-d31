# MLOps Project

## Описание проекта

Проект разработан в рамках ДЗ1 по MLOps. Цель — предсказание стоимости аренды жилья (`price`) с помощью модели `RandomForestRegressor`.
Данные загружаются из S3, обработка и обучение выполняются с помощью Scikit-learn.

## Технологии

- Python 3.11
- Pandas, Scikit-learn
- Boto3
- FastAPI
- Docker и Docker Compose
- Jupyter Notebook

## Структура проекта

```text
├── notebooks/
│   ├── 01_training.ipynb
│   └── 02_pipeline.ipynb
├── src/
│   ├── train.py
│   └── api.py
├── artifacts/
├── requirements.txt
├── Dockerfile.train
├── Dockerfile.api
├── compose.yaml
├── .env.example
├── .gitignore
└── README.md
```

## Данные

Используется датасет с объявлениями об аренде жилья: 20 151 строка и 13 столбцов.
Целевая переменная — `price`.
Данные загружаются напрямую из S3 с помощью `boto3`.

## Настройка

Создайте файл `.env` в корне проекта:

```env
S3_ENDPOINT_URL=
S3_BUCKET=
S3_OBJECT_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

## Обучение

- `01_training.ipynb` — обучение модели и оценка качества.
- `02_pipeline.ipynb` — создание sklearn Pipeline, сохранение и загрузка модели.

Модель сохраняется в `artifacts/model.pkl`.
Для оценки используются MAE, RMSE и R².

## Docker

Обучение модели выполняется в контейнере `train`, API работает в отдельном контейнере `api`.
Оба сервиса используют общий Docker volume для хранения модели.
Запуск всего проекта:

```bash
docker compose up --build
```

Остановка:

```bash
docker compose down
```

## API

После запуска документация API доступна по адресу:
http://localhost:8000/docs
Основной endpoint:

`POST /predict`

Пример JSON:

```json
{
  "latitude": 48.85,
  "longitude": 2.35,
  "accommodates": 2,
  "bathrooms": 1,
  "bedrooms": 1,
  "beds": 1,
  "minimum_nights": 2,
  "neighbourhood_cleansed": "Example neighbourhood",
  "property_type": "Apartment",
  "room_type": "Entire home/apt"
}
```

API возвращает предсказанную стоимость аренды.

## Безопасность

S3-ключи хранятся в `.env` и не добавляются в Git. Файл модели создаётся при запуске и хранится в Docker volume.

## Результат

Реализованы загрузка данных из S3, обучение модели, sklearn Pipeline, сохранение модели, Docker Batch и FastAPI для получения предсказаний.