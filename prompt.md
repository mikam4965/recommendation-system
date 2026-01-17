Привет! Начинаем разработку системы рекомендаций для магистерской диссертации.

## Контекст
Тема: "Проектирование системы анализа поведения пользователей интернет-магазинов с использованием рекомендационных алгоритмов"

Используем реальный датасет **RetailRocket** (Kaggle).

## Задача MVP

### 1. Структура проекта
- Инициализируй проект согласно структуре из CLAUDE.md
- Настрой Docker Compose (PostgreSQL, Redis)
- Создай Makefile с командами

### 2. Загрузка и подготовка RetailRocket

С уже скачанными данными с kaggle, создай pipeline обработки:

```python
# Ожидаемая структура данных после обработки:
events_clean.parquet:
  - visitor_id: int
  - item_id: int  
  - event_type: enum (view, addtocart, transaction)
  - timestamp: datetime
  - session_id: int (сгруппировать по 30 мин gaps)

item_features.parquet:
  - item_id: int
  - category_id: int
  - properties: dict
```

Обработка:
- Удаление ботов (>1000 событий/день)
- Фильтрация редких items (<5 interactions)
- Построение сессий (gap 30 минут)
- Train/validation/test split по времени (70/15/15)

### 3. EDA Notebook
Создай `01_eda_retailrocket.ipynb`:
- Статистика датасета
- Распределение событий по типам
- Воронка view → addtocart → transaction
- Временные паттерны (час, день недели)
- Long-tail распределение items
- Анализ сессий

### 4. Baseline модели
Реализуй и протестируй:
- **Popular Items** — топ по каждому event_type
- **Random** — случайные рекомендации
- **User-based CF** — cosine similarity на user-item matrix
- **Item-based CF** — cosine similarity на item-item matrix

### 5. Базовый API
FastAPI endpoints:
- GET /recommendations/{user_id}?n=10
- POST /events (для симуляции новых событий)
- GET /stats/user/{user_id}

### 6. Evaluation Pipeline
```python
# metrics.py
def precision_at_k(recommended, relevant, k): ...
def recall_at_k(recommended, relevant, k): ...  
def hit_rate(recommended, relevant): ...
def mrr(recommended, relevant): ...
```

## Ожидаемый результат
- `make setup` — поднимает окружение
- `make download-data` — скачивает RetailRocket
- `make process-data` — обрабатывает данные
- `make train-baseline` — обучает baseline модели
- `make evaluate` — выводит таблицу метрик
- Jupyter notebook с EDA
- README с инструкциями

## Вопросы
Перед началом задай уточняющие вопросы. Предложи улучшения.

Начни с создания структуры проекта и скрипта загрузки данных.
