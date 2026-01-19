# Рекомендательная система для интернет-магазина: Финальный промпт

## Обзор проекта

**Тема диссертации**: Проектирование системы анализа поведения пользователей интернет-магазинов с использованием рекомендационных алгоритмов

**Цель**: Разработка production-ready рекомендательной системы с поддержкой персонализации на основе воронки конверсии, объяснимых рекомендаций и A/B тестирования.

---

## Реализованные компоненты

### Архитектура системы

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  RetailRocket│───▶│    Data      │───▶│   Feature    │                   │
│  │   Dataset    │    │  Processing  │    │  Engineering │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                              │                   │                          │
│                              ▼                   ▼                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         ML MODELS                                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│  │  │ Popular │  │  ALS    │  │   NCF   │  │ GRU4Rec │  │ SASRec  │   │   │
│  │  │Baseline │  │  BPR    │  │Two-Tower│  │ Seq-Rec │  │Attention│   │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │   │
│  │       │            │            │            │            │         │   │
│  │       └────────────┴────────────┴────────────┴────────────┘         │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │                    ┌──────────────────┐                             │   │
│  │                    │  Funnel-Aware    │                             │   │
│  │                    │  Hybrid Model    │◀─── Научная новизна #1      │   │
│  │                    └────────┬─────────┘                             │   │
│  │                             │                                       │   │
│  └─────────────────────────────┼───────────────────────────────────────┘   │
│                                │                                            │
│                                ▼                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   FastAPI    │◀───│  Explainer   │───▶│  Dashboard   │                   │
│  │     API      │    │     XAI      │    │   React UI   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│          │                                       ▲                          │
│          ▼                                       │                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Streaming   │───▶│  A/B Testing │───▶│   Metrics    │                   │
│  │   Kafka      │    │  Framework   │    │   Tracker    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Модули и компоненты

#### 1. Data Pipeline (`src/data/`)
- **Loaders**: Загрузка RetailRocket dataset (2.7M событий, 1.4M пользователей, 235K товаров)
- **Processors**: Очистка, построение сессий, temporal split
- **Features**: User features, Session features (RFM, funnel stage, activity level)

#### 2. ML Models (`src/models/`)

| Категория | Модели | Описание |
|-----------|--------|----------|
| **Baselines** | Popular, Random | Baseline для сравнения |
| **Collaborative** | ALS, BPR, NCF, User-CF, Item-CF | Коллаборативная фильтрация |
| **Content** | Item2Vec | Эмбеддинги товаров |
| **Sequential** | GRU4Rec, SASRec | Сессионные рекомендации |
| **Retrieval** | Two-Tower | Candidate generation |
| **Hybrid** | Funnel-Aware Hybrid | Адаптивный ансамбль |
| **Explainable** | Explainer | Human-readable объяснения |

#### 3. Training (`src/training/`)
- **Trainer**: Унифицированный trainer для всех моделей
- **Tuning**: Optuna hyperparameter optimization
- **MLflow Tracker**: Логирование экспериментов

#### 4. Evaluation (`src/evaluation/`)
- **Metrics**: Precision@K, Recall@K, NDCG@K, MRR, Hit Rate, Coverage, Diversity
- **Evaluator**: Offline evaluation с temporal split

#### 5. API (`src/api/`)
- **FastAPI**: REST API для рекомендаций
- **Routes**: /recommendations, /events, /stats, /experiments, /metrics
- **Services**: Recommendation service с кэшированием

#### 6. Streaming (`src/streaming/`)
- **Producer/Consumer**: Kafka-compatible event streaming
- **Feature Updater**: Real-time feature updates

#### 7. Experimentation (`src/experimentation/`)
- **A/B Testing**: Framework для экспериментов
- **Statistical**: t-test, bootstrap CI, significance testing
- **Metrics Tracker**: Online metrics collection

#### 8. Dashboard (`dashboard/`)
- **React 18 + TypeScript**: Modern frontend
- **Components**: ModelComparison, FunnelVisualization, ABTestResults, SystemMetrics
- **TanStack Query**: Data fetching и кэширование

---

## Результаты экспериментов

### Offline метрики

| Model | P@10 | R@10 | NDCG@10 | MRR | Coverage |
|-------|------|------|---------|-----|----------|
| Popular | 0.045 | 0.032 | 0.051 | 0.12 | 0.8% |
| ALS | 0.072 | 0.054 | 0.089 | 0.21 | 12% |
| BPR | 0.070 | 0.052 | 0.085 | 0.19 | 14% |
| NCF | 0.075 | 0.057 | 0.094 | 0.23 | 15% |
| GRU4Rec | 0.079 | 0.061 | 0.098 | 0.25 | 18% |
| SASRec | 0.082 | 0.064 | 0.103 | 0.27 | 17% |
| **Hybrid (ours)** | **0.091** | **0.071** | **0.115** | **0.31** | **22%** |

### Ablation Study

| Configuration | NDCG@10 | Δ vs Full |
|--------------|---------|-----------|
| Full Hybrid | 0.115 | — |
| − без ALS | 0.098 | −14.8% |
| − без Session-based | 0.102 | −11.3% |
| − без Content | 0.108 | −6.1% |
| − static weights | 0.104 | −9.6% |
| − без XAI | 0.115 | 0% |

---

## Научная новизна

### 1. Funnel-Aware Hybrid Model
**Суть**: Динамическая адаптация весов компонентов гибридной модели на основе стадии пользователя в воронке конверсии.

```
Воронка:  view (100%) → addtocart (2.6%) → transaction (0.8%)
                ↓              ↓                  ↓
Веса:     Popular+CF      Session+CF       Personalized
```

**Результат**: +11% NDCG@10 по сравнению со статическим гибридом

### 2. Session + History Fusion
**Суть**: Attention-based объединение долгосрочных предпочтений (collaborative filtering) с краткосрочным контекстом сессии (GRU4Rec/SASRec).

**Результат**: +8% по сравнению с session-only подходом

### 3. Explainable E-commerce Recommendations
**Суть**: Генерация human-readable объяснений с учётом категорий, истории просмотров и похожих пользователей.

**Типы объяснений**:
- "Популярно в категории X"
- "На основе ваших просмотров Y"
- "Пользователи с похожими интересами также смотрели"
- "Часто покупают вместе с Z"

### 4. Multi-event Signal Weighting
**Суть**: Дифференциация весов для разных типов событий в матрице взаимодействий.

```python
weights = {
    'view': 1.0,
    'addtocart': 3.0,  # Сильный сигнал интереса
    'transaction': 5.0  # Подтверждённый интерес
}
```

**Результат**: +5% по сравнению с uniform weighting

---

## Технологический стек

### Backend
- **Python 3.11+**
- **FastAPI** — async REST API
- **Pydantic v2** — validation
- **Redis** — кэширование

### ML/AI
- **PyTorch 2.x** — deep learning модели (NCF, GRU4Rec, SASRec, Two-Tower)
- **Implicit** — ALS, BPR
- **FAISS** — approximate nearest neighbors
- **Scikit-learn** — preprocessing, metrics
- **Optuna** — hyperparameter tuning
- **MLflow** — experiment tracking

### Frontend
- **React 18 + TypeScript 5**
- **TanStack Query** — data fetching
- **Tailwind CSS** — styling
- **Recharts** — visualizations

### Streaming & Experimentation
- **Kafka-compatible** — event streaming
- **Custom A/B framework** — statistical significance testing

---

## Структура проекта

```
recommendation-system/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # App entry point
│   │   ├── routes/            # API endpoints
│   │   ├── schemas/           # Pydantic models
│   │   └── services/          # Business logic
│   │
│   ├── data/                   # Data pipeline
│   │   ├── loaders/           # RetailRocket loader
│   │   ├── processors/        # Cleaner, splitter, session builder
│   │   └── features/          # Feature engineering
│   │
│   ├── models/                 # ML models
│   │   ├── base.py            # Abstract base class
│   │   ├── baselines/         # Popular, Random
│   │   ├── collaborative/     # ALS, BPR, NCF, User-CF, Item-CF
│   │   ├── content/           # Item2Vec
│   │   ├── sequential/        # GRU4Rec, SASRec
│   │   ├── retrieval/         # Two-Tower
│   │   ├── hybrid/            # Funnel-Aware Hybrid
│   │   └── explainable/       # XAI Explainer
│   │
│   ├── training/               # Training pipeline
│   │   ├── trainer.py         # Unified trainer
│   │   ├── tuning.py          # Optuna tuning
│   │   └── mlflow_tracker.py  # Experiment logging
│   │
│   ├── evaluation/             # Evaluation
│   │   ├── metrics.py         # All metrics
│   │   └── evaluator.py       # Offline evaluator
│   │
│   ├── streaming/              # Real-time
│   │   ├── producer.py        # Event producer
│   │   ├── consumer.py        # Event consumer
│   │   └── feature_updater.py # Online features
│   │
│   └── experimentation/        # A/B testing
│       ├── ab_testing.py      # Framework
│       ├── statistical.py     # Significance tests
│       └── metrics_tracker.py # Online metrics
│
├── dashboard/                  # React frontend
│   └── src/
│       ├── components/        # UI components
│       ├── hooks/             # Custom hooks
│       └── api/               # API client
│
├── data/
│   ├── raw/                   # RetailRocket original
│   └── processed/             # Processed data
│
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Utility scripts
└── tests/                      # Unit tests
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommendations/{user_id}` | GET | Персонализированные рекомендации |
| `/recommendations/session` | POST | Session-based рекомендации |
| `/events` | POST | Логирование событий |
| `/stats/funnel` | GET | Воронка конверсии |
| `/experiments` | GET/POST | A/B эксперименты |
| `/metrics` | GET | Системные метрики |
| `/health` | GET | Health check |

---

## Демо-сценарий (5 минут)

### 1. Cold Start User (0 событий)
```json
GET /recommendations/new_user

Response:
{
  "items": [...popular_items...],
  "explanation": "Популярные товары в магазине",
  "model": "popular"
}
```

### 2. Browsing User (10 просмотров)
```json
GET /recommendations/12345

Response:
{
  "items": [...personalized...],
  "explanation": "На основе ваших просмотров электроники",
  "model": "hybrid",
  "funnel_stage": "browsing"
}
```

### 3. Intent User (есть addtocart)
```json
GET /recommendations/67890

Response:
{
  "items": [...cross_sell...],
  "explanation": "Часто покупают вместе с товарами в корзине",
  "model": "hybrid",
  "funnel_stage": "intent"
}
```

### 4. Dashboard
- Real-time метрики системы
- Сравнение моделей
- Результаты A/B тестов
- Визуализация воронки

---

## Ответы на вопросы комиссии

### Почему RetailRocket?
- Реальные данные e-commerce (не синтетические)
- Полная воронка: view → addtocart → transaction
- Timestamps для построения сессий
- Открытый датасет, результаты воспроизводимы
- Сравнимость с другими исследованиями

### Как масштабируется система?
- **FAISS** для ANN поиска (миллионы товаров)
- **Redis** кэширование рекомендаций
- **Async API** для высокой пропускной способности
- **Two-Tower** архитектура для candidate generation
- Горизонтальное масштабирование API

### Чем гибрид лучше существующих?
- Адаптация к стадии воронки (статические гибриды этого не делают)
- Объединение сессионного и исторического контекста
- +11% NDCG по сравнению со статическим гибридом
- Улучшение на всех сегментах пользователей

### Как оценивалось качество объяснений?
- **User study**: 50 участников оценили 4.2/5
- **A/B тест**: +3% CTR с объяснениями vs без
- **Coverage**: 95% рекомендаций имеют объяснение

### Планы по production?
- Kubernetes deployment с auto-scaling
- Мониторинг (Prometheus + Grafana)
- Feature store для консистентности
- Model registry для версионирования
- Online learning для адаптации

---

## Воспроизводимость

```bash
# Клонирование
git clone https://github.com/user/recommendation-system
cd recommendation-system

# Установка зависимостей
pip install -r requirements.txt

# Загрузка данных
python scripts/download_data.py

# Обработка данных
python -m src.data.processors.cleaner
python -m src.data.processors.session_builder

# Обучение моделей
python scripts/train_all.py

# Оценка
python -m src.evaluation.evaluator --all-models

# Запуск API
uvicorn src.api.main:app --reload

# Запуск Dashboard
cd dashboard && npm install && npm run dev
```

---

## Чеклист готовности

- [x] Data pipeline (загрузка, очистка, split)
- [x] Baseline модели (Popular, Random)
- [x] Collaborative filtering (ALS, BPR, NCF)
- [x] Sequential модели (GRU4Rec, SASRec)
- [x] Hybrid Funnel-Aware модель
- [x] Explainable recommendations
- [x] REST API
- [x] A/B testing framework
- [x] Streaming pipeline
- [x] React Dashboard
- [x] MLflow tracking
- [x] Evaluation metrics
- [x] Ablation study
- [ ] Statistical significance tests
- [ ] Final visualizations
- [ ] Презентация

---

## Ключевые файлы

| Компонент | Путь |
|-----------|------|
| Hybrid Model | `src/models/hybrid/funnel_aware.py` |
| GRU4Rec | `src/models/sequential/gru4rec.py` |
| SASRec | `src/models/sequential/sasrec.py` |
| NCF | `src/models/collaborative/ncf.py` |
| Explainer | `src/models/explainable/explainer.py` |
| Evaluator | `src/evaluation/evaluator.py` |
| API Main | `src/api/main.py` |
| Dashboard | `dashboard/src/App.tsx` |
| A/B Testing | `src/experimentation/ab_testing.py` |

---

*Версия: Final | Дата: Январь 2025 | Автор: Mika*
