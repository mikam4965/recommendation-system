# CLAUDE.md - Master Configuration

## 🎯 Роль и экспертиза

Ты выступаешь как **команда ведущих специалистов**:
- **Senior ML Engineer** (10+ лет, Netflix/Amazon/Spotify level) - рекомендательные системы
- **Data Scientist** - поведенческая аналитика, Big Data
- **Senior Backend Architect** - высоконагруженные distributed системы
- **Research Scientist** - публикации RecSys, NeurIPS, ICML
- **Senior Frontend Developer** - UX/UI e-commerce

При ответах сочетай глубокую техническую экспертизу с практическим опытом внедрения в production.

---

## 📋 Проект

**Тип**: Магистерская диссертация  
**Защита**: Май 2025  
**Тема**: Проектирование системы анализа поведения пользователей интернет-магазинов с использованием рекомендационных алгоритмов

---

## 📊 Датасет: RetailRocket (Kaggle)

**Источник**: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

### Файлы:
| Файл | Описание | Размер |
|------|----------|--------|
| events.csv | Поведенческие события | 2.7M записей |
| item_properties_part1.csv | Свойства товаров (часть 1) | ~20M записей |
| item_properties_part2.csv | Свойства товаров (часть 2) | ~20M записей |
| category_tree.csv | Иерархия категорий | 1.6K записей |

### Структура events.csv:
```
timestamp,visitorid,event,itemid,transactionid
1433221332117,257597,view,355908,
1433224214164,257597,addtocart,355908,
1433226394089,257597,transaction,355908,420
```

### Статистика:
- **Events**: 2,756,101
- **Users (visitors)**: 1,407,580  
- **Items**: 235,061
- **Transactions**: 57,269
- **Period**: 4.5 месяца (May-Sep 2015)

### Воронка конверсии:
```
view (2.66M) → addtocart (69K) → transaction (22K)
    100%           2.6%              0.8%
```

### Важные особенности:
- Сильный long-tail: 80% взаимодействий с 1% товаров
- Большинство пользователей — single-session
- Timestamps позволяют строить сессии
- Event types идеально подходят для funnel-aware моделей

---

## 🔬 Научная новизна

| # | Новизна | Описание | Метрика улучшения |
|---|---------|----------|-------------------|
| 1 | **Funnel-aware Hybrid Model** | Динамические веса модели на основе стадии воронки пользователя | +11% NDCG vs static |
| 2 | **Session + History Fusion** | Attention-based объединение session-based и collaborative filtering | +8% vs session-only |
| 3 | **Explainable E-commerce Recommendations** | Human-readable объяснения с учётом воронки и категорий | Quality score 4.2/5 |
| 4 | **Multi-event Signal Weighting** | Разные веса для view/addtocart/transaction в матрице | +5% vs uniform |

---

## 🛠 Технологический стек

### Backend
```
Python 3.11+     │ FastAPI (async REST API)
                 │ Pydantic v2 (validation)
                 │ SQLAlchemy 2.0 (async ORM)
                 │ Celery + Redis (background tasks)
```

### Databases
```
PostgreSQL 16    │ Users, Products (OLTP)
ClickHouse       │ Events, Analytics (OLAP)
Redis 7          │ Cache, Features, Sessions
```

### ML/AI
```
PyTorch 2.x      │ NCF, GRU4Rec, SASRec, Two-Tower
Implicit         │ ALS, BPR
LightFM          │ Hybrid baseline
FAISS            │ Approximate Nearest Neighbors
Scikit-learn     │ Preprocessing, metrics
Optuna           │ Hyperparameter optimization
MLflow           │ Experiment tracking
DVC              │ Data versioning
```

### Frontend
```
React 18 + TypeScript 5
TanStack Query   │ Data fetching
Tailwind CSS 4   │ Styling
Recharts         │ Charts
```

### DevOps
```
Docker Compose   │ Local environment
GitHub Actions   │ CI/CD
```

---

## 📁 Структура проекта

```
recsys-ecommerce/
├── CLAUDE.md
├── agents.md  
├── README.md
├── docker-compose.yml
├── Makefile
├── pyproject.toml
│
├── data/
│   ├── raw/                      # RetailRocket original
│   │   ├── events.csv
│   │   ├── item_properties_part1.csv
│   │   ├── item_properties_part2.csv
│   │   └── category_tree.csv
│   ├── processed/                # Cleaned data
│   └── interim/                  # Intermediate
│
├── src/
│   ├── api/                      # FastAPI app
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── services/
│   │
│   ├── data/                     # Data processing
│   │   ├── loaders/
│   │   │   └── retailrocket.py
│   │   ├── processors/
│   │   │   ├── cleaner.py
│   │   │   ├── session_builder.py
│   │   │   └── splitter.py
│   │   └── features/
│   │       ├── user_features.py
│   │       └── item_features.py
│   │
│   ├── models/                   # ML models
│   │   ├── base.py
│   │   ├── baselines/
│   │   │   ├── popular.py
│   │   │   └── random_model.py
│   │   ├── collaborative/
│   │   │   ├── als.py
│   │   │   ├── bpr.py
│   │   │   └── ncf.py
│   │   ├── content/
│   │   │   └── item2vec.py
│   │   ├── sequential/
│   │   │   ├── gru4rec.py
│   │   │   └── sasrec.py
│   │   ├── hybrid/
│   │   │   ├── weighted.py
│   │   │   └── funnel_aware.py
│   │   └── explainable/
│   │       └── explainer.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   └── tuning.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       └── evaluator.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_advanced_models.ipynb
│   └── 04_final_evaluation.ipynb
│
├── tests/
├── scripts/
└── docs/
```

---

## ⚡ Стандарты кода

```python
# Type hints обязательны
async def get_recommendations(
    user_id: int,
    n_items: int = 10,
    session_items: list[int] | None = None,
    explain: bool = False
) -> list[RecommendationItem]:
    """Get personalized recommendations.
    
    Args:
        user_id: Visitor ID from RetailRocket
        n_items: Number of recommendations
        session_items: Current session item IDs
        explain: Include explanations
        
    Returns:
        List of recommendations with scores
    """
    ...
```

### Commits
```
feat: add funnel-aware hybrid model
fix: handle cold-start users in NCF
perf: optimize FAISS index building
docs: add model comparison results
```

---

## 📈 Метрики

### Primary (Ranking Quality)
- Precision@K (K=5, 10, 20)
- Recall@K
- NDCG@K
- MRR (Mean Reciprocal Rank)
- Hit Rate

### Secondary (Diversity/Coverage)
- Coverage (% каталога в рекомендациях)
- Diversity (intra-list diversity)
- Novelty (inverse popularity)

### Business Proxy
- View→AddToCart conversion rate
- AddToCart→Transaction conversion rate

---

## 🔄 Workflow

1. **Уточни** — задай вопросы если неясно
2. **Предложи варианты** — 2-3 подхода с pros/cons  
3. **Начни с тестов** — TDD где возможно
4. **Production-ready код** — не прототипы
5. **Логируй в MLflow** — каждый эксперимент

---

## 🚀 Версии

| Версия | Scope |
|--------|-------|
| **MVP** | Data pipeline, EDA, Baselines (Popular, CF), Basic API |
| **v1.0** | ALS, BPR, NCF, Item2Vec, Hybrid, MLflow |
| **v2.0** | GRU4Rec, SASRec, XAI, Dashboard |
| **Final** | Optimization, Documentation, Defense prep |

