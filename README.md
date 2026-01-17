# RecSys E-commerce: Recommendation System for RetailRocket

A recommendation system for e-commerce based on the RetailRocket dataset, developed as part of a Master's thesis project.

## Overview

This project implements a complete recommendation pipeline including:
- Data processing and session building
- Baseline models (Popular, Random)
- Collaborative filtering models (User-CF, Item-CF)
- Evaluation metrics (Precision, Recall, NDCG, MRR, Hit Rate)
- REST API for serving recommendations

## Dataset

**RetailRocket E-commerce Dataset** from [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

| Metric | Value |
|--------|-------|
| Total Events | ~2.7M |
| Unique Users | ~1.4M |
| Unique Items | ~235K |
| Period | 4.5 months |

Event types: `view`, `addtocart`, `transaction`

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker & Docker Compose

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd recommendation-system

# Install dependencies
make install

# Install with dev dependencies
make dev-install

# Start Docker services (PostgreSQL, Redis, ClickHouse)
make docker-up
```

### Data Setup

1. Download the RetailRocket dataset from [Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
2. Place the zip file in `data/raw/`
3. Extract and process:

```bash
# Extract the dataset
make extract-data

# Run data processing pipeline
make process-data
```

### Training & Evaluation

```bash
# Train baseline models
make train-baseline

# Evaluate all models
make evaluate
```

### Running the API

```bash
# Start the API server
make api
```

The API will be available at `http://localhost:8000`

- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/recommendations/{user_id}` | GET | Get recommendations for user |
| `/recommendations/` | GET | List available models |
| `/events/` | POST | Record new event |
| `/stats/user/{user_id}` | GET | Get user statistics |
| `/stats/summary` | GET | Get dataset summary |

Example request:
```bash
curl "http://localhost:8000/recommendations/257597?n=5&model=popular"
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov
```

### Running Jupyter Notebook

```bash
make notebook
```

Open `notebooks/01_eda_retailrocket.ipynb` for exploratory data analysis.

## Project Structure

```
recommendation-system/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── routes/             # API endpoints
│   │   ├── schemas/            # Pydantic models
│   │   └── services/           # Business logic
│   ├── data/
│   │   ├── loaders/            # Data loading utilities
│   │   └── processors/         # Data processing
│   ├── models/
│   │   ├── baselines/          # Popular, Random
│   │   └── collaborative/      # User-CF, Item-CF
│   └── evaluation/             # Metrics and evaluator
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # CLI scripts
├── tests/                      # Unit and integration tests
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Processed parquet files
└── models/                     # Trained model artifacts
```

## Models

### Implemented

| Model | Description |
|-------|-------------|
| **Popular** | Recommends items by weighted popularity (view=1, cart=2, purchase=3) |
| **Random** | Random baseline for comparison |
| **User-CF** | User-based collaborative filtering with cosine similarity |
| **Item-CF** | Item-based collaborative filtering with cosine similarity |

### Expected Performance

| Model | Precision@10 | Hit Rate | MRR |
|-------|-------------|----------|-----|
| Random | ~0.001 | ~0.01 | ~0.005 |
| Popular | ~0.02-0.05 | ~0.15-0.25 | ~0.08-0.12 |
| User-CF | ~0.03-0.06 | ~0.18-0.30 | ~0.10-0.15 |
| Item-CF | ~0.04-0.08 | ~0.20-0.35 | ~0.12-0.18 |

## Evaluation Metrics

- **Precision@K**: Fraction of relevant items in top-K
- **Recall@K**: Fraction of relevant items found in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Hit Rate**: Whether at least one relevant item was recommended
- **Coverage**: Fraction of catalog items recommended
- **Novelty**: Inverse popularity of recommended items

## Configuration

Environment variables (`.env` file):

```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# API
API_HOST=0.0.0.0
API_PORT=8000
```

See `.env.example` for all available options.

## Development

```bash
# Format code
make format

# Run linter
make lint

# Clean build artifacts
make clean
```

## Make Commands

| Command | Description |
|---------|-------------|
| `make setup` | Full setup (install + docker) |
| `make install` | Install dependencies |
| `make docker-up` | Start Docker services |
| `make docker-down` | Stop Docker services |
| `make extract-data` | Extract dataset from zip |
| `make process-data` | Run data processing pipeline |
| `make train-baseline` | Train all baseline models |
| `make evaluate` | Evaluate models and print metrics |
| `make api` | Run FastAPI server |
| `make test` | Run tests |
| `make notebook` | Start Jupyter Lab |
| `make lint` | Run linter |
| `make format` | Format code |

## License

MIT

## Author

Developed as part of a Master's thesis on "Designing a System for Analyzing User Behavior in Online Stores Using Recommendation Algorithms"
