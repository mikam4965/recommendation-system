# RecSys Dashboard

Admin dashboard for the Recommendation System.

## Features

- **Overview**: System metrics and performance summary
- **Model Comparison**: Compare metrics across all recommendation models
- **A/B Experiments**: Manage and analyze A/B tests
- **Funnel Analysis**: User journey and conversion metrics

## Tech Stack

- React 18 + TypeScript
- TanStack Query (React Query) for data fetching
- Recharts for visualizations
- Tailwind CSS for styling
- Vite for build

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Development

The dashboard connects to the backend API at `/api` by default.
Configure the API URL using the `VITE_API_URL` environment variable.

```bash
# .env.local
VITE_API_URL=http://localhost:8000/api
```

## Components

- `SystemMetrics` - Key performance indicators
- `ModelComparison` - Bar chart comparing model metrics
- `ModelMetricsTable` - Detailed metrics table
- `FunnelVisualization` - Conversion funnel display
- `ABTestResults` - A/B test management and analysis

## API Endpoints

The dashboard uses the following API endpoints:

- `GET /v1/metrics/models` - Model performance metrics
- `GET /v1/metrics/system` - System-level metrics
- `GET /v1/metrics/funnel` - Conversion funnel data
- `GET /v1/experiments` - List experiments
- `GET /v1/experiments/{name}/analysis` - Statistical analysis
- `POST /v1/experiments/{name}/start` - Start experiment
- `POST /v1/experiments/{name}/stop` - Stop experiment
