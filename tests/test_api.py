"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_recommendation_service():
    """Mock recommendation service with loaded models."""
    with patch("src.api.routes.recommendations.recommendation_service") as mock:
        mock.is_loaded = True
        mock.get_available_models.return_value = ["popular", "random"]
        mock.get_recommendations.return_value = [
            (100, 10.5),
            (101, 9.3),
            (102, 8.1),
        ]
        yield mock


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestRecommendationsEndpoint:
    """Tests for recommendations endpoints."""

    def test_get_recommendations(self, client, mock_recommendation_service):
        """Test getting recommendations."""
        response = client.get("/recommendations/123?n=5&model=popular")
        assert response.status_code == 200

        data = response.json()
        assert data["user_id"] == 123
        assert data["model"] == "popular"
        assert len(data["recommendations"]) == 3
        assert data["recommendations"][0]["item_id"] == 100
        assert data["recommendations"][0]["rank"] == 1

    def test_get_recommendations_default_params(self, client, mock_recommendation_service):
        """Test recommendations with default parameters."""
        response = client.get("/recommendations/456")
        assert response.status_code == 200

        data = response.json()
        assert data["user_id"] == 456

    def test_get_recommendations_invalid_model(self, client, mock_recommendation_service):
        """Test recommendations with invalid model."""
        response = client.get("/recommendations/123?model=nonexistent")
        assert response.status_code == 400
        assert "not available" in response.json()["detail"]

    def test_get_recommendations_n_out_of_range(self, client, mock_recommendation_service):
        """Test recommendations with n out of range."""
        response = client.get("/recommendations/123?n=0")
        assert response.status_code == 422  # Validation error

        response = client.get("/recommendations/123?n=101")
        assert response.status_code == 422

    def test_get_available_models(self, client, mock_recommendation_service):
        """Test getting available models list."""
        response = client.get("/recommendations/")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "default" in data
        assert "popular" in data["models"]


class TestEventsEndpoint:
    """Tests for events endpoints."""

    def test_create_event_view(self, client):
        """Test creating a view event."""
        event_data = {
            "visitor_id": 123,
            "item_id": 456,
            "event": "view",
        }
        response = client.post("/events/", json=event_data)
        assert response.status_code == 201

        data = response.json()
        assert data["visitor_id"] == 123
        assert data["item_id"] == 456
        assert data["event"] == "view"
        assert "id" in data
        assert "timestamp" in data

    def test_create_event_transaction(self, client):
        """Test creating a transaction event with transaction_id."""
        event_data = {
            "visitor_id": 123,
            "item_id": 456,
            "event": "transaction",
            "transaction_id": 789,
        }
        response = client.post("/events/", json=event_data)
        assert response.status_code == 201

        data = response.json()
        assert data["transaction_id"] == 789

    def test_create_transaction_without_id_fails(self, client):
        """Test that transaction without transaction_id fails."""
        event_data = {
            "visitor_id": 123,
            "item_id": 456,
            "event": "transaction",
        }
        response = client.post("/events/", json=event_data)
        assert response.status_code == 400
        assert "transaction_id is required" in response.json()["detail"]

    def test_create_event_invalid_type(self, client):
        """Test creating event with invalid type."""
        event_data = {
            "visitor_id": 123,
            "item_id": 456,
            "event": "invalid_event",
        }
        response = client.post("/events/", json=event_data)
        assert response.status_code == 422  # Validation error

    def test_get_event_count(self, client):
        """Test getting event count."""
        # Create some events first
        for i in range(3):
            client.post("/events/", json={
                "visitor_id": i,
                "item_id": 100,
                "event": "view",
            })

        response = client.get("/events/count")
        assert response.status_code == 200

        data = response.json()
        assert "count" in data
        assert data["count"] >= 3


class TestStatsEndpoint:
    """Tests for stats endpoints."""

    def test_get_summary_no_data(self, client):
        """Test summary when data not available."""
        # This will fail if data files don't exist
        response = client.get("/stats/summary")
        # Either 200 (if data exists) or 503 (if not)
        assert response.status_code in [200, 503]

    def test_get_user_stats_not_found(self, client):
        """Test user stats for non-existent user."""
        response = client.get("/stats/user/999999999")
        # Either 404 (user not found) or 503 (data not available)
        assert response.status_code in [404, 503]
