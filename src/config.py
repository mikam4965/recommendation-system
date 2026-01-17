"""Project configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    data_interim_dir: Path = project_root / "data" / "interim"
    models_dir: Path = project_root / "models"

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "recsys"
    postgres_password: str = "recsys_pass"
    postgres_db: str = "recsys"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_http_port: int = 8123
    clickhouse_native_port: int = 9000
    clickhouse_user: str = "recsys"
    clickhouse_password: str = "recsys_pass"
    clickhouse_db: str = "recsys"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True

    # Data processing
    session_timeout_minutes: int = 30
    min_item_interactions: int = 5
    max_events_per_day: int = 1000  # Bot threshold

    # Event weights for collaborative filtering
    event_weight_view: float = 1.0
    event_weight_addtocart: float = 2.0
    event_weight_transaction: float = 3.0

    # Train/Val/Test split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    @property
    def postgres_url(self) -> str:
        """PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


settings = Settings()
