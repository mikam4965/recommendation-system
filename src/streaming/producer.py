"""Kafka Event Producer for real-time event streaming.

Produces user behavior events to Kafka topics for downstream processing.
"""

import json
from datetime import datetime
from typing import Any

from loguru import logger
from pydantic import BaseModel

# Optional Kafka import
try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python not installed. Install with: pip install kafka-python")


class UserEvent(BaseModel):
    """User behavior event schema."""

    visitor_id: int
    item_id: int
    event: str  # view, addtocart, transaction
    timestamp: int  # Unix timestamp in milliseconds
    session_id: str | None = None
    transaction_id: int | None = None
    properties: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        visitor_id: int,
        item_id: int,
        event: str,
        session_id: str | None = None,
        transaction_id: int | None = None,
        properties: dict[str, Any] | None = None,
    ) -> "UserEvent":
        """Create event with current timestamp."""
        return cls(
            visitor_id=visitor_id,
            item_id=item_id,
            event=event,
            timestamp=int(datetime.now().timestamp() * 1000),
            session_id=session_id,
            transaction_id=transaction_id,
            properties=properties,
        )


class RecommendationEvent(BaseModel):
    """Recommendation served event for tracking."""

    visitor_id: int
    recommended_items: list[int]
    model_name: str
    timestamp: int
    session_id: str | None = None
    request_id: str | None = None
    context: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        visitor_id: int,
        recommended_items: list[int],
        model_name: str,
        session_id: str | None = None,
        request_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "RecommendationEvent":
        """Create event with current timestamp."""
        return cls(
            visitor_id=visitor_id,
            recommended_items=recommended_items,
            model_name=model_name,
            timestamp=int(datetime.now().timestamp() * 1000),
            session_id=session_id,
            request_id=request_id,
            context=context,
        )


class EventProducer:
    """Kafka producer for user behavior events.

    Sends events to Kafka topics for real-time processing.
    Supports both synchronous and asynchronous sending.

    Topics:
    - user-events: User behavior events (view, addtocart, transaction)
    - recommendation-events: Recommendations served for tracking
    - feature-updates: Feature update requests
    """

    # Topic names
    TOPIC_USER_EVENTS = "user-events"
    TOPIC_RECOMMENDATION_EVENTS = "recommendation-events"
    TOPIC_FEATURE_UPDATES = "feature-updates"

    def __init__(
        self,
        bootstrap_servers: str | list[str] = "localhost:9092",
        client_id: str = "recsys-producer",
        async_send: bool = True,
        batch_size: int = 16384,
        linger_ms: int = 10,
        compression_type: str = "gzip",
    ):
        """Initialize Kafka producer.

        Args:
            bootstrap_servers: Kafka broker addresses.
            client_id: Producer client ID.
            async_send: Use asynchronous sending.
            batch_size: Batch size in bytes.
            linger_ms: Time to wait for batch.
            compression_type: Compression type (gzip, snappy, lz4).
        """
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.async_send = async_send
        self._producer: KafkaProducer | None = None
        self._is_connected = False

        self._producer_config = {
            "bootstrap_servers": bootstrap_servers,
            "client_id": client_id,
            "value_serializer": lambda v: json.dumps(v).encode("utf-8"),
            "key_serializer": lambda k: str(k).encode("utf-8") if k else None,
            "batch_size": batch_size,
            "linger_ms": linger_ms,
            "compression_type": compression_type,
            "acks": "all",
            "retries": 3,
        }

    def connect(self) -> bool:
        """Connect to Kafka broker.

        Returns:
            True if connection successful.
        """
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available. Install kafka-python.")
            return False

        try:
            self._producer = KafkaProducer(**self._producer_config)
            self._is_connected = True
            logger.info(f"Connected to Kafka: {self.bootstrap_servers}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Kafka broker."""
        if self._producer:
            self._producer.flush()
            self._producer.close()
            self._producer = None
            self._is_connected = False
            logger.info("Disconnected from Kafka")

    def send_user_event(
        self,
        event: UserEvent,
        callback: callable | None = None,
    ) -> bool:
        """Send user behavior event to Kafka.

        Args:
            event: User event to send.
            callback: Optional callback for async send.

        Returns:
            True if send initiated successfully.
        """
        return self._send(
            topic=self.TOPIC_USER_EVENTS,
            key=event.visitor_id,
            value=event.model_dump(),
            callback=callback,
        )

    def send_recommendation_event(
        self,
        event: RecommendationEvent,
        callback: callable | None = None,
    ) -> bool:
        """Send recommendation served event.

        Args:
            event: Recommendation event to send.
            callback: Optional callback for async send.

        Returns:
            True if send initiated successfully.
        """
        return self._send(
            topic=self.TOPIC_RECOMMENDATION_EVENTS,
            key=event.visitor_id,
            value=event.model_dump(),
            callback=callback,
        )

    def send_feature_update(
        self,
        entity_type: str,
        entity_id: int,
        features: dict[str, Any],
        callback: callable | None = None,
    ) -> bool:
        """Send feature update request.

        Args:
            entity_type: Type of entity (user, item).
            entity_id: Entity ID.
            features: Feature dictionary to update.
            callback: Optional callback for async send.

        Returns:
            True if send initiated successfully.
        """
        value = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "features": features,
            "timestamp": int(datetime.now().timestamp() * 1000),
        }

        return self._send(
            topic=self.TOPIC_FEATURE_UPDATES,
            key=f"{entity_type}:{entity_id}",
            value=value,
            callback=callback,
        )

    def _send(
        self,
        topic: str,
        value: dict,
        key: Any = None,
        callback: callable | None = None,
    ) -> bool:
        """Internal send method.

        Args:
            topic: Kafka topic.
            value: Message value.
            key: Message key.
            callback: Callback for async send.

        Returns:
            True if send initiated successfully.
        """
        if not self._is_connected:
            if not self.connect():
                logger.error("Cannot send: not connected to Kafka")
                return False

        try:
            future = self._producer.send(topic, key=key, value=value)

            if self.async_send:
                if callback:
                    future.add_callback(callback)
                    future.add_errback(lambda e: logger.error(f"Send failed: {e}"))
            else:
                # Wait for send to complete
                future.get(timeout=10)

            return True

        except KafkaError as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            return False

    def flush(self, timeout: float | None = None) -> None:
        """Flush pending messages.

        Args:
            timeout: Maximum time to wait in seconds.
        """
        if self._producer:
            self._producer.flush(timeout=timeout)

    @property
    def is_connected(self) -> bool:
        """Check if producer is connected."""
        return self._is_connected

    def __enter__(self) -> "EventProducer":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


class MockEventProducer(EventProducer):
    """Mock producer for testing without Kafka.

    Stores events in memory for testing purposes.
    """

    def __init__(self, **kwargs):
        """Initialize mock producer."""
        super().__init__(**kwargs)
        self.events: dict[str, list[dict]] = {
            self.TOPIC_USER_EVENTS: [],
            self.TOPIC_RECOMMENDATION_EVENTS: [],
            self.TOPIC_FEATURE_UPDATES: [],
        }
        self._is_connected = True

    def connect(self) -> bool:
        """Mock connect."""
        self._is_connected = True
        return True

    def disconnect(self) -> None:
        """Mock disconnect."""
        self._is_connected = False

    def _send(
        self,
        topic: str,
        value: dict,
        key: Any = None,
        callback: callable | None = None,
    ) -> bool:
        """Store event in memory."""
        self.events[topic].append({"key": key, "value": value})
        if callback:
            callback(None)
        return True

    def get_events(self, topic: str) -> list[dict]:
        """Get stored events for a topic."""
        return self.events.get(topic, [])

    def clear_events(self) -> None:
        """Clear all stored events."""
        for topic in self.events:
            self.events[topic] = []
