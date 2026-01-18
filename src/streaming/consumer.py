"""Kafka Event Consumer for real-time event processing.

Consumes user behavior events and processes them for feature updates,
model retraining triggers, and analytics.
"""

import json
import signal
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable

from loguru import logger

# Optional Kafka import
try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python not installed. Install with: pip install kafka-python")


class StreamProcessor(ABC):
    """Abstract base class for stream processors.

    Implement this class to define custom event processing logic.
    """

    @abstractmethod
    def process(self, event: dict) -> None:
        """Process a single event.

        Args:
            event: Event dictionary.
        """
        pass

    def on_batch_complete(self, batch_size: int) -> None:
        """Called after processing a batch of events.

        Args:
            batch_size: Number of events processed.
        """
        pass

    def on_error(self, event: dict, error: Exception) -> None:
        """Handle processing error.

        Args:
            event: Event that caused error.
            error: Exception raised.
        """
        logger.error(f"Error processing event: {error}")


class UserEventProcessor(StreamProcessor):
    """Processor for user behavior events.

    Updates user statistics and triggers feature updates.
    """

    def __init__(
        self,
        feature_updater: "RealTimeFeatureUpdater | None" = None,
        on_event_callback: Callable[[dict], None] | None = None,
    ):
        """Initialize processor.

        Args:
            feature_updater: Feature updater instance.
            on_event_callback: Callback for each processed event.
        """
        self.feature_updater = feature_updater
        self.on_event_callback = on_event_callback

        # Statistics
        self.stats = defaultdict(int)
        self._lock = threading.Lock()

    def process(self, event: dict) -> None:
        """Process user behavior event.

        Args:
            event: User event dictionary.
        """
        event_type = event.get("event", "unknown")

        with self._lock:
            self.stats["total_events"] += 1
            self.stats[f"event_{event_type}"] += 1

        # Update features if updater available
        if self.feature_updater:
            self.feature_updater.update_from_event(event)

        # Call custom callback if provided
        if self.on_event_callback:
            self.on_event_callback(event)

        logger.debug(f"Processed {event_type} event for user {event.get('visitor_id')}")

    def on_batch_complete(self, batch_size: int) -> None:
        """Log batch completion."""
        with self._lock:
            self.stats["batches_processed"] += 1

        logger.info(
            f"Batch complete: {batch_size} events, "
            f"total: {self.stats['total_events']}"
        )

    def get_stats(self) -> dict[str, int]:
        """Get processing statistics."""
        with self._lock:
            return dict(self.stats)

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self.stats.clear()


class RecommendationEventProcessor(StreamProcessor):
    """Processor for recommendation served events.

    Tracks recommendation performance and A/B test metrics.
    """

    def __init__(self, ab_test_manager: "ABTestManager | None" = None):
        """Initialize processor.

        Args:
            ab_test_manager: A/B test manager for logging metrics.
        """
        self.ab_test_manager = ab_test_manager
        self.stats = defaultdict(int)
        self._model_stats: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()

    def process(self, event: dict) -> None:
        """Process recommendation served event.

        Args:
            event: Recommendation event dictionary.
        """
        model_name = event.get("model_name", "unknown")
        n_items = len(event.get("recommended_items", []))

        with self._lock:
            self.stats["total_recommendations"] += 1
            self._model_stats[model_name]["count"] += 1
            self._model_stats[model_name]["total_items"] += n_items

        logger.debug(
            f"Processed recommendation event: model={model_name}, items={n_items}"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get recommendation statistics."""
        with self._lock:
            return {
                "total": self.stats["total_recommendations"],
                "by_model": dict(self._model_stats),
            }


class EventConsumer:
    """Kafka consumer for event streams.

    Consumes events from Kafka topics and dispatches to processors.
    Supports graceful shutdown and batch processing.
    """

    def __init__(
        self,
        topics: list[str],
        bootstrap_servers: str | list[str] = "localhost:9092",
        group_id: str = "recsys-consumer-group",
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
        auto_commit_interval_ms: int = 5000,
        max_poll_records: int = 500,
        session_timeout_ms: int = 30000,
    ):
        """Initialize Kafka consumer.

        Args:
            topics: List of topics to subscribe to.
            bootstrap_servers: Kafka broker addresses.
            group_id: Consumer group ID.
            auto_offset_reset: Where to start reading (earliest/latest).
            enable_auto_commit: Enable automatic offset commits.
            auto_commit_interval_ms: Interval between auto commits.
            max_poll_records: Maximum records per poll.
            session_timeout_ms: Consumer session timeout.
        """
        self.topics = topics
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id

        self._consumer: KafkaConsumer | None = None
        self._processors: dict[str, StreamProcessor] = {}
        self._running = False
        self._thread: threading.Thread | None = None

        self._consumer_config = {
            "bootstrap_servers": bootstrap_servers,
            "group_id": group_id,
            "auto_offset_reset": auto_offset_reset,
            "enable_auto_commit": enable_auto_commit,
            "auto_commit_interval_ms": auto_commit_interval_ms,
            "max_poll_records": max_poll_records,
            "session_timeout_ms": session_timeout_ms,
            "value_deserializer": lambda v: json.loads(v.decode("utf-8")),
            "key_deserializer": lambda k: k.decode("utf-8") if k else None,
        }

    def register_processor(self, topic: str, processor: StreamProcessor) -> None:
        """Register a processor for a topic.

        Args:
            topic: Kafka topic name.
            processor: StreamProcessor instance.
        """
        self._processors[topic] = processor
        logger.info(f"Registered processor for topic: {topic}")

    def start(self, blocking: bool = True) -> None:
        """Start consuming events.

        Args:
            blocking: If True, block the current thread.
        """
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available. Install kafka-python.")
            return

        if self._running:
            logger.warning("Consumer is already running")
            return

        self._running = True

        if blocking:
            self._consume_loop()
        else:
            self._thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._thread.start()
            logger.info("Consumer started in background thread")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop consuming events.

        Args:
            timeout: Maximum time to wait for shutdown.
        """
        logger.info("Stopping consumer...")
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        if self._consumer:
            self._consumer.close()
            self._consumer = None

        logger.info("Consumer stopped")

    def _consume_loop(self) -> None:
        """Main consumption loop."""
        try:
            self._consumer = KafkaConsumer(*self.topics, **self._consumer_config)
            logger.info(f"Started consuming from topics: {self.topics}")

            while self._running:
                # Poll for messages
                messages = self._consumer.poll(timeout_ms=1000)

                for topic_partition, records in messages.items():
                    topic = topic_partition.topic
                    processor = self._processors.get(topic)

                    if not processor:
                        logger.warning(f"No processor for topic: {topic}")
                        continue

                    for record in records:
                        try:
                            processor.process(record.value)
                        except Exception as e:
                            processor.on_error(record.value, e)

                    if records:
                        processor.on_batch_complete(len(records))

        except KafkaError as e:
            logger.error(f"Kafka consumer error: {e}")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            if self._consumer:
                self._consumer.close()
                self._consumer = None

    @property
    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._running


class MockEventConsumer(EventConsumer):
    """Mock consumer for testing without Kafka.

    Allows manual event injection for testing processors.
    """

    def __init__(self, topics: list[str], **kwargs):
        """Initialize mock consumer."""
        super().__init__(topics, **kwargs)
        self._event_queue: list[tuple[str, dict]] = []

    def start(self, blocking: bool = True) -> None:
        """Start mock consumer (processes queued events)."""
        self._running = True

        while self._event_queue and self._running:
            topic, event = self._event_queue.pop(0)
            processor = self._processors.get(topic)
            if processor:
                processor.process(event)

        self._running = False

    def inject_event(self, topic: str, event: dict) -> None:
        """Inject an event for testing.

        Args:
            topic: Topic name.
            event: Event dictionary.
        """
        self._event_queue.append((topic, event))

    def process_event(self, topic: str, event: dict) -> None:
        """Directly process an event.

        Args:
            topic: Topic name.
            event: Event dictionary.
        """
        processor = self._processors.get(topic)
        if processor:
            processor.process(event)
        else:
            logger.warning(f"No processor for topic: {topic}")


# Import for type hints (avoid circular import)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.streaming.feature_updater import RealTimeFeatureUpdater
    from src.experimentation.ab_testing import ABTestManager
