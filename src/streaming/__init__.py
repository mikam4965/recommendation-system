"""Real-time streaming module for event processing.

Components:
- EventProducer: Produces events to Kafka
- EventConsumer: Consumes and processes events
- FeatureUpdater: Updates user/item features in real-time
"""

from src.streaming.producer import EventProducer
from src.streaming.consumer import EventConsumer, StreamProcessor
from src.streaming.feature_updater import RealTimeFeatureUpdater

__all__ = [
    "EventProducer",
    "EventConsumer",
    "StreamProcessor",
    "RealTimeFeatureUpdater",
]
