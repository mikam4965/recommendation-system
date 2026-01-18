"""Real-time feature updater for streaming events.

Updates user and item features in Redis as events arrive.
"""

import json
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any

from loguru import logger

# Optional Redis import
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not installed. Install with: pip install redis")


class RealTimeFeatureUpdater:
    """Updates user and item features in real-time from event streams.

    Features are stored in Redis for fast access during inference.
    Supports incremental updates for counters and aggregations.

    Feature Keys:
    - user:{user_id}:features - User feature hash
    - user:{user_id}:recent_items - Recent item list
    - user:{user_id}:session:{session_id} - Session items
    - item:{item_id}:features - Item feature hash
    - item:{item_id}:recent_users - Recent user list
    """

    # Redis key prefixes
    PREFIX_USER = "user"
    PREFIX_ITEM = "item"
    PREFIX_SESSION = "session"

    # Feature TTLs (in seconds)
    TTL_USER_FEATURES = 86400 * 30  # 30 days
    TTL_SESSION = 3600  # 1 hour
    TTL_RECENT_ITEMS = 86400 * 7  # 7 days

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str | None = None,
        max_recent_items: int = 100,
        max_recent_users: int = 1000,
        event_weights: dict[str, float] | None = None,
    ):
        """Initialize feature updater.

        Args:
            redis_host: Redis host.
            redis_port: Redis port.
            redis_db: Redis database number.
            redis_password: Redis password.
            max_recent_items: Max recent items to track per user.
            max_recent_users: Max recent users to track per item.
            event_weights: Weights for different event types.
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.max_recent_items = max_recent_items
        self.max_recent_users = max_recent_users

        # Event weights for scoring
        self.event_weights = event_weights or {
            "view": 1.0,
            "addtocart": 3.0,
            "transaction": 5.0,
        }

        self._redis: redis.Redis | None = None
        self._is_connected = False
        self._lock = threading.Lock()

        # Statistics
        self.stats = defaultdict(int)

    def connect(self) -> bool:
        """Connect to Redis.

        Returns:
            True if connection successful.
        """
        if not REDIS_AVAILABLE:
            logger.error("Redis not available. Install redis package.")
            return False

        try:
            self._redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
            )
            self._redis.ping()
            self._is_connected = True
            logger.info(f"Connected to Redis: {self.redis_host}:{self.redis_port}")
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            self._redis.close()
            self._redis = None
            self._is_connected = False
            logger.info("Disconnected from Redis")

    def update_from_event(self, event: dict) -> None:
        """Update features from a user event.

        Args:
            event: User event dictionary with keys:
                - visitor_id: User ID
                - item_id: Item ID
                - event: Event type (view, addtocart, transaction)
                - timestamp: Unix timestamp in milliseconds
                - session_id: Optional session ID
        """
        if not self._is_connected:
            if not self.connect():
                logger.error("Cannot update: not connected to Redis")
                return

        visitor_id = event.get("visitor_id")
        item_id = event.get("item_id")
        event_type = event.get("event", "view")
        timestamp = event.get("timestamp", int(datetime.now().timestamp() * 1000))
        session_id = event.get("session_id")

        if not visitor_id or not item_id:
            logger.warning(f"Invalid event: missing visitor_id or item_id")
            return

        try:
            pipe = self._redis.pipeline()

            # Update user features
            self._update_user_features(pipe, visitor_id, item_id, event_type, timestamp)

            # Update item features
            self._update_item_features(pipe, item_id, visitor_id, event_type, timestamp)

            # Update session if provided
            if session_id:
                self._update_session(pipe, visitor_id, session_id, item_id, timestamp)

            pipe.execute()

            with self._lock:
                self.stats["events_processed"] += 1
                self.stats[f"event_{event_type}"] += 1

        except redis.RedisError as e:
            logger.error(f"Redis error updating features: {e}")
            with self._lock:
                self.stats["errors"] += 1

    def _update_user_features(
        self,
        pipe: "redis.client.Pipeline",
        user_id: int,
        item_id: int,
        event_type: str,
        timestamp: int,
    ) -> None:
        """Update user features in pipeline.

        Args:
            pipe: Redis pipeline.
            user_id: User ID.
            item_id: Item ID.
            event_type: Event type.
            timestamp: Event timestamp.
        """
        user_key = f"{self.PREFIX_USER}:{user_id}:features"
        recent_key = f"{self.PREFIX_USER}:{user_id}:recent_items"

        # Increment event counters
        pipe.hincrby(user_key, f"count_{event_type}", 1)
        pipe.hincrby(user_key, "total_events", 1)

        # Update last activity timestamp
        pipe.hset(user_key, "last_activity", timestamp)

        # Update weighted score
        weight = self.event_weights.get(event_type, 1.0)
        pipe.hincrbyfloat(user_key, "weighted_score", weight)

        # Determine funnel stage based on events
        if event_type == "transaction":
            pipe.hset(user_key, "funnel_stage", "buyer")
        elif event_type == "addtocart":
            pipe.hsetnx(user_key, "funnel_stage", "intender")

        # Set TTL
        pipe.expire(user_key, self.TTL_USER_FEATURES)

        # Add to recent items list (with timestamp as score)
        pipe.zadd(recent_key, {str(item_id): timestamp})
        pipe.zremrangebyrank(recent_key, 0, -self.max_recent_items - 1)
        pipe.expire(recent_key, self.TTL_RECENT_ITEMS)

    def _update_item_features(
        self,
        pipe: "redis.client.Pipeline",
        item_id: int,
        user_id: int,
        event_type: str,
        timestamp: int,
    ) -> None:
        """Update item features in pipeline.

        Args:
            pipe: Redis pipeline.
            item_id: Item ID.
            user_id: User ID.
            event_type: Event type.
            timestamp: Event timestamp.
        """
        item_key = f"{self.PREFIX_ITEM}:{item_id}:features"
        recent_key = f"{self.PREFIX_ITEM}:{item_id}:recent_users"

        # Increment event counters
        pipe.hincrby(item_key, f"count_{event_type}", 1)
        pipe.hincrby(item_key, "total_events", 1)

        # Update last interaction timestamp
        pipe.hset(item_key, "last_interaction", timestamp)

        # Update popularity score
        weight = self.event_weights.get(event_type, 1.0)
        pipe.hincrbyfloat(item_key, "popularity_score", weight)

        # Set TTL
        pipe.expire(item_key, self.TTL_USER_FEATURES)

        # Add to recent users list
        pipe.zadd(recent_key, {str(user_id): timestamp})
        pipe.zremrangebyrank(recent_key, 0, -self.max_recent_users - 1)
        pipe.expire(recent_key, self.TTL_RECENT_ITEMS)

    def _update_session(
        self,
        pipe: "redis.client.Pipeline",
        user_id: int,
        session_id: str,
        item_id: int,
        timestamp: int,
    ) -> None:
        """Update session features in pipeline.

        Args:
            pipe: Redis pipeline.
            user_id: User ID.
            session_id: Session ID.
            item_id: Item ID.
            timestamp: Event timestamp.
        """
        session_key = f"{self.PREFIX_USER}:{user_id}:{self.PREFIX_SESSION}:{session_id}"

        # Add item to session with timestamp as score
        pipe.zadd(session_key, {str(item_id): timestamp})
        pipe.expire(session_key, self.TTL_SESSION)

    def get_user_features(self, user_id: int) -> dict[str, Any]:
        """Get user features from Redis.

        Args:
            user_id: User ID.

        Returns:
            User feature dictionary.
        """
        if not self._is_connected:
            return {}

        try:
            user_key = f"{self.PREFIX_USER}:{user_id}:features"
            features = self._redis.hgetall(user_key)

            # Convert numeric strings to appropriate types
            result = {}
            for k, v in features.items():
                if k.startswith("count_") or k == "total_events":
                    result[k] = int(v)
                elif k in ("weighted_score", "popularity_score"):
                    result[k] = float(v)
                elif k in ("last_activity",):
                    result[k] = int(v)
                else:
                    result[k] = v

            return result
        except redis.RedisError as e:
            logger.error(f"Error getting user features: {e}")
            return {}

    def get_user_recent_items(
        self,
        user_id: int,
        limit: int = 50,
    ) -> list[int]:
        """Get user's recent items.

        Args:
            user_id: User ID.
            limit: Maximum items to return.

        Returns:
            List of recent item IDs (most recent first).
        """
        if not self._is_connected:
            return []

        try:
            recent_key = f"{self.PREFIX_USER}:{user_id}:recent_items"
            items = self._redis.zrevrange(recent_key, 0, limit - 1)
            return [int(item_id) for item_id in items]
        except redis.RedisError as e:
            logger.error(f"Error getting recent items: {e}")
            return []

    def get_session_items(
        self,
        user_id: int,
        session_id: str,
    ) -> list[int]:
        """Get items in a session.

        Args:
            user_id: User ID.
            session_id: Session ID.

        Returns:
            List of item IDs in order.
        """
        if not self._is_connected:
            return []

        try:
            session_key = f"{self.PREFIX_USER}:{user_id}:{self.PREFIX_SESSION}:{session_id}"
            items = self._redis.zrange(session_key, 0, -1)
            return [int(item_id) for item_id in items]
        except redis.RedisError as e:
            logger.error(f"Error getting session items: {e}")
            return []

    def get_item_features(self, item_id: int) -> dict[str, Any]:
        """Get item features from Redis.

        Args:
            item_id: Item ID.

        Returns:
            Item feature dictionary.
        """
        if not self._is_connected:
            return {}

        try:
            item_key = f"{self.PREFIX_ITEM}:{item_id}:features"
            features = self._redis.hgetall(item_key)

            result = {}
            for k, v in features.items():
                if k.startswith("count_") or k == "total_events":
                    result[k] = int(v)
                elif k in ("popularity_score",):
                    result[k] = float(v)
                elif k in ("last_interaction",):
                    result[k] = int(v)
                else:
                    result[k] = v

            return result
        except redis.RedisError as e:
            logger.error(f"Error getting item features: {e}")
            return {}

    def get_item_recent_users(
        self,
        item_id: int,
        limit: int = 100,
    ) -> list[int]:
        """Get item's recent users.

        Args:
            item_id: Item ID.
            limit: Maximum users to return.

        Returns:
            List of recent user IDs (most recent first).
        """
        if not self._is_connected:
            return []

        try:
            recent_key = f"{self.PREFIX_ITEM}:{item_id}:recent_users"
            users = self._redis.zrevrange(recent_key, 0, limit - 1)
            return [int(user_id) for user_id in users]
        except redis.RedisError as e:
            logger.error(f"Error getting recent users: {e}")
            return []

    def get_stats(self) -> dict[str, int]:
        """Get processing statistics."""
        with self._lock:
            return dict(self.stats)

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._is_connected

    def __enter__(self) -> "RealTimeFeatureUpdater":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()


class MockFeatureUpdater(RealTimeFeatureUpdater):
    """Mock feature updater for testing without Redis.

    Stores features in memory.
    """

    def __init__(self, **kwargs):
        """Initialize mock updater."""
        super().__init__(**kwargs)
        self._user_features: dict[int, dict] = defaultdict(dict)
        self._item_features: dict[int, dict] = defaultdict(dict)
        self._user_recent_items: dict[int, list] = defaultdict(list)
        self._item_recent_users: dict[int, list] = defaultdict(list)
        self._sessions: dict[str, list] = defaultdict(list)
        self._is_connected = True

    def connect(self) -> bool:
        """Mock connect."""
        self._is_connected = True
        return True

    def disconnect(self) -> None:
        """Mock disconnect."""
        self._is_connected = False

    def update_from_event(self, event: dict) -> None:
        """Update features in memory."""
        visitor_id = event.get("visitor_id")
        item_id = event.get("item_id")
        event_type = event.get("event", "view")
        session_id = event.get("session_id")

        if not visitor_id or not item_id:
            return

        # Update user features
        uf = self._user_features[visitor_id]
        uf[f"count_{event_type}"] = uf.get(f"count_{event_type}", 0) + 1
        uf["total_events"] = uf.get("total_events", 0) + 1
        uf["weighted_score"] = uf.get("weighted_score", 0) + self.event_weights.get(
            event_type, 1.0
        )

        # Update item features
        itf = self._item_features[item_id]
        itf[f"count_{event_type}"] = itf.get(f"count_{event_type}", 0) + 1
        itf["total_events"] = itf.get("total_events", 0) + 1

        # Update recent lists
        if item_id not in self._user_recent_items[visitor_id]:
            self._user_recent_items[visitor_id].insert(0, item_id)
            self._user_recent_items[visitor_id] = self._user_recent_items[visitor_id][
                : self.max_recent_items
            ]

        if visitor_id not in self._item_recent_users[item_id]:
            self._item_recent_users[item_id].insert(0, visitor_id)
            self._item_recent_users[item_id] = self._item_recent_users[item_id][
                : self.max_recent_users
            ]

        # Update session
        if session_id:
            key = f"{visitor_id}:{session_id}"
            if item_id not in self._sessions[key]:
                self._sessions[key].append(item_id)

        with self._lock:
            self.stats["events_processed"] += 1

    def get_user_features(self, user_id: int) -> dict[str, Any]:
        """Get user features from memory."""
        return self._user_features.get(user_id, {})

    def get_user_recent_items(self, user_id: int, limit: int = 50) -> list[int]:
        """Get user's recent items from memory."""
        return self._user_recent_items.get(user_id, [])[:limit]

    def get_session_items(self, user_id: int, session_id: str) -> list[int]:
        """Get session items from memory."""
        key = f"{user_id}:{session_id}"
        return self._sessions.get(key, [])

    def get_item_features(self, item_id: int) -> dict[str, Any]:
        """Get item features from memory."""
        return self._item_features.get(item_id, {})

    def get_item_recent_users(self, item_id: int, limit: int = 100) -> list[int]:
        """Get item's recent users from memory."""
        return self._item_recent_users.get(item_id, [])[:limit]
