"""
Redis Cache Layer for ML Service
Caches user embeddings, recommendations, and frequently accessed data
"""

import redis
import json
import pickle
import numpy as np
import os
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()


class RedisCache:
    """
    Redis cache manager for the recommendation system.

    Features:
    - User embedding caching (numpy arrays)
    - Recommendation caching with TTL
    - CLIP encoding caching
    - Product metadata caching
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        decode_responses: bool = False
    ):
        """
        Initialize Redis connection.
        Reads from environment variables if parameters not provided.

        Args:
            host: Redis server host (default: from REDIS_HOST env)
            port: Redis server port (default: from REDIS_PORT env)
            db: Redis database number (default: from REDIS_DB env)
            password: Redis password (default: from REDIS_PASSWORD env)
            decode_responses: Whether to decode responses to strings
        """
        # Read from environment if not provided
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = int(port or os.getenv("REDIS_PORT", "6379"))
        self.db = int(db or os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD") or None

        print(f"Connecting to Redis at {self.host}:{self.port}...")

        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=decode_responses,
            socket_keepalive=True,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
        self.enabled = self._check_connection()

    def _check_connection(self) -> bool:
        """Check if Redis is available."""
        try:
            self.client.ping()
            print("✓ Redis connected successfully")
            return True
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"✗ Redis not available: {e}")
            print("  Cache will be disabled (everything will still work)")
            return False

    # ==================== User Embeddings ====================

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get cached user embedding.

        Args:
            user_id: User ID

        Returns:
            Numpy array or None if not cached
        """
        if not self.enabled:
            return None

        try:
            key = f"user_embedding:{user_id}"
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None

    def set_user_embedding(
        self,
        user_id: int,
        embedding: np.ndarray,
        ttl: int = 3600
    ) -> bool:
        """
        Cache user embedding.

        Args:
            user_id: User ID
            embedding: User embedding (numpy array)
            ttl: Time to live in seconds (default: 1 hour)

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        try:
            key = f"user_embedding:{user_id}"
            data = pickle.dumps(embedding)
            self.client.setex(key, ttl, data)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    def invalidate_user_embedding(self, user_id: int) -> bool:
        """Invalidate user embedding when user interactions change."""
        if not self.enabled:
            return False

        try:
            key = f"user_embedding:{user_id}"
            self.client.delete(key)
            return True
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False

    # ==================== Recommendations ====================

    def get_recommendations(
        self,
        user_id: int,
        rec_type: str = "split",
        filters: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Get cached recommendations.

        Args:
            user_id: User ID
            rec_type: Type of recommendations (split, cf, content)
            filters: Applied filters (for cache key)

        Returns:
            Cached recommendations or None
        """
        if not self.enabled:
            return None

        try:
            # Create cache key with filters
            filter_str = json.dumps(filters or {}, sort_keys=True)
            key = f"recommendations:{rec_type}:{user_id}:{hash(filter_str)}"

            data = self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None

    def set_recommendations(
        self,
        user_id: int,
        recommendations: Dict,
        rec_type: str = "split",
        filters: Optional[Dict] = None,
        ttl: int = 300
    ) -> bool:
        """
        Cache recommendations.

        Args:
            user_id: User ID
            recommendations: Recommendation data
            rec_type: Type of recommendations
            filters: Applied filters
            ttl: Time to live in seconds (default: 5 minutes)

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        try:
            filter_str = json.dumps(filters or {}, sort_keys=True)
            key = f"recommendations:{rec_type}:{user_id}:{hash(filter_str)}"
            data = json.dumps(recommendations)
            self.client.setex(key, ttl, data)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    def invalidate_user_recommendations(self, user_id: int) -> int:
        """
        Invalidate all recommendation caches for a user.
        Called when user interactions change.

        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0

        try:
            pattern = f"recommendations:*:{user_id}:*"
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Cache delete error: {e}")
            return 0

    # ==================== CLIP Encodings ====================

    def get_clip_encoding(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached CLIP text encoding.

        Args:
            text: Text to encode

        Returns:
            Cached encoding or None
        """
        if not self.enabled:
            return None

        try:
            key = f"clip:{hash(text)}"
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None

    def set_clip_encoding(
        self,
        text: str,
        encoding: np.ndarray,
        ttl: int = 86400
    ) -> bool:
        """
        Cache CLIP text encoding.

        Args:
            text: Input text
            encoding: CLIP encoding
            ttl: Time to live (default: 24 hours)

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        try:
            key = f"clip:{hash(text)}"
            data = pickle.dumps(encoding)
            self.client.setex(key, ttl, data)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    # ==================== Product Metadata ====================

    def get_product(self, item_id: int) -> Optional[Dict]:
        """Get cached product metadata."""
        if not self.enabled:
            return None

        try:
            key = f"product:{item_id}"
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None

    def set_product(
        self,
        item_id: int,
        product_data: Dict,
        ttl: int = 3600
    ) -> bool:
        """Cache product metadata."""
        if not self.enabled:
            return False

        try:
            key = f"product:{item_id}"
            data = json.dumps(product_data)
            self.client.setex(key, ttl, data)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    # ==================== Utility Methods ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}

        try:
            info = self.client.info("stats")
            return {
                "enabled": True,
                "total_keys": self.client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) /
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                    * 100
                ),
                "memory_used_mb": info.get("used_memory", 0) / 1024 / 1024,
            }
        except Exception as e:
            print(f"Stats error: {e}")
            return {"enabled": False, "error": str(e)}

    def flush_all(self) -> bool:
        """Flush all cache (use with caution!)."""
        if not self.enabled:
            return False

        try:
            self.client.flushdb()
            print("✓ Cache flushed")
            return True
        except Exception as e:
            print(f"Flush error: {e}")
            return False

    def close(self):
        """Close Redis connection."""
        if self.enabled:
            self.client.close()


# Singleton instance
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get or create Redis cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance
