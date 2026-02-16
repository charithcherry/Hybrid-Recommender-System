"""Cache layer for ML service."""

from .redis_cache import RedisCache, get_cache

__all__ = ['RedisCache', 'get_cache']
