"""
Redis cache manager for caching predictions and improving performance.
"""

import redis.asyncio as redis
from typing import Optional
import json

from ..core.config import settings


class CacheManager:
    """Async Redis cache manager."""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None

    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            print("✓ Connected to Redis")
        except Exception as e:
            print(f"✗ Failed to connect to Redis: {e}")
            self.redis_client = None

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if not self.redis_client:
            return None
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            print(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: str, expire: int = 3600):
        """Set value in cache with expiration."""
        if not self.redis_client:
            return
        try:
            await self.redis_client.setex(key, expire, value)
        except Exception as e:
            print(f"Cache set error: {e}")

    async def delete(self, key: str):
        """Delete key from cache."""
        if not self.redis_client:
            return
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            print(f"Cache delete error: {e}")

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            print(f"Cache exists error: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter."""
        if not self.redis_client:
            return 0
        try:
            return await self.redis_client.incrby(key, amount)
        except Exception as e:
            print(f"Cache increment error: {e}")
            return 0

    async def get_ttl(self, key: str) -> int:
        """Get TTL of a key."""
        if not self.redis_client:
            return -1
        try:
            return await self.redis_client.ttl(key)
        except Exception as e:
            print(f"Cache TTL error: {e}")
            return -1


# Global cache manager instance
cache_manager = CacheManager()
