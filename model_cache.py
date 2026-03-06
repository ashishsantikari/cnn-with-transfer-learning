from typing import Any
import gc
import time


class ModelCacheManager:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, dict[str, Any]] = {}

    def _cleanup_bundle(self, bundle: dict[str, Any]) -> None:
        if bundle.get("kind") == "keras":
            try:
                from tensorflow.keras import backend as K

                K.clear_session()
            except Exception:
                pass

    def unload_cache_entry(self, cache_key: str) -> None:
        entry = self._cache.get(cache_key)
        if entry is None:
            return
        bundle = entry.get("bundle") or {}
        self._cleanup_bundle(bundle)
        self._cache.pop(cache_key, None)

    def purge_stale_model_cache(self) -> None:
        if not self._cache:
            return

        now = time.time()
        stale_keys: list[str] = []
        for cache_key, entry in self._cache.items():
            last_used = float(entry.get("last_used", 0.0))
            if now - last_used > self.ttl_seconds:
                stale_keys.append(cache_key)

        for cache_key in stale_keys:
            self.unload_cache_entry(cache_key)

        if stale_keys:
            gc.collect()

    def get_cached_bundle(self, cache_key: str) -> dict[str, Any] | None:
        entry = self._cache.get(cache_key)
        if not entry:
            return None
        entry["last_used"] = time.time()
        bundle = entry.get("bundle")
        if isinstance(bundle, dict):
            return bundle
        return None

    def store_cached_bundle(self, cache_key: str, bundle: dict[str, Any]) -> dict[str, Any]:
        self._cache[cache_key] = {"bundle": bundle, "last_used": time.time()}
        return bundle
