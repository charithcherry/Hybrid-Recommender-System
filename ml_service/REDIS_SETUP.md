# Redis Cache Setup Guide

Redis caching significantly improves performance by caching:
- User embeddings (expensive to compute)
- Recommendations (reuse recent results)
- CLIP encodings (semantic search)

---

## 🌥️ Option 1: Redis Cloud (Recommended - Free & Easy)

### Setup Steps:

1. **Create Free Account**
   - Go to: https://redis.com/try-free/
   - Sign up (no credit card required)
   - Verify email

2. **Create Database**
   - Click "New Database"
   - Select **FREE tier** (30MB - plenty for our use case)
   - Choose region closest to you
   - Click "Activate"

3. **Get Connection Details**
   After creation, you'll see:
   - **Endpoint:** `redis-12345.c123.us-east-1-1.ec2.redns.redis-cloud.com:12345`
   - **Password:** Click "eye icon" to reveal

4. **Update .env File**
   ```bash
   cd ml_service
   cp .env.template .env
   ```

   Edit `.env` with your Redis Cloud details:
   ```env
   REDIS_HOST=redis-12345.c123.us-east-1-1.ec2.redns.redis-cloud.com
   REDIS_PORT=12345
   REDIS_PASSWORD=your-redis-cloud-password-here
   REDIS_DB=0
   ```

5. **Restart ML Service**
   ```bash
   # Stop current service (Ctrl+C)
   source venv/Scripts/activate
   cd ml_service
   python src/api/main.py
   ```

   You should see:
   ```
   Connecting to Redis at redis-xxxxx.redislabs.com:12345...
   ✓ Redis connected successfully
   Redis cache initialized
   ```

---

## 🐳 Option 2: Docker (Local Development)

```bash
docker run -d --name redis-recommender -p 6379:6379 redis:7-alpine
```

Then use default .env settings:
```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
```

---

## 🧪 Testing the Cache

### 1. Check Cache is Working
```bash
curl http://localhost:8000/cache/stats
```

**Expected output (before any requests):**
```json
{
  "enabled": true,
  "total_keys": 0,
  "hits": 0,
  "misses": 0,
  "hit_rate": 0.0,
  "memory_used_mb": 0.5
}
```

### 2. Make a Request (Cache Miss)
```bash
curl "http://localhost:8000/recommend/1000/split?n=10"
```

Check logs - you should see:
```
[CACHE SET] Cached recommendations for user 1000 (TTL: 300s)
```

### 3. Make Same Request Again (Cache Hit)
```bash
curl "http://localhost:8000/recommend/1000/split?n=10"
```

Check logs - you should see:
```
[CACHE HIT] Returning cached recommendations for user 1000
```

**Response time should drop from ~500ms to ~10ms!** ⚡

### 4. Check Cache Stats
```bash
curl http://localhost:8000/cache/stats
```

**Expected output:**
```json
{
  "enabled": true,
  "total_keys": 1,
  "hits": 1,
  "misses": 1,
  "hit_rate": 50.0,
  "memory_used_mb": 1.2
}
```

---

## 📊 Cache Performance

### Without Cache:
- First request: ~500-800ms (CLIP encoding + FAISS search + clustering)
- Second request: ~500-800ms (recomputes everything)

### With Cache:
- First request: ~500-800ms (computes and caches)
- Second request: **~5-10ms** ⚡ (95% faster!)
- Cache hit rate: ~70-80% in production

---

## 🔄 Cache Invalidation

Cache is **automatically invalidated** when:
- User adds/removes an interaction (like, save, buy)
- Ensures fresh recommendations after user behavior changes

Cache TTL (time-to-live):
- **Recommendations:** 5 minutes (300s)
- **User embeddings:** 1 hour (3600s)
- **CLIP encodings:** 24 hours (86400s)

---

## 🐛 Troubleshooting

### "Redis not available" message
- **Check:** Redis server is running
- **Test:** `redis-cli ping` (should return "PONG")
- **Cloud:** Verify host/port/password in .env

### Cache always disabled
- **Check:** .env file exists in ml_service/
- **Verify:** Connection details are correct
- **Logs:** Check startup logs for connection errors

### Low hit rate
- **Normal** for first few requests
- **Expected:** 70-80% after system warms up
- **If low:** Check TTL settings (may be too short)

---

## 💡 Benefits You'll See

1. **Faster responses:** 10ms vs 500ms for cached requests
2. **Reduced CPU:** No recomputation for frequent users
3. **Better UX:** Instant results for repeat visitors
4. **Scalability:** Handle 10x more requests

---

## ✅ Verification Checklist

- [ ] Redis Cloud database created
- [ ] Connection details added to `.env`
- [ ] ML service restarted
- [ ] Startup logs show "Redis connected"
- [ ] `/cache/stats` endpoint works
- [ ] First request slower (cache miss)
- [ ] Second request faster (cache hit)
- [ ] Cache hit rate > 0%

**Status:** Ready to use! 🚀
