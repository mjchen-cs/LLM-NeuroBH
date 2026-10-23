"""OpenAI-compatible chat client for FREE / low-cost LLM endpoints.

No model training is needed: the LLM is used zero/few-shot through an API.
Every provider below exposes the OpenAI chat-completions protocol, so one
client covers all of them (see configs/llm.yaml):

  * Zhipu BigModel  GLM-4-Flash (free)             https://open.bigmodel.cn/api/paas/v4/
  * SiliconFlow     free 7B/9B models (Qwen, GLM)  https://api.siliconflow.cn/v1
  * OpenRouter      ":free" models                 https://openrouter.ai/api/v1
  * Groq            free tier                      https://api.groq.com/openai/v1
  * Ollama (local, free, no key)                   http://localhost:11434/v1

Free tiers have rate limits -> min_interval + retry with back-off, and every
response is cached (data/llm_cache.jsonl) so experiments are reproducible and
re-running never re-bills / re-hits the quota.
"""
import hashlib
import json
import os
import threading
import time

import yaml


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, name, base_url, model, api_key_env=None, temperature=0.0, max_tokens=1024,
                 min_interval=1.0, timeout=60, cache_path="data/llm_cache.jsonl", extra_body=None):
        from openai import OpenAI
        self.name, self.model = name, model
        key = os.environ.get(api_key_env, "") if api_key_env else "ollama"
        # Without a key the client runs in cache-only mode: replies cached in data/llm_cache.jsonl are
        # returned as usual (this is enough to reproduce the paper), and a cache miss raises LLMError.
        self.offline = bool(api_key_env and not key)
        if self.offline:
            print(f"[llm] {name}: {api_key_env} is not set -> cache-only mode")
        self.client = OpenAI(base_url=base_url, api_key=key or "none", timeout=timeout)
        self.temperature, self.max_tokens = temperature, max_tokens
        self.min_interval = min_interval
        self.extra_body = extra_body or {}
        self._last = 0.0
        self._lock = threading.Lock()
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line); self.cache[r["key"]] = r
                    except Exception:
                        pass

    def _key(self, messages):
        h = hashlib.sha256(json.dumps([self.model, self.temperature, messages], ensure_ascii=False,
                                      sort_keys=True).encode("utf-8")).hexdigest()
        return h

    def chat(self, messages, use_cache=True):
        """Returns dict(text, latency, cached, usage)."""
        key = self._key(messages)
        if use_cache and key in self.cache:
            r = self.cache[key]
            return {"text": r["text"], "latency": r["latency"], "cached": True, "usage": r.get("usage")}
        if self.offline:
            raise LLMError(f"{self.name}: request not in cache and no API key set")
        delay = 2.0
        for attempt in range(6):
            with self._lock:
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature,
                    max_tokens=self.max_tokens, extra_body=self.extra_body or None)
                lat = time.time() - t0
                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                usage = usage.model_dump() if hasattr(usage, "model_dump") else None
                rec = {"key": key, "model": self.model, "text": text, "latency": lat, "usage": usage}
                self.cache[key] = rec
                if self.cache_path:
                    os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return {"text": text, "latency": lat, "cached": False, "usage": usage}
            except Exception as e:                     # rate limit / network / 5xx
                msg = str(e)
                code = getattr(e, "status_code", None)
                if code in (400, 401, 403, 404):       # bad key / model disabled / not found: retrying is useless
                    raise LLMError(f"{self.name}: {msg}")
                if attempt == 5:
                    raise LLMError(f"{self.name}: {msg}")
                print(f"[llm:{self.name}] attempt {attempt + 1} failed ({msg[:120]}), retry in {delay:.0f}s")
                time.sleep(delay); delay = min(delay * 2, 60)


def load_dotenv(path=None):
    """Read API keys from .env in the repository root (KEY=VALUE per line) into os.environ.
    Values already set in the environment take precedence.  .env is git-ignored."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    if not os.path.exists(path):
        return
    # utf-8-sig: Windows PowerShell 5 "Set-Content -Encoding utf8" writes a BOM
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v


def load_clients(cfg_path="configs/llm.yaml", names=None):
    """Instantiate the providers listed in the YAML file (those whose key is set)."""
    load_dotenv()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = {}
    for name, c in cfg["models"].items():
        if names and name not in names:
            continue
        if c.get("provider") == "mock":
            from .mock import MockLLM
            out[name] = MockLLM(name, error_rate=c.get("error_rate", 0.2), seed=c.get("seed", 0))
            continue
        try:
            out[name] = LLMClient(name, c["base_url"], c["model"], c.get("api_key_env"),
                                  c.get("temperature", 0.0), c.get("max_tokens", 1024),
                                  c.get("min_interval", 1.0), c.get("timeout", 60),
                                  cfg.get("cache_path", "data/llm_cache.jsonl"), c.get("extra_body"))
        except LLMError as e:
            print(f"[llm] skip {name}: {e}")
    if not out:
        raise LLMError("no LLM client could be created; set API keys or use the 'mock' entry for testing")
    return out
