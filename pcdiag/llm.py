"""Клієнт локальної LLM (Ollama) з детермінованими налаштуваннями, підрахунком токенів і вартості."""
import json, time, hashlib, os, httpx
from pathlib import Path

# Кеш відповідей (temperature=0 → детерміновано): повторний запуск ноутбука не викликає LLM вдруге.
# Зберігаються й виміряні під час першого виклику затримка та токени. Вимкнути: LLM_CACHE=0.
CACHE = Path(__file__).resolve().parent.parent / ".llm_cache"
USE_CACHE = os.environ.get("LLM_CACHE", "1") == "1"

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("LLM_MODEL", "llama3:latest")          # Meta Llama 3 8B Instruct, Q4_0 (Ollama)
# Тариф для оцінки вартості (ПРИПУЩЕННЯ: хмарний API моделі класу 8B, $ за 1 млн токенів)
PRICE_IN, PRICE_OUT = 0.20, 0.20


class LLMError(RuntimeError):
    pass


def chat(messages, model=MODEL, json_mode=True, temperature=0.0, seed=42, max_tokens=300, timeout=180):
    """Повертає dict: text, tokens_in, tokens_out, latency_s, cost_usd."""
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_tokens, "num_ctx": 4096}}
    if json_mode:
        body["format"] = "json"
    key = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cf = CACHE / f"{key}.json"
    if USE_CACHE and cf.exists():
        return {**json.loads(cf.read_text(encoding="utf-8")), "cached": True}
    t = time.perf_counter()
    try:
        r = httpx.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        raise LLMError(f"LLM недоступна: {type(e).__name__}") from e
    d = r.json()
    ti, to = d.get("prompt_eval_count", 0), d.get("eval_count", 0)
    res = {"text": d["message"]["content"], "tokens_in": ti, "tokens_out": to,
           "latency_s": time.perf_counter() - t, "cost_usd": (ti * PRICE_IN + to * PRICE_OUT) / 1e6}
    if USE_CACHE:
        CACHE.mkdir(exist_ok=True); cf.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return {**res, "cached": False}


def parse_json(text):
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e.msg}"
