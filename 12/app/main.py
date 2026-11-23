"""REST API системи «Асистент ІТ-підтримки»: POST /ask, GET /health, GET /sources/{id}, GET /metrics."""
import json, logging, time, uuid, hashlib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from .core import System, GraphUnavailable, LLMError, PROMPT_VERSION, MODEL

# ---------- структуровані журнали (JSON, без персональних даних: текст питання не пишемо) ----------
log = logging.getLogger("api")
if not log.handlers:
    h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h); log.setLevel(logging.INFO); log.propagate = False


def jlog(**kw):
    log.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}, ensure_ascii=False))


# ---------- прості метрики (формат Prometheus) ----------
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    REQS = Counter("ask_requests_total", "Запити /ask", ["status"])
    LAT = Histogram("ask_latency_seconds", "Затримка /ask", buckets=(0.5, 1, 2, 4, 8, 16, 32, 64))
    TOK = Counter("llm_tokens_total", "Токени LLM", ["direction"])
    COST = Counter("llm_cost_usd_total", "Оцінена вартість LLM, $")
except ValueError:          # повторний імпорт у тестах
    pass

app = FastAPI(title="PC Support Assistant API", version="1.0.0",
              description="RAG + граф знань Neo4j + локальна LLM (Ollama)")
_system = None


def system():
    global _system
    if _system is None:
        _system = System()
    return _system


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, description="Питання користувача (3–500 символів)")
    k: int = Field(4, ge=1, le=8, description="Кількість фрагментів")
    use_graph: bool = True

    @field_validator("question")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("порожнє питання")
        return v.strip()


class Citation(BaseModel):
    id: str; title: str; section: str; quote: str


class Edge(BaseModel):
    source: str; rel: str; target: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    refused: bool
    citations: list[Citation]
    graph_edges: list[Edge]
    concepts: list[str]
    chunks: list[str]
    tokens_in: int; tokens_out: int
    latency_ms: float
    model: str; prompt_version: str


@app.middleware("http")
async def request_id_mw(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.rid = rid
    t = time.perf_counter()
    resp = await call_next(request)
    resp.headers["X-Request-ID"] = rid
    jlog(event="http", request_id=rid, method=request.method, path=request.url.path,
         status=resp.status_code, duration_ms=round((time.perf_counter() - t) * 1000, 1))
    return resp


def error(status, code, msg, rid):
    REQS.labels(status=code).inc()
    jlog(event="error", request_id=rid, code=code, status=status)
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": msg}, "request_id": rid})


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    rid = request.state.rid
    t = time.perf_counter()
    try:
        out = system().ask(req.question, k=req.k, use_graph=req.use_graph)
    except GraphUnavailable as e:
        return error(503, "GRAPH_UNAVAILABLE", f"Граф знань недоступний ({e}); повторіть пізніше", rid)
    except LLMError as e:
        return error(504, "LLM_TIMEOUT", f"Мовна модель не відповіла вчасно ({e})", rid)
    lat = (time.perf_counter() - t) * 1000
    REQS.labels(status="ok").inc(); LAT.observe(lat / 1000)
    TOK.labels(direction="in").inc(out["tokens_in"]); TOK.labels(direction="out").inc(out["tokens_out"])
    COST.inc(out["cost_usd"])
    jlog(event="ask", request_id=rid, q_len=len(req.question),
         q_hash=hashlib.sha256(req.question.encode()).hexdigest()[:10], model=MODEL, prompt_version=PROMPT_VERSION,
         latency_ms=round(lat, 1), stages=out["timings_ms"], tokens_in=out["tokens_in"], tokens_out=out["tokens_out"],
         chunks=out["chunks"], graph_edges=len(out["graph_edges"]), refused=out["refused"])
    return AskResponse(request_id=rid, latency_ms=round(lat, 1), **{k: out[k] for k in (
        "answer", "refused", "citations", "graph_edges", "concepts", "chunks", "tokens_in", "tokens_out",
        "model", "prompt_version")})


@app.get("/health")
def health():
    st = system().health()
    ok = all(v["ok"] for v in st.values())
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "degraded", "components": st})


@app.get("/sources/{chunk_id}")
def source(chunk_id: str):
    c = system().by_id.get(chunk_id)
    if c is None:
        raise HTTPException(404, detail={"code": "SOURCE_NOT_FOUND", "message": f"Фрагмент {chunk_id} не існує"})
    return {k: c[k] for k in ("id", "doc", "title", "section", "page", "lang", "text")}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)
