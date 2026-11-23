# %% [Р1.1] Архітектура системи: компоненти, потоки, межі довіри
import sys, json, time, subprocess, os, re
sys.path.insert(0, ".."); sys.path.insert(0, ".")
import warnings, logging; warnings.filterwarnings("ignore")
from transformers.utils import logging as hf_logging; hf_logging.set_verbosity_error()
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 90)
Path("figures").mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 6.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2)
ax.add_patch(Rectangle((2.6, 0.3), 10.2, 5.5, fc="#F4F9F4", ec="#2E7D32", ls="--", lw=1.5))
ax.text(2.75, 5.55, "Межа довіри: сервер служби підтримки (localhost / внутрішня мережа)", color="#2E7D32", fontsize=9, weight="bold")
ax.add_patch(Rectangle((8.8, 0.5), 3.8, 2.3, fc="#FFF8E1", ec="#E65100", ls=":", lw=1.3))
ax.text(8.9, 2.6, "Секрети: .env (NEO4J_PASSWORD), не в образі/репозиторії", color="#E65100", fontsize=7.8)
B = {"Користувач\n(браузер)": (0.2, 2.5, "#FCE4D6"), "Streamlit UI\n:8501": (3.0, 2.5, "#DDEBF7"),
     "FastAPI\n/ask /health /sources\nвалідація, журнали": (5.6, 2.3, "#E2F0D9"),
     "FAISS + BM25\n(69 фрагментів)": (9.2, 4.2, "#EADCF4"), "Neo4j 5.26\nграф знань": (9.2, 1.0, "#FFF2CC"),
     "Ollama\nLlama 3 8B": (9.2, 3.0, "#F4CCCC")}
for t, (x, y, c) in B.items():
    ax.add_patch(FancyBboxPatch((x, y), 2.1 if "FastAPI" not in t else 2.5, 1.0 if "FastAPI" not in t else 1.4,
                                boxstyle="round,pad=0.05", fc=c, ec="#333"))
    ax.text(x + (1.05 if "FastAPI" not in t else 1.25), y + (0.5 if "FastAPI" not in t else 0.7), t, ha="center", va="center", fontsize=9)
arr = lambda a, b, t, dy=0.12: (ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.3)),
                                 ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, t, ha="center", fontsize=7.8, color="#1F4E79"))
arr((2.3, 3.0), (3.0, 3.0), "HTTP")
arr((5.1, 3.0), (5.6, 3.0), "JSON\nAskRequest")
arr((8.1, 3.3), (9.2, 4.6), "q → top-k")
arr((8.1, 3.0), (9.2, 3.5), "промпт")
arr((8.1, 2.6), (9.2, 1.5), "Cypher, глиб. ≤2")
ax.text(6.85, 0.9, "AskResponse: відповідь, цитати,\nзв'язки графа, request_id", ha="center", fontsize=8)
ax.set_title("Архітектура системи «Асистент ІТ-підтримки» (ЛР 9–12)", weight="bold")
fig.tight_layout(); fig.savefig("figures/l12_architecture.png", dpi=150); plt.close(fig)

FLOWS = [("1", "UI → API", "POST /ask, JSON {question ≤500 симв., k, use_graph}"),
         ("2", "API → FAISS/BM25", "embedding запиту (384), top-k фрагментів з метаданими"),
         ("3", "API → Neo4j", "Bolt, параметризований Cypher $concepts (без конкатенації рядків)"),
         ("4", "API → LLM", "HTTP /api/chat: system + «Фрагменти» + «Зв'язки графа» + питання"),
         ("5", "API → UI", "JSON AskResponse; заголовок X-Request-ID"),
         ("6", "API → журнал", "JSON-рядок: request_id, тривалість, етапи, токени (без тексту питання)")]
print(pd.DataFrame(FLOWS, columns=["#", "потік", "формат"]).to_string(index=False))
RISKS = [("prompt injection у питанні/документах", "роздільники, правило «лише фрагменти», перевірка цитат, ліміт 500 симв."),
         ("витік секретів", ".env поза кодом, у журнали не пишемо текст питань, лише хеш і довжину"),
         ("відмова Neo4j", "код 503 GRAPH_UNAVAILABLE, /health показує стан, повторне підключення"),
         ("відмова/таймаут LLM", "таймаут 180 с → 504 LLM_TIMEOUT, повідомлення в UI"),
         ("галюцинації", "відповідь лише з посиланнями [id], «Недостатньо даних», показ цитат користувачу"),
         ("перевантаження", "обмеження k ≤ 8, бюджет контексту 1400 токенів, один процес LLM")]
print(); print(pd.DataFrame(RISKS, columns=["ризик", "захід"]).to_string(index=False))
print("\nВимоги: затримка p95 ≤ 30 с (локальна LLM на ноутбучному GPU), до 200 запитів/день, "
      "доступ — лише внутрішня мережа/службовці 1-ї лінії; Neo4j — тільки читання для API.")

# %% [Р2.1] Векторний і графовий контекст
from app.core import System
S = System()
q = "Комп'ютер пищить при старті і буває синій екран — який компонент несправний і як підтвердити?"
vc = S.vector_context(q, 4)
print("vector_context:"); [print(f"  {c['id']:<9} {c['score']:.4f}  {c['title']} / {c['section']}") for c in vc]
concepts, edges = S.graph_context(q)
print("\ngraph_context: поняття", concepts); print(S.edges_text(edges, concepts))
kept, ged, used = S.merge(vc + vc[:2], edges)
print(f"\nЗлиття: {len(vc) + 2} фрагм. на вході (з дублікатами) → {len(kept)} після дедуплікації, {used} токенів; "
      f"зв'язків графа {len(ged)}")

# %% [Р2.2] Багатокрокове питання: RAG без графа vs RAG + граф
MULTI = ["Комп'ютер пищить при старті і буває синій екран — який компонент несправний і як підтвердити?",
         "ПК раптово вимикається і корпус гарячий: яку дію виконати для підтвердження і як усунути?"]
for q in MULTI:
    a0 = S.ask(q, use_graph=False); a1 = S.ask(q, use_graph=True)
    print(f"Q: {q}\n  БЕЗ графа: {a0['answer']}\n    фрагменти {a0['chunks']}\n"
          f"  З ГРАФОМ:  {a1['answer']}\n    фрагменти {a1['chunks']}, зв'язків {len(a1['graph_edges'])}, "
          f"граф процитовано: {a1['graph_used']}\n")

# %% [Р3.1] Запуск API (uvicorn) і перевірка через curl
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
api_log = open("outputs_api.log", "w", encoding="utf-8")
api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"], stdout=api_log,
                       stderr=subprocess.STDOUT, env=env)
import httpx
for _ in range(120):
    try:
        if httpx.get("http://127.0.0.1:8000/health", timeout=60).status_code: break
    except httpx.HTTPError: time.sleep(1)
def curl(args):
    r = subprocess.run(["curl", "-s", "-i", *args], capture_output=True, text=True, encoding="utf-8")
    head, body = (re.split(r"\r?\n\r?\n", r.stdout, maxsplit=1) + [""])[:2]
    status = head.splitlines()[0]; rid = re.search(r"x-request-id: (\w+)", head, re.I)
    try: body = json.dumps(json.loads(body), ensure_ascii=False, indent=1)
    except Exception: pass
    return f"{status}  X-Request-ID: {rid.group(1) if rid else '-'}\n{body}"
print("$ curl http://127.0.0.1:8000/health"); print(curl(["http://127.0.0.1:8000/health"]))
print("\n$ curl http://127.0.0.1:8000/sources/KB-04:s2"); print(curl(["http://127.0.0.1:8000/sources/KB-04:s2"])[:600])
print("\n$ curl http://127.0.0.1:8000/sources/XX-00:s0"); print(curl(["http://127.0.0.1:8000/sources/XX-00:s0"]))

# %% [Р3.2] POST /ask: успіх, валідація, довгий ввід
Path("req.json").write_text(json.dumps({"question": "Як перевірити батарейку CMOS і коли її міняти?"}, ensure_ascii=False), encoding="utf-8")
print("$ curl -X POST /ask -d @req.json"); out = curl(["-X", "POST", "http://127.0.0.1:8000/ask", "-H",
                                                       "Content-Type: application/json", "--data-binary", "@req.json"])
print(re.sub(r'"quote": "[^"]{60}[^"]*"', lambda m: m.group(0)[:75] + '…"', out))
print("\n$ curl -X POST /ask -d '{\"question\": \"\"}'")
print(curl(["-X", "POST", "http://127.0.0.1:8000/ask", "-H", "Content-Type: application/json", "-d", '{"question": ""}'])[:400])
Path("req_long.json").write_text(json.dumps({"question": "x" * 800}), encoding="utf-8")
print("\n$ curl -X POST /ask (800 символів)")
print(curl(["-X", "POST", "http://127.0.0.1:8000/ask", "-H", "Content-Type: application/json", "--data-binary", "@req_long.json"])[:400])

# %% [Р3.3] Обробка відмов: зупиняємо Neo4j-підключення (невірний URI) і LLM-таймаут — у тестах Р5; тут журнал
time.sleep(1); api_log.flush()
lines = [l for l in open("outputs_api.log", encoding="utf-8") if l.startswith("{")]
print("Структуровані журнали API (останні 6 записів):")
for l in lines[-6:]: print(" ", l.strip()[:230])

# %% [Р3.4] Сторінка OpenAPI /docs (скріншот)
from playwright.sync_api import sync_playwright
from concurrent.futures import ThreadPoolExecutor
def shot_docs():
    with sync_playwright() as pw:
        br = pw.chromium.launch(channel="msedge", headless=True)
        p = br.new_page(viewport={"width": 1300, "height": 900})
        p.goto("http://127.0.0.1:8000/docs"); p.wait_for_selector(".opblock", timeout=30000); p.wait_for_timeout(1500)
        p.screenshot(path="figures/l12_docs.png"); br.close()
ThreadPoolExecutor(1).submit(shot_docs).result(); print("Збережено figures/l12_docs.png")

# %% [Р4.1] Веб-інтерфейс Streamlit: 3 сценарії (скріншоти)
ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.port", "8501",
                       "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
time.sleep(8)
from urllib.parse import quote
SCEN = [("simple", "Яка допустима напруга на лінії +12 В?"),
        ("multistep", MULTI[0]),
        ("noanswer", "Як налаштувати мережевий принтер HP в аудиторії?")]
def shots():
    with sync_playwright() as pw:
        br = pw.chromium.launch(channel="msedge", headless=True)
        for name, q in SCEN:
            p = br.new_page(viewport={"width": 1400, "height": 1000})
            p.goto(f"http://127.0.0.1:8501/?q={quote(q)}")
            p.wait_for_selector("text=Джерела", timeout=240000); p.wait_for_timeout(2500)
            p.screenshot(path=f"figures/l12_ui_{name}.png", full_page=True); p.close()
        # помилка: API зупинено → повідомлення про недоступність
        br.close()
ThreadPoolExecutor(1).submit(shots).result()
print("Збережено:", [f"figures/l12_ui_{n}.png" for n, _ in SCEN])

# %% [Р4.2] Повідомлення про помилку в UI, коли API недоступне
api.terminate(); api.wait()
def shot_err():
    with sync_playwright() as pw:
        br = pw.chromium.launch(channel="msedge", headless=True)
        p = br.new_page(viewport={"width": 1400, "height": 600})
        p.goto(f"http://127.0.0.1:8501/?q={quote('Що означає код 0x0000007B?')}")
        p.wait_for_selector("text=API недоступне", timeout=60000); p.wait_for_timeout(1000)
        p.screenshot(path="figures/l12_ui_error.png"); br.close()
ThreadPoolExecutor(1).submit(shot_err).result()
ui.terminate(); print("Збережено figures/l12_ui_error.png; API і UI зупинено")

# %% [Р5.1] Автоматичні тести API (pytest + TestClient)
r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v", "-p", "no:cacheprovider"], capture_output=True,
                   text=True, encoding="utf-8", env=env)
print("\n".join(l for l in r.stdout.splitlines() if "PASSED" in l or "FAILED" in l or "passed" in l or "failed" in l))

# %% [Р5.2] Перевірка запуску «з нуля»: новий процес, порожній кеш LLM, health і один запит
env0 = {**env, "LLM_CACHE": "0"}
t = time.perf_counter()
api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8001"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, env=env0)
while True:
    try:
        h = httpx.get("http://127.0.0.1:8001/health", timeout=60); break
    except httpx.HTTPError: time.sleep(1)
print(f"Старт API до готовності: {time.perf_counter() - t:.1f} с; /health → {h.status_code} {h.json()['status']}")
r = httpx.post("http://127.0.0.1:8001/ask", json={"question": "Що означає код 0x0000007B?"}, timeout=240).json()
print("Відповідь без кешу:", r["answer"][:200], "| цитати:", [c["id"] for c in r["citations"]], f"| {r['latency_ms']:.0f} мс")
api.terminate()
