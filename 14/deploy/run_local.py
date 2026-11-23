"""Локальний еквівалент `docker compose up` (Docker недоступний на робочому ПК):
одна команда — перевірка Neo4j, ETL графа, запуск API, очікування healthcheck.
    python 14/deploy/run_local.py [--port 8000]"""
import subprocess, sys, time, os
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[2]
port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8000"
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
print("[1/3] ETL онтології в Neo4j (MERGE, ідемпотентно)…", flush=True)
subprocess.run([sys.executable, str(ROOT / "pcdiag" / "load_graph.py")], check=True, env=env)
print("[2/3] Старт API…", flush=True)
log = open(ROOT / "14" / "api_run.log", "a", encoding="utf-8")
api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", port], cwd=ROOT / "12",
                       stdout=log, stderr=subprocess.STDOUT, env=env)
t = time.time()
while time.time() - t < 180:
    try:
        h = httpx.get(f"http://127.0.0.1:{port}/health", timeout=30)
        print(f"[3/3] healthcheck: {h.status_code} {h.json()['status']} за {time.time() - t:.1f} с; PID {api.pid}", flush=True)
        break
    except httpx.HTTPError:
        time.sleep(1)
Path(ROOT / "14" / "api.pid").write_text(str(api.pid))
