# %% [Р1.1] Контейнеризація: requirements, Dockerfile, docker-compose — статична перевірка
import sys, json, re, time, os, subprocess
sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd, yaml
pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 90)
D = Path("deploy")
req = [l for l in (D / "requirements.txt").read_text().splitlines() if l and not l.startswith(("#", "--"))]
print(f"requirements.txt: {len(req)} пакетів, усі з точною версією (==): {all('==' in l for l in req)}")
df = (D / "Dockerfile").read_text(encoding="utf-8")
CHECKS = {"базовий образ python:3.11-slim": "FROM python:3.11-slim" in df,
          "непривілейований користувач (USER ≠ root)": bool(re.search(r"^USER (?!root)\w+", df, re.M)),
          "EXPOSE 8000": "EXPOSE 8000" in df, "HEALTHCHECK": "HEALTHCHECK" in df,
          "CMD uvicorn": 'CMD ["uvicorn"' in df,
          "секрети не копіюються (.env у .dockerignore)": ".env" in (D / ".dockerignore").read_text().split(),
          "у Dockerfile немає паролів": not re.search(r"PASSWORD\s*=", df)}
dc = yaml.safe_load((D / "docker-compose.yml").read_text(encoding="utf-8"))
CHECKS |= {"compose: сервіси api + neo4j + loader": set(dc["services"]) == {"api", "neo4j", "loader"},
           "compose: том для даних Neo4j": "neo4j_data" in dc["volumes"],
           "compose: змінні через env_file .env": dc["services"]["api"]["env_file"] == ".env",
           "compose: healthcheck Neo4j і залежності": dc["services"]["api"]["depends_on"]["neo4j"]["condition"] == "service_healthy",
           "compose: порти лише на 127.0.0.1": all(p.startswith("127.0.0.1") for s in dc["services"].values() for p in s.get("ports", []))}
print(pd.Series(CHECKS).map({True: "✔", False: "✘"}).to_string())
r = subprocess.run(["docker", "--version"], capture_output=True, text=True, shell=True)
print("\ndocker на цьому ПК:", r.stdout.strip() or "не встановлено → запуск перевіряється локальним еквівалентом run_local.py")

# %% [Р1.2] Запуск однією командою (локальний еквівалент docker compose up) і healthcheck
t = time.perf_counter()
out = subprocess.run([sys.executable, "deploy/run_local.py", "--port", "8000"], capture_output=True, text=True,
                     encoding="utf-8", env={**os.environ, "LLM_CACHE": "0", "PYTHONIOENCODING": "utf-8"})
print(out.stdout.strip()); print(f"Загалом: {time.perf_counter() - t:.1f} с")
import httpx
print(json.dumps(httpx.get("http://127.0.0.1:8000/health", timeout=30).json(), ensure_ascii=False, indent=1))

# %% [Р2.1] 50 тестових запитів: структуровані журнали і метрики
sys.path.insert(0, "../13"); from evalset import EVAL
QS = [q for _, _, q, *_ in EVAL]
BAD = [{"question": ""}, {"question": "x" * 900}, {"question": "ok?", "k": 99}, {"q": "без поля"}]
reqs = [{"question": QS[i % len(QS)]} for i in range(46)] + BAD
rng = np.random.default_rng(42); rng.shuffle(reqs)
t = time.perf_counter(); codes = []
for body in reqs:
    codes.append(httpx.post("http://127.0.0.1:8000/ask", json=body, timeout=300).status_code)
codes.append(httpx.get("http://127.0.0.1:8000/sources/XX-00:s0").status_code)
print(f"Виконано {len(reqs)} запитів /ask + 1 /sources за {time.perf_counter() - t:.0f} с; коди:",
      pd.Series(codes).value_counts().to_dict())
logs = [json.loads(l) for l in open("api_run.log", encoding="utf-8") if l.startswith("{")]
ask = pd.DataFrame([l for l in logs if l["event"] == "ask"]).tail(46)
print("\nПриклад JSON-журналу:"); print(json.dumps(ask.iloc[0].to_dict(), ensure_ascii=False, default=str)[:520])

# %% [Р2.2] Ендпоінт /metrics (Prometheus) і зведення
m = httpx.get("http://127.0.0.1:8000/metrics").text
print("\n".join(l for l in m.splitlines() if l.startswith(("ask_requests_total", "llm_tokens_total", "llm_cost",
                                                          "ask_latency_seconds_count", "ask_latency_seconds_sum"))))
SUM = {"запитів /ask (успішних)": len(ask), "помилок валідації 422": codes.count(422), "404": codes.count(404),
       "частка помилок, %": round(100 * sum(c >= 400 for c in codes) / len(codes), 1),
       "затримка медіана, с": round(ask.latency_ms.median() / 1000, 2), "затримка p95, с": round(ask.latency_ms.quantile(.95) / 1000, 2),
       "токени вводу (сер.)": round(ask.tokens_in.mean()), "токени виводу (сер.)": round(ask.tokens_out.mean()),
       "відмов «Недостатньо даних»": int(ask.refused.sum()),
       "вартість 50 запитів, $": round(float(re.search(r"llm_cost_usd_total ([\d.e-]+)", m).group(1)), 5),
       "версії": f"{ask.model.iloc[0]} / {ask.prompt_version.iloc[0]}"}
print(); print(pd.Series(SUM).to_string())
st = pd.DataFrame(ask.stages.tolist())
print("\nЕтапи, мс (медіана / p95):", {c: f"{st[c].median():.0f} / {st[c].quantile(.95):.0f}" for c in st})
ask.to_csv("outputs_requests.csv", index=False)

# %% [Р3.1] Функції PSI і KS; reference-вибірка для 3 ознак
from scipy.stats import ks_2samp
def psi(expected, actual, bins=10):
    """Population Stability Index: межі — квантилі еталону; ε захищає від log(0)."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected); a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
# Еталон: параметри підібрано за 50 реальними запитами (довжина питання, схожість top-1, затримка LLM),
# далі 2000 значень згенеровано з відповідних розподілів (реальних запитів замало для ≥ 1000).
SYS_SCORES = []
sys.path.insert(0, "../12"); from app.core import System
S = System(); SYS_SCORES = [S.dense.search(q, 1)[0][1] for q in QS]
qlen = np.array([len(q) for q in QS]); llm_s = ask.stages.map(lambda d: d["llm_ms"] / 1000).values
rng = np.random.default_rng(42); N = 2000
REF = {"довжина питання, симв.": rng.lognormal(np.log(qlen).mean(), np.log(qlen).std(), N),
       "схожість top-1 (cos)": np.clip(rng.normal(np.mean(SYS_SCORES), np.std(SYS_SCORES), N), 0, 1),
       "затримка LLM, с": rng.lognormal(np.log(llm_s).mean(), np.log(llm_s).std(), N)}
print(pd.DataFrame({k: {"реальних": len(x), "середнє": np.mean(x), "σ": np.std(x)} for k, x in
                    zip(REF, [qlen, SYS_SCORES, llm_s])}).round(3).to_string())
print(pd.DataFrame({k: pd.Series(v).describe() for k, v in REF.items()}).round(3).to_string())
print("Самоперевірка: PSI(еталон, нова вибірка з того ж розподілу) =",
      {k: round(psi(v, rng.choice(v, 500)), 4) for k, v in REF.items()})

# %% [Р3.2] Два типи змін: зсув середнього на 1σ і зміна формату/одиниць (×100)
def current(kind, n=500, seed=7):
    r = np.random.default_rng(seed); out = {}
    for k, v in REF.items():
        x = r.choice(v, n)
        if kind == "shift": x = x + np.std(v)
        if kind == "format": x = x * 100
        out[k] = x
    return out
rows = []
for kind, name in [("none", "без змін"), ("shift", "зсув +1σ"), ("format", "одиниці ×100")]:
    cur = current(kind)
    for k in REF:
        ks = ks_2samp(REF[k], cur[k])
        rows.append({"сценарій": name, "ознака": k, "PSI": round(psi(REF[k], cur[k]), 3), "KS D": round(ks.statistic, 3),
                     "KS p": f"{ks.pvalue:.2e}"})
DR = pd.DataFrame(rows); print(DR.to_string(index=False))

# %% [Р3.3] Графіки: еталон vs поточний; PSI по тижнях
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Path("figures").mkdir(exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
cur = current("shift")
for a, k in zip(ax, REF):
    a.hist(REF[k], bins=40, density=True, alpha=.55, label="еталон", color="#4472C4")
    a.hist(cur[k], bins=40, density=True, alpha=.55, label="поточний (+1σ)", color="#ED7D31")
    a.set_title(f"{k}\nPSI={psi(REF[k], cur[k]):.2f}", fontsize=10); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/l14_dist.png", dpi=150); plt.close(fig)
# 12 тижнів: тижні 1–6 стабільні, з 7-го поступовий зсув (0.2σ/тиждень), тиждень 11 — дефект формату
WEEK = []
for w in range(1, 13):
    r = np.random.default_rng(100 + w); row = {"тиждень": w}
    for k, v in REF.items():
        x = r.choice(v, 300) + max(0, w - 6) * 0.2 * np.std(v)
        if w == 11 and k == "схожість top-1 (cos)": x = x * 100
        row[k] = psi(v, x)
    WEEK.append(row)
WEEK = pd.DataFrame(WEEK).set_index("тиждень")
fig, ax = plt.subplots(figsize=(10, 4))
WEEK.clip(upper=1.5).plot(marker="o", ax=ax)
ax.axhline(0.1, ls="--", c="orange", label="warning 0.10"); ax.axhline(0.25, ls="--", c="red", label="critical 0.25")
ax.set_ylabel("PSI (обрізано до 1.5)"); ax.set_title("PSI ознак по тижнях"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figures/l14_psi_weeks.png", dpi=150); plt.close(fig)
print(WEEK.round(3).to_string())

# %% [Р3.4] Обґрунтування порогів: розподіл PSI на «спокійних» вибірках
calm = np.array([[psi(v, np.random.default_rng(1000 + i).choice(v, 300)) for v in REF.values()] for i in range(300)])
print("PSI на 300 спокійних вибірках (n=300): 95-й / 99-й перцентиль / максимум:",
      np.round(np.percentile(calm, [95, 99]), 3), round(calm.max(), 3))
print("Ефект зсуву 0.2σ / 0.5σ / 1σ на PSI:",
      [round(np.mean([psi(v, np.random.default_rng(5).choice(v, 300) + s * np.std(v)) for v in REF.values()]), 3) for s in (0.2, 0.5, 1)])

# %% [Р4.1] Тривоги: рівні warning/critical, cooldown, підтвердження за двома ознаками
WARN, CRIT, KS_P, COOLDOWN = 0.10, 0.25, 0.01, 2
class Alerter:
    def __init__(self): self.last = {}; self.log = []
    def check(self, t, cur, metrics=None):
        fired, per = [], {}
        for k, v in REF.items():
            p = psi(v, cur[k]); ks = ks_2samp(v, cur[k]).pvalue; per[k] = (p, ks)
        crit = [k for k, (p, ks) in per.items() if p >= CRIT and ks < KS_P]
        warn = [k for k, (p, ks) in per.items() if WARN <= p and k not in crit]
        if len(crit) >= 2: fired.append(("critical", "data_drift", crit))          # підтвердження ≥ 2 ознаками
        elif crit: fired.append(("warning", "data_drift(1 ознака)", crit))
        elif warn: fired.append(("warning", "data_drift", warn))
        if any(np.nanmax(cur[k]) > 50 * np.nanmax(REF[k]) for k in REF):         # дефект формату/одиниць
            fired = [("critical", "data_defect", [k for k in REF if np.nanmax(cur[k]) > 50 * np.nanmax(REF[k])])]
        if metrics:
            if metrics["error_rate"] > 0.05: fired.append(("critical", "llm_failure", [f"errors={metrics['error_rate']:.0%}"]))
            elif metrics["p95_s"] > 60: fired.append(("warning", "llm_latency", [f"p95={metrics['p95_s']:.0f}s"]))
        out = []
        for lvl, name, feats in fired:
            key = (lvl, name)
            if key in self.last and t - self.last[key] <= COOLDOWN:                  # придушення повторів
                self.log.append((t, lvl, name, feats, "suppressed")); continue
            self.last[key] = t; self.log.append((t, lvl, name, feats, "SENT")); out.append((lvl, name, feats))
        return out

# %% [Р4.2] Три сценарії: дрейф даних, дефект даних, відмова LLM
def scenario(name, gen):
    al = Alerter()
    for t in range(1, 9): al.check(t, *gen(t))
    L = pd.DataFrame(al.log, columns=["крок", "рівень", "тривога", "ознаки", "статус"])
    print(f"=== {name}"); print(L.to_string(index=False) if len(L) else "  тривог немає"); print()
    return L
ok_m = {"error_rate": 0.0, "p95_s": 25}
drift = lambda t: ({k: np.random.default_rng(t).choice(v, 300) + (np.std(v) * (t - 3) * 0.35 if t > 3 else 0) for k, v in REF.items()}, ok_m)
defect = lambda t: ({k: np.random.default_rng(t).choice(v, 300) * (100 if (t >= 4 and k.startswith("схож")) else 1) for k, v in REF.items()}, ok_m)
llmfail = lambda t: ({k: np.random.default_rng(t).choice(v, 300) for k, v in REF.items()},
                     {"error_rate": 0.4 if 4 <= t <= 6 else 0.01, "p95_s": 180 if 4 <= t <= 6 else 25})
SC = {n: scenario(n, g) for n, g in [("Сценарій 1: дрейф даних (з кроку 4)", drift),
                                       ("Сценарій 2: дефект даних — схожість у відсотках (з кроку 4)", defect),
                                       ("Сценарій 3: відмова LLM (кроки 4–6)", llmfail)]}

# %% [Р4.3] Хибні тривоги на спокійних даних
def false_alarms(n_checks=500, size=300, two_feature=True):
    al = Alerter(); sent = 0
    for t in range(n_checks):
        cur = {k: np.random.default_rng(10_000 + t).choice(v, size) for k, v in REF.items()}
        sent += sum(1 for _ in al.check(t * 10, cur))   # t*10 — без впливу cooldown
    return sent
fa = false_alarms(); fa_small = false_alarms(size=100)
print(f"Спокійні дані, 500 перевірок по 300 значень: надіслано тривог {fa} ({fa / 500:.1%})")
print(f"Ті самі перевірки, але вибірки по 100 значень: {fa_small} ({fa_small / 500:.1%}) — малі вікна шумніші")
subprocess.run(["taskkill", "/F", "/PID", Path("api.pid").read_text()], capture_output=True)
