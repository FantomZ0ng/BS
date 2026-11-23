# Вибрані розділи науки про дані — ЛР 1–6

Кожна папка `1`–`6` містить:
- `lab0N.py` — код (комірки `# %%`), `lab0N.ipynb` — виконаний ноутбук з виводами;
- `figures/` — рисунки PNG (150 dpi), `report_img/` — скріни коду/виводу для звіту;
- `ЛР0N_Сваричевський_Феі-12с.docx` — звіт (у папці 2 також `Datasheet_Adult.docx`).

Дані — у `data/` (UCI Wine Quality, UCI Adult, Medical Cost, UCI Online Retail,
Hotel Booking Demand; ряд CO2 вбудований у statsmodels).

## Запуск
```
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe _tools/run_lab.py 1      # виконати ноутбук ЛР 1 з нуля
cd _tools && ../.venv/Scripts/python.exe build_lab1.py   # перезібрати звіт
```
Seed: `RANDOM_STATE = 42`, `np.random.default_rng(42)`.

## Розташування (з 22.11.2026)
Папку перенесено з робочого столу на `D:\Вибрані розділи`.
Python-середовище: `D:\venvs\vibrani` (запуск з цієї папки:
`D:\venvs\vibrani\Scripts\python.exe _tools\run_lab.py 7`).
ЛР 7: `7\kappa_ratings.csv` містить ДЕМОНСТРАЦІЙНІ оцінки — замініть реальними оцінками двох експертів.
## ЛР 9–16 (з 23.11.2026)
Система «Асистент ІТ-підтримки»: онтологія ЛР 8 → Neo4j (ЛР 9) → підказки LLM (ЛР 10) → RAG (ЛР 11)
→ API + UI (ЛР 12) → оцінювання (ЛР 13) → експлуатація (ЛР 14) → покращення (ЛР 15) → Whitepaper (ЛР 16).
- Спільний код: `pcdiag/` (llm.py — Ollama, onto.py — онтологія/Neo4j, rag.py — chunking/FAISS/BM25/RRF/rerank,
  load_graph.py — ETL у Neo4j). API: `12/app`, UI: `12/ui`, тести: `12/tests`, розгортання: `14/deploy`.
- Сервіси: Neo4j 5.26 Community (`D:\venvs\neo4j`, Java 21), Ollama з моделями llama3:8b і mistral:7b.
- Секрети: файл `.env` у корені (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) — не передавати й не комітити.
- Кеш відповідей LLM: `.llm_cache/` (temperature=0; зберігає й виміряну затримку). `LLM_CACHE=0` — вимкнути.
- Запуск ЛР N: `D:\venvs\vibrani\Scripts\python.exe _tools\run_lab.py N`, звіт: `cd _tools; python build_labN.py`.
