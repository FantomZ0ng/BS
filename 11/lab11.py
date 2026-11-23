# %% [Р1.1] Завантаження корпусу й очищення тексту (колонтитули, пробіли)
import sys, json, re, time, statistics as st
sys.path.insert(0, "..")
import warnings, logging; warnings.filterwarnings("ignore"); logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers.utils import logging as hf_logging; hf_logging.set_verbosity_error()
from pathlib import Path
import numpy as np, pandas as pd
from pcdiag import rag
from pcdiag.rag import load_docs, chunk_fixed, chunk_structural, DenseIndex, BM25Index, HybridRetriever, rrf

pd.set_option("display.width", 170); pd.set_option("display.max_colwidth", 80)
docs = load_docs("corpus")
print(pd.DataFrame([{k: d[k] for k in ("doc", "title", "lang", "removed", "raw_chars", "clean_chars")} for d in docs])
      .to_string(index=False))
print(f"\nДокументів: {len(docs)}; видалено рядків-колонтитулів: {sum(d['removed'] for d in docs)}; "
      f"символів до/після: {sum(d['raw_chars'] for d in docs)} / {sum(d['clean_chars'] for d in docs)}")
print("Ліцензія корпусу: власні тексти автора (CC BY 4.0), створені для ЛР; дані про обладнання — демонстраційні.")
raw = (Path("corpus") / docs[0]["file"]).read_text(encoding="utf-8").splitlines()
print("\nДо очищення (KB-01, рядки 11–14):", *raw[10:14], sep="\n  ")
print("Після очищення той самий фрагмент:", *[l for _, l in docs[0]["lines"][8:10]], sep="\n  ")

# %% [Р1.2] Дві стратегії chunking і метадані
F = chunk_fixed(docs, size=400, overlap=0.15)
S = chunk_structural(docs, max_tokens=200)
def stats(ch):
    t = [c["tokens"] for c in ch]
    return {"фрагментів": len(ch), "сер. токенів": round(st.mean(t), 1), "мін": min(t), "макс": max(t),
            "> 128 токенів (вікно моделі)": sum(x > 128 for x in t), "на документ": round(len(ch) / len(docs), 2)}
print(pd.DataFrame({"фіксований 400 / 15 %": stats(F), "структурний (розділ ##)": stats(S)}).to_string())
print("\nМетадані фрагмента (структурний):"); print({k: v for k, v in S[15].items() if k != "text"})
print("Метадані фрагмента (фіксований):"); print({k: v for k, v in F[5].items() if k != "text"})
Path("index").mkdir(exist_ok=True)
for name, ch in [("fixed", F), ("structural", S)]:
    with open(f"index/chunks_{name}.jsonl", "w", encoding="utf-8") as f:
        for c in ch: f.write(json.dumps(c, ensure_ascii=False) + "\n")

# %% [Р1.3] Порівняння стратегій на прикладах «поганих» розрізів
def show(c, n=160):
    return f"[{c['id']}] стор.{c['page']} «{c['section']}» ({c['tokens']} ток.): …{c['text'][-n:]}"
# фіксований фрагмент, що обривається посеред нумерованого списку / речення
bad = [c for c in F if not re.search(r"[.!?:»)]$", c["text"].strip())]
print(f"Фіксованих фрагментів, що закінчуються не на межі речення: {len(bad)} з {len(F)}")
for c in bad[:3]: print("  " + show(c))
nxt = F[F.index(bad[0]) + 1]
print(f"\n  початок наступного [{nxt['id']}]: {nxt['text'][:160]}…")
mixed = [c for c in F if len({s for s in re.findall(r'##', c['text'])}) or c["section"] != ""]
cross = [c for c in F if c["doc"] == "KB-01"]
print(f"\nKB-01: фіксованих фрагментів {len(cross)}, структурних {sum(c['doc'] == 'KB-01' for c in S)}; "
      "фіксований фрагмент змішує кілька розділів (метадані section — лише перший):")
print("  " + cross[0]["text"][:400] + "…")
print("\nСтруктурні фрагменти KB-01:"); [print("  " + show(c, 90)) for c in S if c["doc"] == "KB-01"]

# %% [Р2.1] Embeddings, індекс FAISS (IndexFlatIP) і збереження на диск
t = time.perf_counter()
dense_S = DenseIndex(S); dense_F = DenseIndex(F)
print(f"Модель: {rag.EMB_MODEL}; розмірність {dense_S.vecs.shape[1]}; макс. довжина входу "
      f"{rag.model().max_seq_length} токенів; час кодування {time.perf_counter() - t:.1f} с")
print("Норми векторів (мають бути 1):", np.round(np.linalg.norm(dense_S.vecs, axis=1)[:5], 4))
dense_S.save("index/structural"); dense_F.save("index/fixed")
loaded = DenseIndex.load("index/structural")
print("Збережено:", sorted(str(p) for p in Path("index").rglob("*.*")))
print("Перевірка завантаження з диска: векторів", loaded.index.ntotal, "| той самий результат:",
      [c["id"] for c, _ in loaded.search("TjMax", 3)] == [c["id"] for c, _ in dense_S.search("TjMax", 3)])

# %% [Р2.2] Пошук top-k з фільтрацією за метаданими (мова, документ)
q = "критична температура процесора"
print("Без фільтра:", [(c["id"], c["lang"], round(s, 3)) for c, s in dense_S.search(q, 5)])
print("lang = en:  ", [(c["id"], c["lang"], round(s, 3)) for c, s in dense_S.search(q, 5, where={"lang": "en"})])
print("doc ∈ {KB-03, KB-17}:", [(c["id"], round(s, 3)) for c, s in
      dense_S.search("чи можна відкривати блок живлення", 3, where={"doc": ["KB-03", "KB-17"]})])

# %% [Р2.3] Перевірка якості на 5 запитах: чи є релевантний фрагмент у top-3
CHECK = [("Яка допустима напруга на лінії +12 В?", "KB-03:s2"),
         ("Що означає безперервний довгий писк при увімкненні?", "KB-07:s1"),
         ("як відновити завантажувач bootrec", "KB-11:s2"),
         ("скільки триває гарантія на комп'ютери аудиторій", "KB-18:s0"),
         ("комп став тихим, а потім вимкнувся", "KB-02:s0")]
for q, gold in CHECK:
    hits = dense_S.search(q, 3)
    ids = [c["id"] for c, _ in hits]
    print(f"{'✔' if gold in ids else '✘'} {q}\n    очікується {gold}; top-3: {[(i, round(s, 2)) for (c, s), i in zip(hits, ids)]}")

# %% [Р3.1] RAG-конвеєр з цитуванням і відмовою на 10 питаннях
print(rag.RAG_SYSTEM); print()
RAGQ = ["Яка допустима напруга на лінії +12 В за стандартом ATX?",
        "Що означає синій екран з кодом 0x0000007B?",
        "Як перевірити батарейку CMOS і коли її міняти?",
        "Скільки проходів MemTest86 потрібно і що означає одна помилка?",
        "Який строк вирішення заявки з пріоритетом P1?",
        "Чи можна прибирати пил всередині ПК пилососом?",
        "Як відновити завантажувач, якщо пише BOOTMGR is missing?",
        "Які ознаки прихованого майнера?",
        "Скільки коштує заміна відеокарти в сервісному центрі?",            # немає в корпусі
        "Як налаштувати мережевий принтер HP у аудиторії?"]                  # немає в корпусі
H = HybridRetriever(S, dense=dense_S)
by_id = {c["id"]: c for c in S}
ANS = []
for q in RAGQ:
    a = rag.answer(q, H, k=4); a["check"] = rag.check_citations(a, by_id); ANS.append(a)
    print(f"Q: {q}\nA: {a['answer']}\n   джерела top-4: {a['hits']} | цитати: {a['citations']} | "
          f"перевірка: {[(c['id'], c['overlap'], c['cos'], '✔' if c['ok'] else '✘') for c in a['check']]}\n")

# %% [Р3.2] Підсумок автоматичної перевірки цитат
rows = []
for a in ANS:
    rows.append({"питання": a["question"][:48], "відмова": a["refused"], "цитат": len(a["citations"]),
                 "цитати ⊆ top-k": set(a["citations"]) <= set(a["hits"]),
                 "підтверджено": sum(c["ok"] for c in a["check"]), "перевірено": len(a["check"]),
                 "ток. вх/вих": f"{a['tokens_in']}/{a['tokens_out']}", "с": round(a["latency_s"], 1)})
R3 = pd.DataFrame(rows); print(R3.to_string(index=False))
allc = [c for a in ANS for c in a["check"]]
print(f"\nЦитат підтверджено: {sum(c['ok'] for c in allc)}/{len(allc)}; "
      f"відмов на 2 питання без відповіді: {sum(a['refused'] for a in ANS[-2:])}/2; "
      f"хибних відмов на 8 питань з відповіддю: {sum(a['refused'] for a in ANS[:8])}/8")
Path("outputs").mkdir(exist_ok=True)
json.dump(ANS, open("outputs/rag_answers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# %% [Р4.1] BM25, dense і гібрид RRF: recall@1/3/5 на 26 питаннях
EVAL = [  # (питання, допустимі документи, тип)
 ("Яка допустима напруга на лінії +12 В за стандартом ATX?", {"KB-03"}, "факт"),
 ("Що означає код 0x0000007B?", {"KB-08"}, "код"),
 ("bootrec /fixboot відмовлено в доступі", {"KB-11"}, "код"),
 ("Яка критична температура процесора i5-10400?", {"KB-01", "KB-21"}, "факт"),
 ("Який атрибут SMART показує перепризначені сектори?", {"KB-09"}, "факт"),
 ("Що означають безперервні довгі звукові сигнали при старті?", {"KB-07"}, "факт"),
 ("Скільки проходів MemTest86 рекомендовано?", {"KB-05"}, "факт"),
 ("Чому після вимкнення з розетки збивається годинник?", {"KB-04"}, "синонім"),
 ("комп став тихим а потім гасне через хвилину", {"KB-02"}, "синонім"),
 ("Які строки вирішення заявки з критичним пріоритетом?", {"KB-16"}, "факт"),
 ("Чи можна розкривати БЖ щоб полагодити?", {"KB-03", "KB-17"}, "синонім"),
 ("Як чисто перевстановити драйвер відеокарти?", {"KB-13"}, "факт"),
 ("VIDEO_TDR_FAILURE nvlddmkm.sys", {"KB-08", "KB-13", "KB-14"}, "код"),
 ("DISM /Online /Cleanup-Image /RestoreHealth", {"KB-12"}, "код"),
 ("ознаки прихованого майнера", {"KB-15"}, "факт"),
 ("INC-2026-00417", {"KB-16"}, "код"),
 ("Скільки місяців гарантія на ПК аудиторій?", {"KB-18", "KB-22"}, "факт"),
 ("Монітор пише немає сигналу, що перевірити?", {"KB-19"}, "факт"),
 ("Диск завантажений на 100 % при малому навантаженні", {"KB-20", "KB-09"}, "факт"),
 ("What is the PL2 power limit of the lab processor?", {"KB-21"}, "англ."),
 ("Which graphics card is installed in classroom PCs?", {"KB-22"}, "англ."),
 ("У нас в аудиторії після того як прибиральниця посунула системний блок, комп'ютер при включенні довго пищить і на "
  "екрані нічого немає, що могло статися і що робити?", {"KB-05", "KB-07"}, "довгий"),
 ("Після того як я в біосі ввімкнув профіль пам'яті щоб пришвидшити комп почалися сині екрани а без нього все "
  "стабільно, це поломка планки чи щось інше?", {"KB-06"}, "довгий"),
 ("Чим чистити радіатор і чи підійде звичайний пилосос?", {"KB-17", "KB-01"}, "синонім"),
 ("Чи можна самостійно замінити модуль пам'яті на гарантійному ПК?", {"KB-18"}, "довгий"),
 ("шифрувальник змінив розширення файлів що робити", {"KB-15"}, "синонім"),
]
bm_S = BM25Index(S)
METHODS = {"BM25": lambda q, k: bm_S.search(q, k), "dense": lambda q, k: dense_S.search(q, k),
           "hybrid RRF": lambda q, k: rrf([dense_S.search(q, 20), bm_S.search(q, 20)], k)}
def first_rank(res, gold):
    for i, (c, _) in enumerate(res, 1):
        if c["doc"] in gold: return i
    return None
RANKS = {m: [first_rank(f(q, 5), g) for q, g, _ in EVAL] for m, f in METHODS.items()}
rec = lambda ranks, k: np.mean([r is not None and r <= k for r in ranks])
TAB = pd.DataFrame({m: {f"recall@{k}": round(rec(r, k), 3) for k in (1, 3, 5)} for m, r in RANKS.items()}).T
print(f"Питань: {len(EVAL)}; релевантність — документ з допустимого набору у top-k"); print(TAB.to_string())
types = pd.Series([t for *_, t in EVAL])
print("\nrecall@3 за типом питання:")
print(pd.DataFrame({m: pd.Series([r is not None and r <= 3 for r in RANKS[m]]).groupby(types).mean()
                    for m in METHODS}).round(2).assign(n=types.value_counts()).to_string())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Path("figures").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
TAB.T.plot(marker="o", ax=ax, color=["#ED7D31", "#4472C4", "#70AD47"]); ax.set_ylim(0.3, 1.02)
ax.set_title("Recall@k для BM25, dense і гібриду (26 питань)"); ax.grid(alpha=.3); ax.set_ylabel("recall")
fig.tight_layout(); fig.savefig("figures/l11_recall.png", dpi=150); plt.close(fig)

# %% [Р4.2] Де виграє кожен метод: питання з різними рангами
diff = pd.DataFrame({"питання": [q[:60] for q, *_ in EVAL], "тип": types, **{m: RANKS[m] for m in METHODS}})
print(diff[(diff.BM25 != diff.dense) | diff.isna().any(axis=1)].fillna("—").to_string(index=False))

# %% [Р5.1] Аналіз помилок: причини промахів (recall@3, гібрид і dense)
def cause(q, gold):
    """Евристична категоризація з ручною перевіркою (див. звіт)."""
    if not any(c["doc"] in gold for c in S): return "немає в корпусі"
    bm_ok = first_rank(bm_S.search(q, 3), gold) is not None
    best = max((c for c in S if c["doc"] in gold), key=lambda c: float(rag.embed([q])[0] @ dense_S.vecs[S.index(c)]))
    if not bm_ok and best["tokens"] > 128: return "chunking (фрагмент довший за вікно моделі)"
    return "embeddings (семантична схожість нижча за інші фрагменти)" if bm_ok else "лексичний розрив (синоніми/розмовна мова)"
miss = []
for m in ("dense", "hybrid RRF"):
    for (q, g, t), r in zip(EVAL, RANKS[m]):
        if r is None or r > 3:
            top = [c["id"] for c, _ in METHODS[m](q, 3)]
            miss.append({"метод": m, "питання": q[:55], "тип": t, "очікувано": ",".join(sorted(g)), "top-3": top,
                         "причина": cause(q, g)})
M = pd.DataFrame(miss); print(M.to_string(index=False))
print("\n", M.groupby(["метод", "причина"]).size().to_string())

# %% [Р5.2] Покращення: переписування запиту (видалення розмовних слів + розширення синонімами з онтології)
from pcdiag.onto import extract_concepts, ALL_CLASSES
FILLER = r"\b(у нас|в аудиторії|після того як|що могло статися|що робити|і що|це|чи щось інше|щоб|звичайний|а потім)\b"
def rewrite(q):
    q2 = re.sub(FILLER, " ", q.lower())
    terms = [ALL_CLASSES.get(c, "") for c in extract_concepts(q)]
    return re.sub(r"\s+", " ", q2 + " " + " ".join(terms)).strip()
for q, *_ in EVAL[21:24]: print(f"«{q[:70]}…»\n   → «{rewrite(q)[:140]}»")
NEW = {m: [first_rank(f(rewrite(q), 5), g) for q, g, _ in EVAL] for m, f in METHODS.items()}
cmp = pd.DataFrame({f"{m} {k}": {"до": round(rec(RANKS[m], int(k[-1])), 3), "після": round(rec(NEW[m], int(k[-1])), 3)}
                    for m in METHODS for k in ("recall@1", "recall@3")}).T
cmp["Δ"] = (cmp["після"] - cmp["до"]).round(3); print(); print(cmp.to_string())
# Альтернатива: фіксовані фрагменти 400 токенів (базова стратегія Р1) — для порівняння конфігурацій
RF = [first_rank(dense_F.search(q, 5), g) for q, g, _ in EVAL]
print(f"\nDense на фіксованих фрагментах 400 ток.: recall@1={rec(RF, 1):.3f}, @3={rec(RF, 3):.3f}, @5={rec(RF, 5):.3f}")
