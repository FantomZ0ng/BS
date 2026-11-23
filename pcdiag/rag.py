"""RAG над базою знань ІТ-підтримки: очищення, chunking, FAISS, BM25, RRF, відповідь з цитуванням."""
import re, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "11" / "corpus"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMB_MODEL, device="cpu")
    return _model


def ntok(text):
    return len(model().tokenizer.tokenize(text))


# ---------------- завантаження та очищення ----------------
HEADER = re.compile(r"^(БАЗА ЗНАНЬ ІТ-ПІДТРИМКИ|LNU IT SUPPORT KNOWLEDGE BASE).*$")
FOOTER = re.compile(r"^—\s*(Сторінка|Page)\s+(\d+)\s+(з|of)\s+\d+\s*—$")


def load_docs(folder=CORPUS):
    """→ список документів: id, title, lang, lines=[(page, line)] без колонтитулів."""
    docs = []
    for p in sorted(Path(folder).glob("*.md")):
        raw = p.read_text(encoding="utf-8").splitlines()
        page, lines, removed = 1, [], 0
        for l in raw:
            l = re.sub(r"[ \t]+", " ", l).strip()
            if HEADER.match(l):
                removed += 1; continue
            m = FOOTER.match(l)
            if m:
                removed += 1; page = int(m.group(2)) + 1; continue
            lines.append((page, l))
        text = "\n".join(l for _, l in lines)
        cyr = len(re.findall(r"[а-яіїєґ]", text.lower())) / max(1, len(re.findall(r"[a-zа-яіїєґ]", text.lower())))
        title = next(l[2:] for _, l in lines if l.startswith("# "))
        docs.append({"doc": p.stem.split("_")[0], "file": p.name, "title": title,
                     "lang": "uk" if cyr > 0.5 else "en", "lines": lines, "removed": removed,
                     "raw_chars": sum(map(len, raw)), "clean_chars": len(text)})
    return docs


def _sections(d):
    """Розділи ## з номером сторінки, де розділ починається."""
    secs, cur = [], None
    for page, l in d["lines"]:
        if l.startswith("# "):
            continue
        if l.startswith("## "):
            cur = {"section": l[3:], "page": page, "text": []}; secs.append(cur); continue
        if cur is not None and l:
            cur["text"].append(l)
    return secs


# ---------------- стратегії chunking ----------------
def chunk_fixed(docs, size=400, overlap=0.15):
    """Фіксований розмір у токенах моделі embeddings з перекриттям."""
    tok = model().tokenizer
    out = []
    for d in docs:
        body = [(pg, l) for pg, l in d["lines"] if l and not l.startswith("# ")]
        words, pages, secs, cur = [], [], [], ""
        for pg, l in body:
            if l.startswith("## "):
                cur = l[3:]; continue
            for w in l.split():
                words.append(w); pages.append(pg); secs.append(cur)
        wt = [len(tok.tokenize(w)) for w in words]
        step_tokens = int(size * (1 - overlap))
        i, k = 0, 0
        while i < len(words):
            j, t = i, 0
            while j < len(words) and t + wt[j] <= size:
                t += wt[j]; j += 1
            out.append({"id": f"{d['doc']}:f{k}", "doc": d["doc"], "title": d["title"], "section": secs[i],
                        "page": pages[i], "lang": d["lang"], "strategy": "fixed",
                        "text": " ".join(words[i:j]), "tokens": t})
            k += 1
            if j >= len(words): break
            # наступний старт: відступаємо так, щоб перекриття було ≈ overlap*size токенів
            back, s = j, 0
            while back > i + 1 and s < size - step_tokens:
                back -= 1; s += wt[back]
            i = back
    return out


def chunk_structural(docs, max_tokens=200):
    """За структурою: один розділ ## = фрагмент (назва документа + розділ); довгі розділи — по абзацах."""
    out = []
    for d in docs:
        for si, s in enumerate(_sections(d)):
            head = f"{d['title']}. {s['section']}."
            parts, buf = [], []
            for para in s["text"]:
                if buf and ntok(" ".join(buf + [para])) > max_tokens:
                    parts.append(buf); buf = []
                buf.append(para)
            if buf: parts.append(buf)
            for pi, p in enumerate(parts):
                cid = f"{d['doc']}:s{si}" + (f".{pi}" if len(parts) > 1 else "")
                text = head + " " + " ".join(p)
                out.append({"id": cid, "doc": d["doc"], "title": d["title"], "section": s["section"],
                            "page": s["page"], "lang": d["lang"], "strategy": "structural",
                            "text": text, "tokens": ntok(text)})
    return out


# ---------------- векторний індекс ----------------
def embed(texts, batch_size=32):
    return model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False).astype("float32")


class DenseIndex:
    def __init__(self, chunks, vectors=None):
        import faiss
        self.chunks = chunks
        self.vecs = vectors if vectors is not None else embed([c["text"] for c in chunks])
        self.index = faiss.IndexFlatIP(self.vecs.shape[1])
        self.index.add(self.vecs)

    def save(self, folder):
        import faiss
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        (folder / "index.faiss").write_bytes(faiss.serialize_index(self.index).tobytes())  # шлях з кирилицею
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, folder):
        import faiss
        folder = Path(folder)
        chunks = [json.loads(l) for l in open(folder / "chunks.jsonl", encoding="utf-8")]
        obj = cls.__new__(cls)
        obj.chunks, obj.index = chunks, faiss.deserialize_index(np.frombuffer((folder / "index.faiss").read_bytes(), dtype="uint8"))
        obj.vecs = obj.index.reconstruct_n(0, obj.index.ntotal)
        return obj

    def search(self, query, k=5, where=None):
        """top-k з фільтром метаданих where={'lang': 'uk', 'doc': [...]} через IDSelector FAISS."""
        import faiss
        q = embed([query])
        params = None
        if where:
            ok = [i for i, c in enumerate(self.chunks)
                  if all((c[key] in v) if isinstance(v, (list, set, tuple)) else c[key] == v for key, v in where.items())]
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.array(ok, dtype="int64")))
        D, I = self.index.search(q, k, params=params)
        return [(self.chunks[i], float(s)) for s, i in zip(D[0], I[0]) if i >= 0]


# ---------------- BM25 і гібрид ----------------
def tokenize(text):
    return re.findall(r"[a-zа-яіїєґ0-9_']+|0x[0-9a-f]+", text.lower().replace("’", "'"))


class BM25Index:
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query, k=5):
        sc = self.bm.get_scores(tokenize(query))
        idx = np.argsort(-sc)[:k]
        return [(self.chunks[i], float(sc[i])) for i in idx]


def rrf(result_lists, k=5, c=60):
    """Reciprocal Rank Fusion: score = Σ 1/(c + rank)."""
    score, by_id = {}, {}
    for res in result_lists:
        for rank, (ch, _) in enumerate(res, 1):
            score[ch["id"]] = score.get(ch["id"], 0) + 1 / (c + rank); by_id[ch["id"]] = ch
    best = sorted(score, key=lambda i: -score[i])[:k]
    return [(by_id[i], score[i]) for i in best]


class HybridRetriever:
    def __init__(self, chunks, dense=None):
        self.dense = dense or DenseIndex(chunks)
        self.bm25 = BM25Index(chunks)
        self.chunks = chunks

    def search(self, query, k=5, method="hybrid", pool=20):
        if method == "dense": return self.dense.search(query, k)
        if method == "bm25": return self.bm25.search(query, k)
        return rrf([self.dense.search(query, pool), self.bm25.search(query, pool)], k)


# ---------------- RAG-відповідь ----------------
NO_ANSWER = "Недостатньо даних"
RAG_SYSTEM = (
    "Ти — асистент служби ІТ-підтримки. Відповідай українською ЛИШЕ на основі наведених фрагментів бази знань.\n"
    "Правила:\n1. Кожне твердження супроводжуй посиланням на джерело у квадратних дужках, напр. [KB-05:s1].\n"
    "2. Використовуй лише ідентифікатори фрагментів, наведені нижче; не вигадуй фактів.\n"
    f"3. Якщо у фрагментах немає відповіді — відповідай рівно: «{NO_ANSWER}».\n"
    "4. Відповідь — 1–4 речення, без вступу.")


def build_prompt(question, hits, extra=""):
    ctx = "\n\n".join(f"[{c['id']}] ({c['title']} / {c['section']})\n{c['text']}" for c, _ in hits)
    user = f"Фрагменти:\n{ctx}\n{extra}\nПитання: {question}"
    return [{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": user}]


def answer(question, retriever, k=4, llm=None, **kw):
    from .llm import chat
    hits = retriever.search(question, k) if not isinstance(retriever, list) else retriever
    r = (llm or chat)(build_prompt(question, hits), json_mode=False, max_tokens=220, **kw)
    text = r["text"].strip()
    cites = sorted(set(re.findall(r"\[([A-Z]{2}-\d{2}:[sf]\d+(?:\.\d+)?)\]", text)))
    return {"question": question, "answer": text, "citations": cites, "hits": [c["id"] for c, _ in hits],
            "refused": NO_ANSWER.lower() in text.lower(), **{k_: r[k_] for k_ in ("tokens_in", "tokens_out", "latency_s", "cost_usd")}}


STOP = set("і й та а але або що це як для на в у з із до від по при за не є чи ж же the a of to and in is for".split())


def check_citations(ans, chunks_by_id, threshold=0.35):
    """Автоперевірка: для кожного речення з посиланням — частка його змістовних слів, наявних у цитованому фрагменті,
    і косинусна схожість embeddings. Підтверджено, якщо overlap ≥ threshold або cos ≥ 0.6."""
    results = []
    text = ans["answer"]
    pat = re.compile(r"\[([A-Z]{2}-\d{2}:[sf]\d+(?:\.\d+)?)\]")
    prev_end, last_claim = 0, ""
    for m in pat.finditer(text):
        seg = re.sub(r"\[[^\]]+\]", "", text[prev_end:m.start()]).strip(" .,;:\n")
        # посилання після крапки або кілька посилань поспіль стосуються попереднього речення
        claim = seg if len(seg) >= 15 else last_claim
        prev_end, last_claim = m.end(), claim
        cid = m.group(1)
        words = [w for w in tokenize(claim) if w not in STOP and len(w) > 2]
        ch = chunks_by_id.get(cid)
        if ch is None:
            results.append({"id": cid, "overlap": 0.0, "cos": 0.0, "ok": False, "reason": "id не існує"}); continue
        stems = {w[:5] for w in tokenize(ch["text"])}
        ov = np.mean([w[:5] in stems for w in words]) if words else 0.0
        cos = float(embed([claim])[0] @ embed([ch["text"]])[0]) if claim else 0.0
        results.append({"id": cid, "overlap": round(float(ov), 2), "cos": round(cos, 2),
                        "ok": bool(ov >= threshold or cos >= 0.6), "reason": ""})
    return results


# ---------------- переранжування (cross-encoder) ----------------
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_ce = None


def rerank(query, hits, k=4):
    """Переранжування кандидатів багатомовним cross-encoder (запит і фрагмент кодуються разом)."""
    global _ce
    if _ce is None:
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder(RERANK_MODEL, device="cpu")
    sc = _ce.predict([(query, c["text"]) for c, _ in hits])
    order = np.argsort(-sc)[:k]
    return [(hits[i][0], float(sc[i])) for i in order]
