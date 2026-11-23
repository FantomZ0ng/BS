"""Ядро системи: векторний контекст (FAISS + BM25), графовий контекст (Neo4j), злиття, промпт, LLM."""
import re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcdiag import rag                                   # noqa: E402
from pcdiag.llm import chat, LLMError, MODEL              # noqa: E402
from pcdiag.onto import extract_concepts, neo4j_driver, ALL_CLASSES   # noqa: E402

INDEX_DIR = ROOT / "11" / "index" / "structural"
PROMPT_VERSION = "rag-graph-v1"
TOKEN_BUDGET = 1400            # обмеження контексту (токени моделі embeddings як наближення)


class GraphUnavailable(RuntimeError):
    pass


class System:
    def __init__(self, llm=chat):
        self.dense = rag.DenseIndex.load(INDEX_DIR)
        self.retriever = rag.HybridRetriever(self.dense.chunks, dense=self.dense)
        self.by_id = {c["id"]: c for c in self.dense.chunks}
        self.llm = llm
        self._driver = None

    # ---------- джерела ----------
    def driver(self):
        if self._driver is None:
            self._driver = neo4j_driver()
        return self._driver

    def vector_context(self, q, k=4, rerank=False, pool=12):
        """top-k фрагментів (гібрид BM25 + dense, RRF) з метаданими; опційно — переранжування cross-encoder."""
        hits = self.retriever.search(q, pool if rerank else k)
        if rerank:
            t = time.perf_counter(); hits = rag.rerank(q, hits, k); self.last_rerank_ms = (time.perf_counter() - t) * 1000
        return [{"id": c["id"], "doc": c["doc"], "title": c["title"], "section": c["section"], "page": c["page"],
                 "lang": c["lang"], "text": c["text"], "score": round(s, 4)} for c, s in hits]

    GRAPH_Q = """
    UNWIND $concepts AS cname
    MATCH (c:Class {name: cname})
    // індивіди класу поняття або його підкласів, далі сусіди на відстані ≤ 2
    MATCH (i:Individual)-[:INSTANCE_OF]->(:Class)-[:SUBCLASS_OF*0..2]->(c)
    MATCH p = (i)-[r1]-(n1:Individual)
    WHERE type(r1) <> 'INSTANCE_OF'
    OPTIONAL MATCH (n1)-[r2]-(n2:Individual)
    WHERE type(r2) IN ['RESOLVED_BY', 'CONFIRMED_BY', 'AFFECTS', 'MANIFESTS_AS'] AND n2 <> i
    WITH i, r1, n1, r2, n2
    MATCH (i)-[:INSTANCE_OF]->(ci), (n1)-[:INSTANCE_OF]->(c1)
    OPTIONAL MATCH (n2)-[:INSTANCE_OF]->(c2)
    RETURN DISTINCT ci.name AS a, type(r1) AS r, c1.name AS b, startNode(r1) = i AS fwd,
           type(r2) AS r2, c2.name AS c, (r2 IS NULL OR startNode(r2) = n1) AS fwd2
    LIMIT 60
    """

    def graph_context(self, q):
        """Сутності запиту → сусіди й шляхи в Neo4j (глибина ≤ 2) → список трійок (клас, зв'язок, клас)."""
        concepts = extract_concepts(q)
        if not concepts:
            return concepts, []
        try:
            recs, _, _ = self.driver().execute_query(self.GRAPH_Q, concepts=concepts)
        except Exception as e:  # ServiceUnavailable, AuthError тощо
            self._driver = None
            raise GraphUnavailable(type(e).__name__) from e
        edges = set()
        for r in recs:
            edges.add((r["a"], r["r"], r["b"]) if r["fwd"] else (r["b"], r["r"], r["a"]))
            if r["r2"] and r["c"]:
                edges.add((r["b"], r["r2"], r["c"]) if r["fwd2"] else (r["c"], r["r2"], r["b"]))
        # прибираємо зв'язки з випадками (заявки — шум для відповіді)
        edges = sorted(e for e in edges if "Case" not in e[0] and "Computer" not in e[2] and e[1] != "HAS_SYMPTOM")
        return concepts, edges

    @staticmethod
    def edges_text(edges, concepts=()):
        lab = lambda c: f"{c} ({ALL_CLASSES.get(c, c)})"
        lines = [f"- {lab(a)} —{r}→ {lab(b)}" for a, r, b in edges]
        # багатокроковий висновок: несправності, що пояснюють найбільше симптомів/понять запиту
        cover = {}
        for a, r, b in edges:
            if r in ("MANIFESTS_AS", "AFFECTS") and b in concepts:
                cover.setdefault(a, set()).add(b)
        if cover:
            best = max(len(v) for v in cover.values())
            for f in sorted(k for k, v in cover.items() if len(v) == best):
                conf = [b for a, r, b in edges if a == f and r == "CONFIRMED_BY"]
                fix = [b for a, r, b in edges if a == f and r == "RESOLVED_BY"]
                comp = [b for a, r, b in edges if a == f and r == "AFFECTS"]
                lines.append(f"ВИСНОВОК ГРАФА: {lab(f)} пояснює {best} з {len(set(concepts))} понять запиту "
                             f"({', '.join(sorted(cover[f]))}); компонент: {', '.join(map(lab, comp)) or '—'}; "
                             f"підтвердження: {', '.join(map(lab, conf)) or '—'}; усунення: {', '.join(map(lab, fix)) or '—'}")
        return "\n".join(lines)

    # ---------- злиття та промпт ----------
    def merge(self, chunks, edges):
        seen, kept, used = set(), [], 0
        for c in chunks:                       # дедуплікація за id і за текстом
            key = c["text"][:120]
            if c["id"] in seen or key in seen:
                continue
            t = rag.ntok(c["text"])
            if used + t > TOKEN_BUDGET:
                break
            seen |= {c["id"], key}; kept.append(c); used += t
        g_lines = edges[: max(0, (TOKEN_BUDGET + 600 - used) // 25)]
        return kept, g_lines, used

    SYSTEM = (rag.RAG_SYSTEM + "\n5. Розділ «Зв'язки графа» містить перевірені зв'язки онтології (несправність → симптом, "
              "компонент, дія підтвердження, дія усунення). Використовуй їх, щоб пов'язати симптоми з несправністю "
              "й дією; рядок «ВИСНОВОК ГРАФА» — результат багатокрокового виведення: якщо він є, назви цю несправність, "
              "компонент і дію підтвердження та познач посиланням [граф].")

    def ask(self, question, k=4, use_graph=True, use_vector=True, rerank=False):
        timings, t0 = {}, time.perf_counter()
        self.last_rerank_ms = 0.0
        chunks = self.vector_context(question, k, rerank=rerank) if use_vector else []
        timings["rerank_ms"] = self.last_rerank_ms
        timings["vector_ms"] = (time.perf_counter() - t0) * 1000 - self.last_rerank_ms
        t1 = time.perf_counter()
        concepts, edges = self.graph_context(question) if use_graph else ([], [])
        timings["graph_ms"] = (time.perf_counter() - t1) * 1000
        kept, g_edges, ctx_tokens = self.merge(chunks, edges)
        frag = "\n\n".join(f"[{c['id']}] ({c['title']} / {c['section']})\n{c['text']}" for c in kept)
        user = (f"Фрагменти:\n{frag or '(немає)'}\n\nЗв'язки графа:\n{self.edges_text(g_edges, concepts) or '(немає)'}\n\n"
                f"Питання: {question}")
        t2 = time.perf_counter()
        r = self.llm([{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": user}],
                     json_mode=False, max_tokens=260)
        timings["llm_ms"] = r["latency_s"] * 1000   # виміряно під час виклику LLM (для кешованих — первинний вимір)
        text = r["text"].strip()
        cites = sorted(set(re.findall(r"\[([A-Z]{2}-\d{2}:[sf]\d+(?:\.\d+)?)\]", text)))
        return {"answer": text, "refused": rag.NO_ANSWER.lower() in text.lower(),
                "citations": [{"id": c, "title": self.by_id[c]["title"], "section": self.by_id[c]["section"],
                               "quote": self.by_id[c]["text"][:220]} for c in cites if c in self.by_id],
                "invalid_citations": [c for c in cites if c not in self.by_id],
                "chunks": [c["id"] for c in kept], "concepts": concepts,
                "graph_edges": [{"source": a, "rel": rr, "target": b} for a, rr, b in g_edges],
                "graph_used": "[граф]" in text, "context_tokens": ctx_tokens,
                "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"], "cost_usd": r["cost_usd"],
                "timings_ms": {k_: round(v, 1) for k_, v in timings.items()}, "model": MODEL,
                "prompt_version": PROMPT_VERSION}

    def health(self):
        st = {"index": {"ok": self.dense.index.ntotal > 0, "chunks": self.dense.index.ntotal}}
        try:
            self.driver().verify_connectivity(); st["neo4j"] = {"ok": True}
        except Exception as e:
            self._driver = None; st["neo4j"] = {"ok": False, "error": type(e).__name__}
        try:
            import httpx
            httpx.get(f"{__import__('pcdiag.llm').llm.OLLAMA}/api/version", timeout=2).raise_for_status(); st["llm"] = {"ok": True, "model": MODEL}
        except Exception as e:
            st["llm"] = {"ok": False, "error": type(e).__name__}
        return st
