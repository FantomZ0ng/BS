"""Веб-інтерфейс: питання → відповідь, джерела з цитатами, підграф зв'язків."""
import os, httpx
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Асистент ІТ-підтримки", page_icon="🛠", layout="wide")
st.title("🛠 Асистент ІТ-підтримки ЛНУ")
st.caption("RAG над базою знань + граф знань Neo4j + локальна LLM")

q = st.text_area("Опишіть проблему або поставте питання", value=st.query_params.get("q", ""), height=90, max_chars=500)
use_graph = st.checkbox("Використовувати граф знань", value=True)
go = st.button("Запитати", type="primary") or bool(st.query_params.get("q"))

if go and q.strip():
    with st.spinner("Шукаю у базі знань і графі, генерую відповідь…"):
        try:
            r = httpx.post(f"{API}/ask", json={"question": q, "use_graph": use_graph}, timeout=180)
            data = r.json()
        except httpx.HTTPError as e:
            st.error(f"API недоступне: {type(e).__name__}"); st.stop()
    if r.status_code != 200:
        err = data.get("error") or data.get("detail")
        st.error(f"Помилка {r.status_code}: {err}"); st.stop()
    (st.warning if data["refused"] else st.success)(data["answer"])
    st.caption(f"request_id {data['request_id']} · {data['latency_ms'] / 1000:.1f} с · "
               f"токени {data['tokens_in']}/{data['tokens_out']} · {data['model']} · {data['prompt_version']}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Джерела")
        if not data["citations"]:
            st.info("Відповідь не містить посилань на фрагменти.")
        for c in data["citations"]:
            st.markdown(f"**[{c['id']}] {c['title']}** — {c['section']}")
            st.markdown(f"> {c['quote']}…")
    with c2:
        st.subheader("Зв'язки графа")
        st.write("Виявлені поняття:", ", ".join(data["concepts"]) or "—")
        if data["graph_edges"]:
            dot = "digraph G { rankdir=LR; node [shape=box, style=rounded, fontsize=10]; edge [fontsize=8];\n"
            dot += "\n".join(f'"{e["source"]}" -> "{e["target"]}" [label="{e["rel"]}"];' for e in data["graph_edges"][:18])
            st.graphviz_chart(dot + "}")
        else:
            st.info("Граф не використано для цього питання.")
elif go:
    st.error("Введіть питання.")
