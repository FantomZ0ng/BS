# %% [Р1.1] Підключення до Neo4j і схема відображення OWL → властивісний граф
import os, re, time, statistics
from pathlib import Path
import pandas as pd
from rdflib import Graph, RDF, RDFS, OWL, URIRef, Literal
from neo4j import GraphDatabase

pd.set_option("display.width", 160)
# пароль — лише у файлі ../.env (не в коді і не у звіті)
env = dict(l.strip().split("=", 1) for l in Path("../.env").read_text().splitlines() if "=" in l)
driver = GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]),
                              notifications_min_severity="OFF")
driver.verify_connectivity()
info = driver.execute_query("CALL dbms.components() YIELD name, versions, edition RETURN *").records[0]
print("Підключено:", info["name"], info["versions"][0], info["edition"])

def run(q, **p):
    """Виконати Cypher і повернути DataFrame."""
    recs, summary, keys = driver.execute_query(q, p)
    return pd.DataFrame([r.values() for r in recs], columns=keys)

MAPPING = [
    ("owl:Class ex:X", "вузол (:Class {iri, name, label})", "клас як вузол, щоб зберегти ієрархію та запити до неї"),
    ("rdfs:subClassOf", "ребро [:SUBCLASS_OF]", "лише іменовані суперкласи; анонімні обмеження — втрачаються"),
    ("індивід (owl:NamedIndividual)", "вузол (:Individual:<Категорія>)", "категорія = кореневий клас: Component, Fault, Symptom, Action, DiagnosticCase"),
    ("rdf:type", "ребро [:INSTANCE_OF]", "до прямого (асертованого) класу"),
    ("object property hasPart, affects, …", "тип ребра HAS_PART, AFFECTS, …", "camelCase → UPPER_SNAKE_CASE"),
    ("data property cpuTemperature, …", "атрибут вузла", "xsd:float → Float, xsd:dateTime → DateTime"),
    ("rdfs:label@uk", "атрибут label", "українська назва"),
]
print(pd.DataFrame(MAPPING, columns=["OWL/RDF", "Neo4j", "Примітка"]).to_string(index=False))

# %% [Р1.2] Схема властивісного графа (мітки, типи ребер, ключові атрибути)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
Path("figures").mkdir(exist_ok=True)
POS = {"Class": (5, 5.0), "DiagnosticCase": (0.6, 2.6), "Symptom": (3.0, 0.3), "Fault": (5.4, 2.6),
       "Component": (9.6, 2.6), "Action": (7.4, 0.3)}
TXT = {"Class": "Class\n{iri, name, label}", "DiagnosticCase": "DiagnosticCase\n{cpuTemperature,\nreportedOn}",
       "Symptom": "Symptom", "Fault": "Fault", "Component": "Component\n{modelName, ageYears}",
       "Action": "Action\n(Repair / Diagnostic)"}
COL = {"Class": "#D9E2F3", "DiagnosticCase": "#FCE4D6", "Symptom": "#E2F0D9", "Fault": "#FFF2CC",
       "Component": "#DDEBF7", "Action": "#F4CCCC"}
E = [("DiagnosticCase", "Symptom", "HAS_SYMPTOM", 0), ("DiagnosticCase", "Fault", "DIAGNOSED_AS", 0),
     ("DiagnosticCase", "Component", "CONCERNS_COMPUTER", -0.25), ("Fault", "Symptom", "MANIFESTS_AS", 0),
     ("Fault", "Component", "AFFECTS", 0), ("Fault", "Action", "RESOLVED_BY /\nCONFIRMED_BY", 0),
     ("Fault", "Class", "INSTANCE_OF\n(усі індивіди)", 0)]
fig, ax = plt.subplots(figsize=(12, 6.6)); ax.axis("off"); ax.set_xlim(-1, 11.5); ax.set_ylim(-0.7, 6.4)
for a, b, lab, rad in E:
    (x1, y1), (x2, y2) = POS[a], POS[b]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#555",
                connectionstyle=f"arc3,rad={rad}", shrinkA=40, shrinkB=40))
    ty = (y1 + y2) / 2 + (0.15 if rad == 0 else 0.95)
    ax.text((x1 + x2) / 2 + (0.9 if b == "Class" else 0), ty, lab, fontsize=8.5, ha="center", color="#7030A0", weight="bold")
ax.annotate("", xy=(5.8, 5.45), xytext=(4.2, 5.45), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#555",
            connectionstyle="arc3,rad=-1.2"))
ax.text(5, 6.2, "SUBCLASS_OF", fontsize=8.5, ha="center", color="#7030A0", weight="bold")
ax.annotate("", xy=(10.3, 3.0), xytext=(10.7, 2.5), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#555",
            connectionstyle="arc3,rad=-2.2"))
ax.text(10.6, 3.5, "HAS_PART", fontsize=8.5, color="#7030A0", weight="bold")
for k, (x, y) in POS.items():
    ax.text(x, y, TXT[k], ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.6", fc=COL[k], ec="#333", lw=1.2))
ax.text(-0.9, -0.6, "Мітки індивідів: :Individual + категорія (DiagnosticCase, Symptom, Fault, Component, Action); "
        "унікальні ключі: Class.iri, Individual.iri", fontsize=8.5, style="italic")
ax.set_title("Схема властивісного графа онтології PC Diagnostics у Neo4j", weight="bold")
fig.tight_layout(); fig.savefig("figures/l9_schema.png", dpi=150); plt.close(fig)
print("Збережено figures/l9_schema.png")

# %% [Р1.3] Обмеження унікальності та індекси
run("MATCH (n) DETACH DELETE n")                      # чиста база для відтворюваності
for q in ["DROP CONSTRAINT class_iri IF EXISTS", "DROP CONSTRAINT ind_iri IF EXISTS",
          "DROP CONSTRAINT component_id IF EXISTS",
          "DROP INDEX class_name IF EXISTS", "DROP INDEX ind_name IF EXISTS", "DROP INDEX ind_label IF EXISTS"]:
    run(q)
SCHEMA = [
    "CREATE CONSTRAINT class_iri IF NOT EXISTS FOR (c:Class) REQUIRE c.iri IS UNIQUE",
    "CREATE CONSTRAINT ind_iri IF NOT EXISTS FOR (i:Individual) REQUIRE i.iri IS UNIQUE",
    "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (d:Component) REQUIRE d.name IS UNIQUE",
    "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)",
    "CREATE INDEX ind_name IF NOT EXISTS FOR (i:Individual) ON (i.name)",
    "CREATE TEXT INDEX ind_label IF NOT EXISTS FOR (c:Class) ON (c.label)",
]
for q in SCHEMA:
    run(q)
run("CALL db.awaitIndexes()")
print(run("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties").to_string(index=False))
print()
print(run("SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state "
          "WHERE type <> 'LOOKUP'").to_string(index=False))

# %% [Р2.1] Власний ETL: rdflib → neo4j, лише MERGE (ідемпотентний)
EX = "http://example.org/pcdiag#"
g = Graph().parse("../8/pcdiag.ttl")
short = lambda u: str(u).replace(EX, "")
snake = lambda s: re.sub(r"(?<!^)(?=[A-Z])", "_", s).upper()
ROOTS = ["Component", "Fault", "Symptom", "Action", "DiagnosticCase"]

classes = {c for c in g.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef) and str(c).startswith(EX)}
subs = [(s, o) for s, o in g.subject_objects(RDFS.subClassOf) if s in classes and o in classes]
inds = {i for i in g.subjects(RDF.type, OWL.NamedIndividual)}
obj_props = {p for p in g.subjects(RDF.type, OWL.ObjectProperty)}
data_props = {p for p in g.subjects(RDF.type, OWL.DatatypeProperty)}

def root_of(cls):
    """Кореневий клас (категорія) для мітки індивіда — через транзитивний subClassOf."""
    for anc in g.transitive_objects(cls, RDFS.subClassOf):
        if short(anc) in ROOTS:
            return short(anc)
    return "Thing"

def to_neo(v):
    """Літерал RDF → тип Neo4j (decimal → float, dateTime → ISO-рядок, далі datetime())."""
    x = v.toPython()
    if hasattr(x, "isoformat"): return x.isoformat()
    if type(x).__name__ == "Decimal": return float(x)
    return x

def etl():
    with driver.session() as s:
        s.run("""UNWIND $rows AS r MERGE (c:Class {iri: r.iri})
                 SET c.name = r.name, c.label = r.label""",
              rows=[{"iri": str(c), "name": short(c), "label": str(g.value(c, RDFS.label) or short(c))}
                    for c in classes])
        s.run("""UNWIND $rows AS r MATCH (a:Class {iri: r.s}), (b:Class {iri: r.o})
                 MERGE (a)-[:SUBCLASS_OF]->(b)""", rows=[{"s": str(a), "o": str(b)} for a, b in subs])
        for i in inds:
            t = [c for c in g.objects(i, RDF.type) if c in classes][0]
            props = {short(p): to_neo(v) for p, v in g.predicate_objects(i) if p in data_props}
            label = root_of(t) if short(t) != "Computer" else "Component"
            s.run(f"""MERGE (n:Individual {{iri: $iri}}) SET n:{label}, n.name = $name, n += $props
                      WITH n MATCH (c:Class {{iri: $cls}}) MERGE (n)-[:INSTANCE_OF]->(c)""",
                  iri=str(i), name=short(i), props=props, cls=str(t))
        s.run("MATCH (n:DiagnosticCase) WHERE n.reportedOn IS NOT NULL SET n.reportedOn = datetime(n.reportedOn)")
        for p in obj_props:
            pairs = [{"s": str(a), "o": str(b)} for a, b in g.subject_objects(p) if a in inds and b in inds]
            s.run(f"""UNWIND $rows AS r MATCH (a:Individual {{iri: r.s}}), (b:Individual {{iri: r.o}})
                      MERGE (a)-[:{snake(short(p))}]->(b)""", rows=pairs)

def counters():
    n = run("MATCH (n) RETURN count(n) AS n").n[0]
    r = run("MATCH ()-[r]->() RETURN count(r) AS r").r[0]
    return int(n), int(r)

t = time.perf_counter(); etl(); print(f"Імпорт 1: {time.perf_counter() - t:.2f} с, вузлів/ребер =", counters())
t = time.perf_counter(); etl(); print(f"Імпорт 2: {time.perf_counter() - t:.2f} с, вузлів/ребер =", counters())

# %% [Р2.2] Контрольні лічильники: граф vs джерело (трійки .ttl)
print(run("MATCH (n) RETURN labels(n) AS labels, count(*) AS n ORDER BY n DESC").to_string(index=False))
print()
print(run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC").to_string(index=False))
obj_triples = sum(1 for p in obj_props for a, b in g.subject_objects(p) if a in inds and b in inds)
dp_triples = sum(1 for p in data_props for _ in g.subject_objects(p))
src = {"класи (owl:Class, ex:)": len(classes), "subClassOf між іменованими класами": len(subs),
       "індивіди": len(inds), "rdf:type індивідів": len(inds),
       "трійки object property між індивідами": obj_triples, "трійки data property": dp_triples}
neo = {"класи (owl:Class, ex:)": run("MATCH (c:Class) RETURN count(c) AS n").n[0],
       "subClassOf між іменованими класами": run("MATCH ()-[r:SUBCLASS_OF]->() RETURN count(r) AS n").n[0],
       "індивіди": run("MATCH (i:Individual) RETURN count(i) AS n").n[0],
       "rdf:type індивідів": run("MATCH ()-[r:INSTANCE_OF]->() RETURN count(r) AS n").n[0],
       "трійки object property між індивідами":
           run("MATCH (:Individual)-[r]->(:Individual) RETURN count(r) AS n").n[0],
       "трійки data property": run("MATCH (i:Individual) UNWIND [k IN keys(i) WHERE NOT k IN ['iri','name']] AS k "
                                   "RETURN count(k) AS n").n[0]}
cmp_df = pd.DataFrame({"джерело (.ttl)": src, "Neo4j": neo})
cmp_df["збіг"] = cmp_df.iloc[:, 0] == cmp_df.iloc[:, 1]
print(); print("Усього трійок у pcdiag.ttl:", len(g)); print(cmp_df.to_string())

# %% [Р3.1] Запити Cypher за компетентнісними питаннями
CQ = {
 "C1 (CQ1) Які симптоми має відмова блока живлення? — MATCH": """
MATCH (f:Fault)-[:INSTANCE_OF]->(:Class {name:'PSUFailure'}), (f)-[:MANIFESTS_AS]->(s)-[:INSTANCE_OF]->(c)
RETURN s.name AS symptom, c.label AS class""",
 "C2 (CQ8) Випадки з температурою CPU > 90 °C — WHERE + ORDER BY": """
MATCH (k:DiagnosticCase) WHERE k.cpuTemperature > 90
RETURN k.name AS case, k.cpuTemperature AS t, date(k.reportedOn) AS date ORDER BY t DESC""",
 "C3 (CQ10) Випадки без діагнозу — OPTIONAL MATCH": """
MATCH (k:DiagnosticCase) OPTIONAL MATCH (k)-[:DIAGNOSED_AS]->(f)
WITH k, f WHERE f IS NULL RETURN k.name AS case, date(k.reportedOn) AS date""",
 "C4 (CQ6) Скільки несправностей стосується кожного компонента — count, collect": """
MATCH (f:Fault)-[:AFFECTS]->(c:Component)
RETURN c.name AS component, c.modelName AS model, count(f) AS n, collect(f.name) AS faults ORDER BY n DESC, component""",
 "C5 Середня температура CPU за типом несправності — avg": """
MATCH (k:DiagnosticCase) WHERE k.cpuTemperature IS NOT NULL
OPTIONAL MATCH (k)-[:DIAGNOSED_AS]->(f)-[:INSTANCE_OF]->(c)
RETURN coalesce(c.label, '(без діагнозу)') AS fault, round(avg(k.cpuTemperature), 1) AS avg_t, count(k) AS cases ORDER BY avg_t DESC""",
 "C6 (CQ9) Дії, що підтверджують несправність — параметр $fault": """
MATCH (f:Fault)-[:INSTANCE_OF]->(:Class {name: $fault}) MATCH (f)-[:CONFIRMED_BY]->(d)-[:INSTANCE_OF]->(dc)
OPTIONAL MATCH (f)-[:RESOLVED_BY]->(a)-[:INSTANCE_OF]->(ac)
RETURN f.name AS fault, dc.label AS confirm, ac.label AS repair""",
 "C7 Топ-3 вузли-контейнери за кількістю прямих частин — ORDER BY + LIMIT": """
MATCH (c:Component)-[:HAS_PART]->(p) RETURN c.name AS whole, count(p) AS parts, collect(p.name) AS list
ORDER BY parts DESC LIMIT 3""",
}
PARAMS = {"C6 (CQ9) Дії, що підтверджують несправність — параметр $fault": {"fault": "BadSectors"}}
for name, q in CQ.items():
    df = run(q, **PARAMS.get(name, {}))
    print(f"--- {name}  [{len(df)} рядк.]"); print(df.to_string(index=False)); print()
print("Той самий C6 з іншим параметром fault='RAMModuleDefect':")
print(run(CQ[list(CQ)[5]], fault="RAMModuleDefect").to_string(index=False))

# %% [Р3.2] PROFILE: кількість db hits до і після використання індексу
def profile(q, **p):
    _, summ, _ = driver.execute_query("PROFILE " + q, p)
    def walk(op, depth=0, acc=None):
        acc = acc if acc is not None else []
        acc.append(("  " * depth + op["operatorType"].split("@")[0], op.get("dbHits", 0), op.get("rows", 0)))
        for ch in op.get("children", []):
            walk(ch, depth + 1, acc)
        return acc
    plan = walk(summ.profile)
    return plan, sum(h for _, h, _ in plan)

q_idx = "MATCH (i:Individual {name: $n})-[:INSTANCE_OF]->(c) RETURN i.name, c.label"
q_scan = "MATCH (i) WHERE i.name = $n MATCH (i)-[:INSTANCE_OF]->(c) RETURN i.name, c.label"
for title, q in [("з міткою та індексом ind_name", q_idx), ("без мітки (сканування)", q_scan)]:
    plan, total = profile(q, n="case3")
    print(f"PROFILE {title}: усього db hits = {total}")
    for op, h, rows in plan:
        print(f"   {op:<32} dbHits={h:<5} rows={rows}")

# %% [Р4.1] Шляхи змінної довжини: предки й нащадки класу
print("Предки RAMModuleDefect (SUBCLASS_OF*1..):")
print(run("""MATCH p = (:Class {name:'RAMModuleDefect'})-[:SUBCLASS_OF*1..]->(a:Class)
             RETURN a.name AS ancestor, a.label AS label, length(p) AS depth ORDER BY depth""").to_string(index=False))
print("\nНащадки HardwareFault (усі рівні) з кількістю індивідів:")
print(run("""MATCH p = (d:Class)-[:SUBCLASS_OF*1..]->(:Class {name:'HardwareFault'})
             OPTIONAL MATCH (i)-[:INSTANCE_OF]->(d)
             RETURN d.name AS descendant, length(p) AS depth, count(i) AS individuals
             ORDER BY depth, descendant""").to_string(index=False))
print("\nУсі частини pc1 транзитивно (HAS_PART*):")
print(run("""MATCH p = (:Individual {name:'pc1'})-[:HAS_PART*]->(x)
             RETURN x.name AS part, length(p) AS level ORDER BY level, part""").T.to_string(header=False))

# %% [Р4.2] Найкоротший шлях і ступені вузлів
for a, b in [("s_beeps", "a_replace"), ("case5", "d_temp")]:
    df = run("""MATCH (a:Individual {name:$a}), (b:Individual {name:$b}),
                p = shortestPath((a)-[*..8]-(b))
                RETURN [n IN nodes(p) | n.name] AS nodes, [r IN relationships(p) | type(r)] AS rels,
                       length(p) AS len""", a=a, b=b)
    nodes, rels = df.nodes[0], df.rels[0]
    print(f"shortestPath({a} → {b}), довжина {df.len[0]}:")
    print("   " + nodes[0] + "".join(f" -[{r}]- {n}" for r, n in zip(rels, nodes[1:])))
print("\nТоп-5 найбільш зв'язаних індивідів (degree, усі ребра крім INSTANCE_OF):")
print(run("""MATCH (n:Individual) RETURN n.name AS node, labels(n)[1] AS kind,
             COUNT { (n)-[r]-() WHERE type(r) <> 'INSTANCE_OF' } AS degree
             ORDER BY degree DESC, node LIMIT 5""").to_string(index=False))
print("\nТоп-5 класів за degree (SUBCLASS_OF + INSTANCE_OF):")
print(run("""MATCH (c:Class) RETURN c.name AS class, COUNT { (c)--() } AS degree,
             COUNT { (c)<-[:SUBCLASS_OF]-() } AS subclasses
             ORDER BY degree DESC, class LIMIT 5""").to_string(index=False))

# %% [Р4.3] Скріншот підграфа в Neo4j Browser (Playwright + Microsoft Edge)
from playwright.sync_api import sync_playwright
Path("figures").mkdir(exist_ok=True)
SUBGRAPH = ("MATCH p=(k:DiagnosticCase)-[:DIAGNOSED_AS]->(f)-[:AFFECTS|RESOLVED_BY|CONFIRMED_BY]->(x) "
            "WHERE k.name IN ['case3','case6'] RETURN p")
def browser_shot():
    """Sync API Playwright не працює в циклі asyncio Jupyter — запускаємо в окремому потоці."""
    with sync_playwright() as pw:
        br = pw.chromium.launch(channel="msedge", headless=True)
        page = br.new_page(viewport={"width": 1500, "height": 950})
        page.goto("http://localhost:7474/browser/")
        page.wait_for_selector("input[data-testid='boltaddress']", timeout=60000)
        page.fill("input[data-testid='username']", env["NEO4J_USER"])
        page.fill("input[data-testid='password']", env["NEO4J_PASSWORD"])
        page.click("button[data-testid='connect']")
        page.wait_for_timeout(4000)
        page.locator(".monaco-editor").first.click()
        page.keyboard.type(SUBGRAPH); page.keyboard.press("Control+Enter")
        page.wait_for_timeout(7000)
        page.screenshot(path="figures/l9_browser_subgraph.png", clip={"x": 60, "y": 125, "width": 1440, "height": 545})
        br.close()
from concurrent.futures import ThreadPoolExecutor
ThreadPoolExecutor(1).submit(browser_shot).result()
print("Збережено figures/l9_browser_subgraph.png;  запит:", SUBGRAPH)
print(run(SUBGRAPH.replace("RETURN p", "RETURN k.name AS case, f.name AS fault, x.name AS related")).to_string(index=False))

# %% [Р5.1] Порівняння SPARQL та Cypher на 3 однакових питаннях
g_inf = Graph().parse("../8/pcdiag_inferred.ttl")
PFX = "PREFIX ex: <http://example.org/pcdiag#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
PAIRS = {
 "Q1 Підкласи MemoryFault (усі рівні)": (
   "SELECT ?c WHERE { ?c rdfs:subClassOf+ ex:MemoryFault } ORDER BY ?c",
   "MATCH (c:Class)-[:SUBCLASS_OF*1..]->(:Class {name:'MemoryFault'}) RETURN c.name ORDER BY c.name"),
 "Q2 Склад unit1 (транзитивно)": (
   "SELECT ?p WHERE { ex:unit1 ex:hasPart+ ?p } ORDER BY ?p",
   "MATCH (:Individual {name:'unit1'})-[:HAS_PART*1..]->(p) RETURN p.name ORDER BY p.name"),
 "Q3 Випадки з t > 90": (
   "SELECT ?k WHERE { ?k ex:cpuTemperature ?t FILTER(?t > 90) } ORDER BY ?k",
   "MATCH (k:DiagnosticCase) WHERE k.cpuTemperature > 90 RETURN k.name ORDER BY k.name"),
 "Q4 Визначений клас OverheatingCase": (
   "SELECT ?k WHERE { ?k a ex:OverheatingCase } ORDER BY ?k",
   "MATCH (k)-[:INSTANCE_OF]->(:Class {name:'OverheatingCase'}) RETURN k.name ORDER BY k.name"),
}
def timeit(f, n=30):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)
rows = []
for name, (sq, cq) in PAIRS.items():
    rs = [short(r[0]) for r in g.query(PFX + sq)]
    ri = [short(r[0]) for r in g_inf.query(PFX + sq)]
    rc = list(run(cq).iloc[:, 0])
    rows.append([name, len(rs), len(ri), len(rc), rs == rc,
                 round(timeit(lambda: list(g.query(PFX + sq))), 2), round(timeit(lambda: run(cq)), 2),
                 len(sq), len(cq)])
    print(f"--- {name}\n  SPARQL: {rs}\n  SPARQL+reasoner: {ri}\n  Cypher: {rc}")
res = pd.DataFrame(rows, columns=["питання", "SPARQL", "SPARQL+inf", "Cypher", "збіг", "SPARQL мс",
                                  "Cypher мс", "симв. SPARQL", "симв. Cypher"])
print(); print(res.to_string(index=False))
driver.close()
