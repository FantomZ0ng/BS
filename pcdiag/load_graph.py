"""Ідемпотентний ETL онтології (8/pcdiag.ttl) у Neo4j — той самий алгоритм, що в ЛР 9 (MERGE за iri)."""
import re, sys
from pathlib import Path
from rdflib import Graph, RDF, RDFS, OWL, URIRef

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcdiag.onto import neo4j_driver, ROOT  # noqa: E402

EX = "http://example.org/pcdiag#"
ROOTS = ["Component", "Fault", "Symptom", "Action", "DiagnosticCase"]


def main():
    g = Graph().parse(ROOT / "8" / "pcdiag.ttl")
    short = lambda u: str(u).replace(EX, "")
    snake = lambda s: re.sub(r"(?<!^)(?=[A-Z])", "_", s).upper()
    classes = {c for c in g.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef) and str(c).startswith(EX)}
    inds = set(g.subjects(RDF.type, OWL.NamedIndividual))
    dprops = set(g.subjects(RDF.type, OWL.DatatypeProperty)); oprops = set(g.subjects(RDF.type, OWL.ObjectProperty))

    def conv(v):
        x = v.toPython()
        return x.isoformat() if hasattr(x, "isoformat") else float(x) if type(x).__name__ == "Decimal" else x

    def root(c):
        for a in g.transitive_objects(c, RDFS.subClassOf):
            if short(a) in ROOTS: return short(a)
        return "Component"

    d = neo4j_driver()
    with d.session() as s:
        for q in ["CREATE CONSTRAINT class_iri IF NOT EXISTS FOR (c:Class) REQUIRE c.iri IS UNIQUE",
                  "CREATE CONSTRAINT ind_iri IF NOT EXISTS FOR (i:Individual) REQUIRE i.iri IS UNIQUE",
                  "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)",
                  "CREATE INDEX ind_name IF NOT EXISTS FOR (i:Individual) ON (i.name)"]:
            s.run(q)
        s.run("UNWIND $r AS r MERGE (c:Class {iri:r.iri}) SET c.name=r.name, c.label=r.label",
              r=[{"iri": str(c), "name": short(c), "label": str(g.value(c, RDFS.label) or short(c))} for c in classes])
        s.run("UNWIND $r AS r MATCH (a:Class {iri:r.s}),(b:Class {iri:r.o}) MERGE (a)-[:SUBCLASS_OF]->(b)",
              r=[{"s": str(a), "o": str(b)} for a, b in g.subject_objects(RDFS.subClassOf) if a in classes and b in classes])
        for i in inds:
            t = [c for c in g.objects(i, RDF.type) if c in classes][0]
            s.run(f"MERGE (n:Individual {{iri:$iri}}) SET n:{root(t)}, n.name=$name, n += $p "
                  "WITH n MATCH (c:Class {iri:$c}) MERGE (n)-[:INSTANCE_OF]->(c)",
                  iri=str(i), name=short(i), c=str(t), p={short(p): conv(v) for p, v in g.predicate_objects(i) if p in dprops})
        for p in oprops:
            s.run(f"UNWIND $r AS r MATCH (a:Individual {{iri:r.s}}),(b:Individual {{iri:r.o}}) MERGE (a)-[:{snake(short(p))}]->(b)",
                  r=[{"s": str(a), "o": str(b)} for a, b in g.subject_objects(p) if a in inds and b in inds])
        n = s.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    print(f"Граф завантажено: {n} вузлів")


if __name__ == "__main__":
    main()
