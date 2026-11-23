"""Доступ до онтології ЛР 8 / графа ЛР 9: мітки, словник синонімів, фрагменти з Neo4j."""
import re
from pathlib import Path
from rdflib import Graph, RDFS, URIRef

ROOT = Path(__file__).resolve().parent.parent
EX = "http://example.org/pcdiag#"
_g = Graph().parse(ROOT / "8" / "pcdiag.ttl")


def _desc(root):
    r = URIRef(EX + root)
    return {str(c).replace(EX, ""): str(_g.value(c, RDFS.label)) for c in _g.transitive_subjects(RDFS.subClassOf, r)
            if str(c).startswith(EX) and c != r}


FAULTS_ALL = _desc("Fault")
# листові класи несправностей — допустимі мітки діагнозу
FAULT_LABELS = {k: v for k, v in FAULTS_ALL.items()
                if not any(str(s).replace(EX, "") == k for s in _g.objects(None, RDFS.subClassOf))}
SYMPTOMS, COMPONENTS, ACTIONS = _desc("Symptom"), _desc("Component"), _desc("Action")
ALL_CLASSES = {**FAULTS_ALL, **SYMPTOMS, **COMPONENTS, **ACTIONS, "Fault": "Несправність"}
SPECIAL = {"Insufficient": "Недостатньо даних для діагнозу", "Refuse": "Запит поза межами / шкідливий"}

# Словник синонімів: основа слова (нижній регістр) → клас онтології.
# Джерела: rdfs:label онтології, глосарій ЛР 7, розмовні вирази користувачів.
SYNONYMS = {
    "не вмикається": "NoPower", "не вмикаєтьс": "NoPower", "не реагує на кнопку": "NoPower", "немає живлення": "NoPower",
    "не запускається": "NoPower", "мертвий": "NoPower",
    "синій екран": "BlueScreen", "bsod": "BlueScreen", "синього екрана": "BlueScreen", "синім екраном": "BlueScreen",
    "memory_management": "BlueScreen", "критичн": "BlueScreen",
    "пищить": "POSTBeeps", "писк": "POSTBeeps", "сигнал": "POSTBeeps", "гудк": "POSTBeeps",
    "вимикається": "SuddenShutdown", "вимикаєтьс": "SuddenShutdown", "раптово гасне": "SuddenShutdown",
    "гальмує": "SlowPerformance", "повільно": "SlowPerformance", "тормозить": "SlowPerformance", "лагає": "SlowPerformance",
    "завантажувач": "BootloaderError", "bootmgr": "BootloaderError", "no bootable": "BootloaderError",
    "перезавантажується": "BootLoop", "циклічно": "BootLoop",
    "немає зображення": "NoDisplay", "чорний екран": "NoDisplay", "монітор не показує": "NoDisplay",
    "артефакт": "DisplayArtifacts", "смуги на екрані": "DisplayArtifacts", "квадрат": "DisplayArtifacts",
    "збивається час": "ClockReset", "скидається дата": "ClockReset", "скидаються налаштування bios": "ClockReset",
    "процесор": "CPU", "cpu": "CPU", "проц": "CPU",
    "оперативн": "RAM", "озп": "RAM", "пам'ят": "RAM", "ram": "RAM", "планк": "RAM", "xmp": "RAM",
    "жорстк": "HDD", "hdd": "HDD", "вінчестер": "HDD", "ssd": "SSD", "накопичувач": "StorageDrive", "диск": "StorageDrive",
    "блок живлення": "PowerSupplyUnit", "бж": "PowerSupplyUnit", "блока живлення": "PowerSupplyUnit",
    "відеокарт": "GraphicsCard", "gpu": "GraphicsCard", "відяха": "GraphicsCard",
    "вентилятор": "Fan", "кулер": "Fan", "шумить": "Fan", "радіатор": "Heatsink", "термопаст": "ThermalPaste",
    "батарейк": "CMOSBattery", "cmos": "CMOSBattery", "материнськ": "Motherboard", "монітор": "Monitor",
    "гарячий": "CPUOverheating", "перегрів": "CPUOverheating", "температур": "CPUOverheating",
    "вірус": "Malware", "реклам": "Malware", "майнер": "Malware", "драйвер": "DriverFault",
    "сектор": "BadSectors", "клацає": "BadSectors", "smart": "BadSectors",
}


def extract_concepts(text):
    """Пошук понять за словником синонімів (підрядковий збіг за основою слова)."""
    t = text.lower()
    found = []
    for syn, cls in sorted(SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"(?<![а-яіїєґa-z])" + re.escape(syn), t) and cls not in found:
            found.append(cls)
    return found


def neo4j_driver():
    from neo4j import GraphDatabase
    import os
    f = ROOT / ".env"
    env = dict(l.strip().split("=", 1) for l in f.read_text().splitlines() if "=" in l) if f.exists() else {}
    env = {**env, **{k: os.environ[k] for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD") if k in os.environ}}
    return GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]),
                                notifications_min_severity="OFF", connection_timeout=5)


FRAGMENT_CYPHER = """
UNWIND $concepts AS cname
MATCH (c:Class {name: cname})
// несправності, пов'язані з поняттям: сам клас-несправність, її підкласи,
// несправності з таким симптомом або такою ураженою складовою (через індивідів)
CALL (c) {
  MATCH (f:Class)-[:SUBCLASS_OF*0..]->(c) WHERE (f)-[:SUBCLASS_OF*1..]->(:Class {name:'Fault'})
  RETURN f
  UNION
  MATCH (fi:Fault)-[:MANIFESTS_AS|AFFECTS]->(x)-[:INSTANCE_OF]->(xc:Class)-[:SUBCLASS_OF*0..]->(c),
        (fi)-[:INSTANCE_OF]->(f:Class)
  RETURN f
  UNION
  MATCH (f:Class)-[:SUBCLASS_OF]->(p:Class)<-[:SUBCLASS_OF]-(sib:Class {name: c.name})
  WHERE (p)-[:SUBCLASS_OF*1..]->(:Class {name:'Fault'})
  RETURN f
}
WITH DISTINCT f
// додаємо «братів» — інші несправності тієї ж підсистеми (альтернативні гіпотези)
OPTIONAL MATCH (f)-[:SUBCLASS_OF]->(:Class)<-[:SUBCLASS_OF]-(f2:Class)
WITH collect(DISTINCT f) + collect(DISTINCT f2) AS fs
UNWIND fs AS f
WITH DISTINCT f
OPTIONAL MATCH (f)-[:SUBCLASS_OF]->(parent:Class)
OPTIONAL MATCH (fi:Fault)-[:INSTANCE_OF]->(f)
OPTIONAL MATCH (fi)-[:MANIFESTS_AS]->(s)-[:INSTANCE_OF]->(sc)
OPTIONAL MATCH (fi)-[:AFFECTS]->(comp)-[:INSTANCE_OF]->(cc)
OPTIONAL MATCH (fi)-[:CONFIRMED_BY]->(d)-[:INSTANCE_OF]->(dc)
OPTIONAL MATCH (fi)-[:RESOLVED_BY]->(a)-[:INSTANCE_OF]->(ac)
RETURN f.name AS fault, f.label AS label, parent.name AS parent,
       collect(DISTINCT sc.name) AS symptoms, collect(DISTINCT cc.name) AS components,
       collect(DISTINCT dc.name) AS confirm, collect(DISTINCT ac.name) AS repair
ORDER BY fault
"""


def graph_fragment(driver, concepts):
    recs, _, _ = driver.execute_query(FRAGMENT_CYPHER, concepts=concepts)
    rows = [dict(r) for r in recs if r["fault"] in FAULT_LABELS]
    return rows


def fragment_text(rows):
    lines = []
    for r in rows:
        rel = []
        if r["symptoms"]: rel.append("симптоми: " + ", ".join(r["symptoms"]))
        if r["components"]: rel.append("компонент: " + ", ".join(r["components"]))
        if r["confirm"]: rel.append("перевірка: " + ", ".join(r["confirm"]))
        if r["repair"]: rel.append("усунення: " + ", ".join(r["repair"]))
        lines.append(f"- {r['fault']} ({r['label']}; підклас {r['parent']})" + ("; " + "; ".join(rel) if rel else ""))
    return "\n".join(lines)
