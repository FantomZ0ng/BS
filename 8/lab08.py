# %% [Р1.1] Онтологія: IRI, класи з таксономії ЛР 7 (subClassOf)
import sys, json, datetime, warnings
import owlready2, rdflib, pandas as pd
from owlready2 import (get_ontology, Thing, ObjectProperty, DataProperty, FunctionalProperty,
                       TransitiveProperty, AllDisjoint, ConstrainedDatatype, Not, locstr,
                       sync_reasoner, OwlReadyInconsistentOntologyError, default_world)
warnings.filterwarnings("ignore")
pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 70)
print("Python", sys.version.split()[0], "| owlready2", owlready2.VERSION, "| rdflib", rdflib.__version__)
IRI = "http://example.org/pcdiag#"
onto = get_ontology(IRI)
TAX = json.load(open("../7/taxonomy.json", encoding="utf-8"))["is_a"]
TAX.update({"Діагностичний випадок": "Поняття", "Персональний комп'ютер": "Комутаційний компонент",
            "Системний блок": "Комутаційний компонент", "Система охолодження": "Компонент охолодження",
            "Термопаста": "Компонент охолодження", "Монітор": "Обчислювальний компонент"})
ID = {  # українська назва -> ідентифікатор класу в IRI
 "Поняття": "DiagnosticConcept", "Несправність": "Fault", "Симптом": "Symptom", "Компонент": "Component",
 "Дія": "Action", "Діагностичний випадок": "DiagnosticCase",
 "Апаратна несправність": "HardwareFault", "Програмна несправність": "SoftwareFault",
 "Несправність живлення": "PowerFault", "Несправність пам'яті": "MemoryFault",
 "Несправність накопичувача": "StorageFault", "Несправність охолодження": "CoolingFault",
 "Несправність відеопідсистеми": "VideoFault", "Відмова блока живлення": "PSUFailure",
 "Розряджена батарейка CMOS": "CMOSBatteryDepleted", "Дефект модуля ОЗП": "RAMModuleDefect",
 "Поганий контакт ОЗП": "RAMPoorContact", "Нестабільний розгін ОЗП": "RAMUnstableOverclock",
 "Збійні сектори": "BadSectors", "Перегрів процесора": "CPUOverheating", "Відмова вентилятора": "FanFailure",
 "Відмова відеокарти": "GPUFailure", "Несправність ОС": "OSFault", "Несправність драйвера": "DriverFault",
 "Шкідливе ПЗ": "Malware", "Пошкодження завантажувача": "BootloaderCorruption",
 "Пошкодження системних файлів": "SystemFileCorruption",
 "Симптом увімкнення": "PowerOnSymptom", "Симптом завантаження": "BootSymptom", "Симптом роботи": "RuntimeSymptom",
 "Немає живлення": "NoPower", "Сигнали POST": "POSTBeeps", "Немає зображення": "NoDisplay",
 "Синій екран": "BlueScreen", "Помилка завантажувача": "BootloaderError", "Циклічне перезавантаження": "BootLoop",
 "Раптове вимкнення": "SuddenShutdown", "Повільна робота": "SlowPerformance",
 "Артефакти зображення": "DisplayArtifacts", "Скидання дати й часу": "ClockReset",
 "Компонент живлення": "PowerComponent", "Обчислювальний компонент": "ComputeComponent",
 "Компонент зберігання": "StorageComponent", "Компонент охолодження": "CoolingComponent",
 "Комутаційний компонент": "InterconnectComponent", "Блок живлення": "PowerSupplyUnit",
 "Батарейка CMOS": "CMOSBattery", "Процесор": "CPU", "Відеокарта": "GraphicsCard",
 "Оперативна пам'ять": "RAM", "Накопичувач": "StorageDrive", "Жорсткий диск": "HDD",
 "Твердотільний накопичувач": "SSD", "Вентилятор": "Fan", "Радіатор": "Heatsink",
 "Системна плата": "Motherboard", "Персональний комп'ютер": "Computer", "Системний блок": "SystemUnit",
 "Система охолодження": "CoolingSystem", "Термопаста": "ThermalPaste", "Монітор": "Monitor",
 "Діагностична дія": "DiagnosticAction", "Ремонтна дія": "RepairAction", "Тест пам'яті": "MemoryTest",
 "Перевірка SMART": "SMARTCheck", "Вимірювання температури": "TemperatureMeasurement",
 "Перевірка іншим БЖ": "PSUSwapTest", "Заміна компонента": "ComponentReplacement",
 "Чищення від пилу": "DustCleaning", "Відновлення завантажувача": "BootloaderRepair",
 "Перевстановлення драйвера": "DriverReinstall", "Антивірусна перевірка": "AntivirusScan",
 "Вимкнення XMP": "DisableXMP",
}
missing = set(TAX) - set(ID)
assert not missing, missing
C = {}
def make(uk):
    if uk in C: return C[uk]
    parent = Thing if uk == "Поняття" else make(TAX[uk])
    with onto:
        cls = type(ID[uk], (parent,), {}); cls.label = [locstr(uk, "uk")]
    C[uk] = cls; return cls
for uk in TAX: make(uk)
K = {ID[k]: v for k, v in C.items()}           # доступ за англ. ідентифікатором
print("Класів:", len(list(onto.classes())), "| перенесено з ЛР 7:", len(TAX) - 6, "+ 6 з part-of/випадків")

def tree(cls, ind=0, out=None):   # вигляд ієрархії як у вкладці Classes Protégé
    out = [] if out is None else out
    out.append("    " * ind + ("▸ " if list(cls.subclasses()) else "• ") + f"{cls.name}  «{cls.label.first()}»")
    for s in sorted(cls.subclasses(), key=lambda c: c.name): tree(s, ind + 1, out)
    return out
print("\n".join(tree(K["DiagnosticConcept"])))

# %% [Р1.2] Object/Data properties: domain, range, інверсія, транзитивність, функціональність
with onto:
    class hasPart(ObjectProperty, TransitiveProperty):
        domain = [K["Component"]]; range = [K["Component"]]
    class partOf(ObjectProperty, TransitiveProperty):
        domain = [K["Component"]]; range = [K["Component"]]; inverse_property = hasPart
    class hasSymptom(ObjectProperty):  domain = [K["DiagnosticCase"]]; range = [K["Symptom"]]
    class diagnosedAs(ObjectProperty): domain = [K["DiagnosticCase"]]; range = [K["Fault"]]
    class concernsComputer(ObjectProperty): domain = [K["DiagnosticCase"]]; range = [K["Computer"]]
    class affects(ObjectProperty):     domain = [K["Fault"]]; range = [K["Component"]]
    class manifestsAs(ObjectProperty): domain = [K["Fault"]]; range = [K["Symptom"]]
    class resolvedBy(ObjectProperty):  domain = [K["Fault"]]; range = [K["RepairAction"]]
    class confirmedBy(ObjectProperty): domain = [K["Fault"]]; range = [K["DiagnosticAction"]]
    class cpuTemperature(DataProperty, FunctionalProperty): domain = [K["DiagnosticCase"]]; range = [float]
    class reportedOn(DataProperty, FunctionalProperty):     domain = [K["DiagnosticCase"]]; range = [datetime.datetime]  # xsd:date не входить у datatype map OWL 2
    class ageYears(DataProperty, FunctionalProperty):       domain = [K["Computer"]]; range = [int]
    class modelName(DataProperty):                          domain = [K["Component"]]; range = [str]
for p in list(onto.object_properties()) + list(onto.data_properties()):
    chars = [c.__name__ for c in p.is_a if c.__name__ in ("TransitiveProperty", "FunctionalProperty")]
    inv = p.inverse_property.name if getattr(p, "inverse_property", None) else ""
    rng = [r.name if hasattr(r, "name") else r.__name__ for r in p.range]
    print(f"{p.name:17s} domain={[d.name for d in p.domain]} range={rng} {chars} {('inverse=' + inv) if inv else ''}")

# %% [Р2.1] Аксіоми п'яти типів (запис у Manchester-синтаксисі нижче)
with onto:
    # 1) SubClassOf з обмеженнями some / only
    K["HardwareFault"].is_a.append(affects.some(K["Component"]))
    K["Fault"].is_a.append(resolvedBy.only(K["RepairAction"]))
    K["MemoryFault"].is_a.append(affects.some(K["RAM"]))
    # 2) EquivalentTo — визначені класи для автокласифікації
    class OverheatingCase(K["DiagnosticCase"]):
        equivalent_to = [K["DiagnosticCase"] & cpuTemperature.some(ConstrainedDatatype(float, min_exclusive=90.0))]
        label = [locstr("Випадок перегріву", "uk")]
    class MemoryRelatedCase(K["DiagnosticCase"]):
        equivalent_to = [K["DiagnosticCase"] & diagnosedAs.some(K["MemoryFault"])]
        label = [locstr("Випадок, пов'язаний з пам'яттю", "uk")]
    class StartupProblemCase(K["DiagnosticCase"]):
        equivalent_to = [K["DiagnosticCase"] & hasSymptom.some(K["PowerOnSymptom"] | K["BootSymptom"])]
        label = [locstr("Випадок проблеми запуску", "uk")]
    class UndiagnosedCase(K["DiagnosticCase"]):
        equivalent_to = [K["DiagnosticCase"] & Not(diagnosedAs.some(K["Fault"]))]
        label = [locstr("Недіагностований випадок", "uk")]
    # 3) DisjointWith
    AllDisjoint([K["HardwareFault"], K["SoftwareFault"]])
    AllDisjoint([K["PowerOnSymptom"], K["BootSymptom"], K["RuntimeSymptom"]])
    AllDisjoint([K["Fault"], K["Symptom"], K["Component"], K["Action"], K["DiagnosticCase"]])
    # 4) кардинальність (на простих, не транзитивних властивостях)
    K["DiagnosticCase"].is_a.append(diagnosedAs.max(1, K["Fault"]))
    K["DiagnosticCase"].is_a.append(concernsComputer.exactly(1, K["Computer"]))
    K["DiagnosticCase"].is_a.append(hasSymptom.min(1, K["Symptom"]))
    # 5) domain/range задано в Р1.2
AX = [
 "HardwareFault SubClassOf: affects some Component",
 "Fault SubClassOf: resolvedBy only RepairAction",
 "MemoryFault SubClassOf: affects some RAM",
 "OverheatingCase EquivalentTo: DiagnosticCase and (cpuTemperature some xsd:float[> 90.0])",
 "MemoryRelatedCase EquivalentTo: DiagnosticCase and (diagnosedAs some MemoryFault)",
 "StartupProblemCase EquivalentTo: DiagnosticCase and (hasSymptom some (PowerOnSymptom or BootSymptom))",
 "UndiagnosedCase EquivalentTo: DiagnosticCase and not (diagnosedAs some Fault)",
 "DisjointClasses: HardwareFault, SoftwareFault",
 "DisjointClasses: PowerOnSymptom, BootSymptom, RuntimeSymptom",
 "DisjointClasses: Fault, Symptom, Component, Action, DiagnosticCase",
 "DiagnosticCase SubClassOf: diagnosedAs max 1 Fault",
 "DiagnosticCase SubClassOf: concernsComputer exactly 1 Computer",
 "DiagnosticCase SubClassOf: hasSymptom min 1 Symptom",
 "ObjectProperty: hasPart  Domain: Component  Range: Component  Characteristics: Transitive",
 "ObjectProperty: partOf  InverseOf: hasPart  Characteristics: Transitive",
 "DataProperty: cpuTemperature  Domain: DiagnosticCase  Range: xsd:float  Characteristics: Functional",
]
print("\n".join(AX))

# %% [Р2.2] Індивіди (A-Box): компоненти, несправності, симптоми, дії, випадки
with onto:
    I = {}
    def ind(name, cls, **kw):
        o = K[cls](name)
        for k, v in kw.items(): setattr(o, k, v if isinstance(v, list) or k in
                                         ("cpuTemperature", "reportedOn", "ageYears") else [v])
        I[name] = o; return o
    # компоненти ПК лабораторії (ланцюг hasPart — лише прямі зв'язки)
    for n, c, m in [("cpu1", "CPU", "Intel Core i5-10400"), ("ram1", "RAM", "Kingston 8GB DDR4"),
                    ("ssd1", "SSD", "Samsung 870 EVO"), ("hdd1", "HDD", "WD Blue 1TB"),
                    ("psu1", "PowerSupplyUnit", "Chieftec 500W"), ("fan1", "Fan", "Deepcool GAMMAXX"),
                    ("gpu1", "GraphicsCard", "GTX 1650"), ("bat1", "CMOSBattery", "CR2032")]:
        ind(n, c, modelName=m)
    ind("cool1", "CoolingSystem", hasPart=[I["fan1"]])
    ind("mb1", "Motherboard", modelName="ASUS PRIME B460", hasPart=[I["cpu1"], I["ram1"], I["bat1"]])
    ind("unit1", "SystemUnit", hasPart=[I["mb1"], I["psu1"], I["ssd1"], I["hdd1"], I["cool1"], I["gpu1"]])
    pc1 = ind("pc1", "Computer", hasPart=[I["unit1"]], ageYears=5)
    # дії
    for n, c in [("a_replace", "ComponentReplacement"), ("a_clean", "DustCleaning"),
                 ("a_driver", "DriverReinstall"), ("a_boot", "BootloaderRepair"),
                 ("a_av", "AntivirusScan"), ("d_memtest", "MemoryTest"), ("d_smart", "SMARTCheck"),
                 ("d_temp", "TemperatureMeasurement"), ("d_psu", "PSUSwapTest")]:
        ind(n, c)
    # симптоми
    for n, c in [("s_nopower", "NoPower"), ("s_bsod", "BlueScreen"), ("s_booterr", "BootloaderError"),
                 ("s_shutdown", "SuddenShutdown"), ("s_slow", "SlowPerformance"), ("s_beeps", "POSTBeeps")]:
        ind(n, c)
    # несправності
    ind("f_psu", "PSUFailure", affects=I["psu1"], manifestsAs=[I["s_nopower"], I["s_shutdown"]],
        resolvedBy=I["a_replace"], confirmedBy=I["d_psu"])
    ind("f_overheat", "CPUOverheating", affects=I["cpu1"], manifestsAs=[I["s_shutdown"], I["s_slow"]],
        resolvedBy=I["a_clean"], confirmedBy=I["d_temp"])
    ind("f_sectors", "BadSectors", affects=I["hdd1"], manifestsAs=[I["s_booterr"], I["s_bsod"]],
        resolvedBy=I["a_replace"], confirmedBy=I["d_smart"])
    ind("f_ram", "RAMModuleDefect", affects=I["ram1"], manifestsAs=[I["s_bsod"], I["s_beeps"]],
        resolvedBy=I["a_replace"], confirmedBy=I["d_memtest"])
    ind("f_driver", "DriverFault", manifestsAs=[I["s_bsod"]], resolvedBy=I["a_driver"])
    ind("f_malware", "Malware", manifestsAs=[I["s_slow"]], resolvedBy=I["a_av"])
    # діагностичні випадки (заявки)
    D = lambda y, m, d: datetime.datetime(y, m, d, 9, 0)
    ind("case1", "DiagnosticCase", concernsComputer=pc1, hasSymptom=[I["s_nopower"]], diagnosedAs=I["f_psu"], reportedOn=D(2026, 9, 3))
    ind("case2", "DiagnosticCase", concernsComputer=pc1, hasSymptom=[I["s_bsod"]], diagnosedAs=I["f_driver"], cpuTemperature=54.0, reportedOn=D(2026, 9, 15))
    ind("case3", "DiagnosticCase", concernsComputer=pc1, hasSymptom=[I["s_shutdown"]], diagnosedAs=I["f_overheat"], cpuTemperature=96.5, reportedOn=D(2026, 10, 1))
    ind("case4", "DiagnosticCase", concernsComputer=pc1, hasSymptom=[I["s_booterr"]], diagnosedAs=I["f_sectors"], reportedOn=D(2026, 10, 12))
    ind("case5", "DiagnosticCase", concernsComputer=pc1, hasSymptom=[I["s_slow"]], cpuTemperature=92.0, reportedOn=D(2026, 11, 2))
    ind("case6", "DiagnosticCase", concernsComputer=pc1, hasSymptom=[I["s_bsod"], I["s_beeps"]], diagnosedAs=I["f_ram"], cpuTemperature=61.0, reportedOn=D(2026, 11, 9))
print("Індивідів:", len(list(onto.individuals())))
print(pd.DataFrame([(i.name, i.is_a[0].name, ", ".join(f"{p.python_name}={[getattr(v, 'name', v) for v in p[i]]}"
       for p in i.get_properties() if p.python_name not in ("label",)))
       for i in onto.individuals()], columns=["індивід", "клас", "властивості"]).to_string(index=False, max_colwidth=95))
onto.save("pcdiag.owl", format="rdfxml")
g = rdflib.Graph().parse("pcdiag.owl", format="xml"); g.bind("ex", IRI)
g.serialize("pcdiag.ttl", format="turtle")
print("Збережено pcdiag.owl і pcdiag.ttl; трійок:", len(g))

# %% [Р3.1] Reasoner HermiT: узгодженість і автоматична класифікація
asserted_types = {i.name: {c.name for c in i.is_a} for i in onto.individuals()}
with onto:
    sync_reasoner(infer_property_values=True, debug=0)       # HermiT
print("Онтологія узгоджена: так (HermiT не повідомив про суперечності)")
rows = []
for i in onto.individuals():
    new = {c.name for c in i.is_a} - asserted_types[i.name]
    if new: rows.append((i.name, ", ".join(sorted(asserted_types[i.name])), ", ".join(sorted(new))))
print(pd.DataFrame(rows, columns=["індивід", "задано", "виведено reasoner-ом"]).to_string(index=False))
print("\nInferred part-of (транзитивність + інверсія): cpu1.partOf =", sorted(p.name for p in I["cpu1"].partOf))
print("pc1.hasPart (з транзитивністю) =", sorted(p.name for p in I["pc1"].INDIRECT_hasPart))
print("UndiagnosedCase:", [i.name for i in onto.UndiagnosedCase.instances()],
      "← порожньо через припущення відкритого світу")
g_inf = rdflib.Graph().parse(data=default_world.as_rdflib_graph().serialize(format="turtle"), format="turtle")
g_inf.bind("ex", IRI); g_inf.serialize("pcdiag_inferred.ttl", format="turtle")
print("Трійок до/після виведення:", len(g), "/", len(g_inf))

# %% [Р3.2] Навмисна суперечність: індивід одночасно апаратна й програмна несправність
with onto:
    bad = K["PSUFailure"]("f_conflict"); bad.is_a.append(K["DriverFault"])
try:
    with onto: sync_reasoner(debug=0)
    print("узгоджено")
except OwlReadyInconsistentOntologyError as e:
    print("HermiT: OwlReadyInconsistentOntologyError — онтологія НЕузгоджена")
# пояснення (аналог Explain inconsistency): шукаємо диз'юнктні пари серед предків типів індивіда
def explain(ind):
    anc = set().union(*[set(c.ancestors()) for c in ind.is_a if hasattr(c, "ancestors")])
    for dj in onto.disjoint_classes():
        hit = [c for c in dj.entities if c in anc]
        if len(hit) > 1:
            print(f"  {ind.name}: типи {[c.name for c in ind.is_a]} → предки {[c.name for c in hit]}"
                  f" оголошено DisjointClasses({', '.join(c.name for c in dj.entities)})")
explain(bad)
# виправлення: індивід «f_conflict» описував одну подію — БЖ вийшов з ладу; тип DriverFault помилковий
bad.is_a.remove(K["DriverFault"])
with onto: sync_reasoner(debug=0)
print("Після вилучення типу DriverFault з f_conflict онтологія знову узгоджена")
owlready2.destroy_entity(bad)

# %% [Р3.3] Припущення відкритого світу: приклад і закриття
print("case5 типи:", [c.name for c in I["case5"].is_a])
print("case5 ∈ UndiagnosedCase?", I["case5"] in onto.UndiagnosedCase.instances())
with onto:   # закриваємо світ для case5: явно стверджуємо, що діагнозу немає
    I["case5"].is_a.append(diagnosedAs.max(0, K["Fault"]))
    sync_reasoner(debug=0)
print("Після аксіоми «diagnosedAs max 0 Fault»: case5 ∈ UndiagnosedCase?",
      I["case5"] in onto.UndiagnosedCase.instances())
onto.save("pcdiag.owl", format="rdfxml")
g = rdflib.Graph().parse("pcdiag.owl", format="xml"); g.bind("ex", IRI)
# асертований граф: прибираємо виведені reasoner-ом типи, щоб порівняти «без/з reasoner»
g_asserted = rdflib.Graph().parse("pcdiag.ttl", format="turtle")
g_inf = rdflib.Graph().parse(data=default_world.as_rdflib_graph().serialize(format="turtle"), format="turtle")
g_inf.serialize("pcdiag_inferred.ttl", format="turtle")
print("Збережено фінальні pcdiag.owl / pcdiag_inferred.ttl")

# %% [Р4.1] SPARQL-запити за компетентнісними питаннями (rdflib)
PFX = f"""PREFIX ex: <{IRI}>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""
Q = {
 "Q1 (CQ2) підкласи пам'яті, шлях +": """
SELECT ?cls ?label WHERE { ?cls rdfs:subClassOf+ ex:MemoryFault ; rdfs:label ?label } ORDER BY ?cls""",
 "Q2 (CQ4) склад системного блока, шлях +": """
SELECT ?part WHERE { ex:unit1 ex:hasPart+ ?part } ORDER BY ?part""",
 "Q3 (CQ8) FILTER за температурою": """
SELECT ?case ?t WHERE { ?case ex:cpuTemperature ?t FILTER(?t > 90) } ORDER BY DESC(?t)""",
 "Q4 (CQ10) OPTIONAL + !bound": """
SELECT ?case ?date WHERE { ?case a ex:DiagnosticCase ; ex:reportedOn ?date .
  OPTIONAL { ?case ex:diagnosedAs ?f } FILTER(!BOUND(?f)) }""",
 "Q5 (CQ6) COUNT/GROUP BY": """
SELECT ?comp (COUNT(?f) AS ?n) WHERE { ?f ex:affects ?comp } GROUP BY ?comp ORDER BY DESC(?n) ?comp""",
 "Q6 (CQ5) апаратні vs програмні": """
SELECT ?kind (COUNT(DISTINCT ?f) AS ?n) WHERE { ?f rdf:type/rdfs:subClassOf* ?kind
  VALUES ?kind { ex:HardwareFault ex:SoftwareFault } } GROUP BY ?kind""",
 "Q7 (CQ3) дія при перегріві": """
SELECT ?f ?action ?type WHERE { ?f a ex:CPUOverheating ; ex:resolvedBy ?action . ?action a ?type
  FILTER(?type != <http://www.w3.org/2002/07/owl#NamedIndividual>) }""",
 "Q8 (CQ1) симптоми відмови БЖ": """
SELECT ?s ?type WHERE { ?f a ex:PSUFailure ; ex:manifestsAs ?s . ?s a ?type
  FILTER(STRSTARTS(STR(?type), STR(ex:))) }""",
 "Q9 (CQ9) підтвердження збійних секторів": """
SELECT ?d WHERE { ?f a ex:BadSectors ; ex:confirmedBy ?d }""",
 "Q10 (CQ7) несправність за BSOD + тест пам'яті": """
SELECT ?f (COUNT(?s) AS ?k) WHERE { ?f ex:manifestsAs ?s .
  { ?s a ex:BlueScreen } UNION { ?f ex:confirmedBy ?d . ?d a ex:MemoryTest } } GROUP BY ?f HAVING (COUNT(?s) >= 2)""",
 "Q11 визначений клас OverheatingCase": """
SELECT ?case WHERE { ?case a ex:OverheatingCase }""",
 "Q12 partOf (лише через інверсію)": """
SELECT ?whole WHERE { ex:cpu1 ex:partOf ?whole }""",
}
short = lambda v: str(v).replace(IRI, "ex:").replace("http://www.w3.org/2001/XMLSchema#", "xsd:")
RES = {}
for name, q in Q.items():
    ra = [tuple(short(x) for x in r) for r in g_asserted.query(PFX + q)]
    ri = [tuple(short(x) for x in r) for r in g_inf.query(PFX + q)]
    RES[name] = (ra, ri)
    print(f"--- {name}\n  без reasoner ({len(ra)}): {ra}\n  з reasoner  ({len(ri)}): {ri}")

# %% [Р5.1] Покриття компетентнісних питань
cq = pd.read_csv("../7/glossary.csv").shape  # перевірка доступу до матеріалів ЛР 7
COV = [("CQ1", "Q8", 1.0), ("CQ2", "Q1", 1.0), ("CQ3", "Q7", 1.0), ("CQ4", "Q2", 1.0), ("CQ5", "Q6", 1.0),
       ("CQ6", "Q5", 1.0), ("CQ7", "Q10", 0.5), ("CQ8", "Q3", 1.0), ("CQ9", "Q9", 1.0), ("CQ10", "Q4", 0.5)]
cov = pd.DataFrame(COV, columns=["CQ", "запит", "покриття"])
cov["відповідь"] = [len(RES[[k for k in RES if k.startswith(q + " ")][0]][1]) for q in cov.запит]
print(cov.to_string(index=False))
print(f"Повністю покрито: {(cov.покриття == 1).sum()}/10; зважене покриття = {cov.покриття.mean():.0%}")
print("Класів:", len(list(onto.classes())), "| об'єктних властивостей:", len(list(onto.object_properties())),
      "| властивостей даних:", len(list(onto.data_properties())), "| індивідів:", len(list(onto.individuals())))
