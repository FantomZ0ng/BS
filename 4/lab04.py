# %% [Р1] Імпорт, завантаження, вимоги до якості (пороги як конфігурація)
import sys, re, warnings
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
RANDOM_STATE = 42
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
pd.set_option("display.width", 150); pd.set_option("display.max_columns", 20)
raw = pd.read_excel("../data/online_retail.xlsx", dtype={"InvoiceNo": str, "StockCode": str})
print("Python", sys.version.split()[0], "| pandas", pd.__version__, "| shape:", raw.shape)
# Вимоги: вимір -> (метрика, поріг); дані для RFM-сегментації клієнтів і звітів продажів
REQ = pd.DataFrame([
    ["Повнота", "CustomerID", "1 - n_null/n", 0.95],
    ["Повнота", "Description", "1 - n_null/n", 0.99],
    ["Унікальність", "запис (усі поля)", "1 - n_dup/n", 0.999],
    ["Валідність", "Quantity", "частка Quantity>0 серед продажів", 0.99],
    ["Валідність", "UnitPrice", "частка UnitPrice>0", 0.99],
    ["Валідність", "StockCode", "частка ^\\d{5}[A-Za-z]{0,2}$", 0.98],
    ["Узгодженість", "InvoiceNo~Quantity", "частка (C ⇔ Q<0)", 0.99],
    ["Актуальність", "InvoiceDate", "середнє max(0,1-age/365)", 0.50]],
    columns=["вимір", "поле", "метрика", "поріг"])
print(REQ.to_string(index=False))

# %% [Р2.1] Функція profile(df)
def profile(df):
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
        rows.append({"стовпець": c, "тип": str(s.dtype),
                     "пропуски,%": round(s.isna().mean() * 100, 2),
                     "унікальних": s.nunique(),
                     "унікальність,%": round(s.nunique() / len(s) * 100, 2),
                     "min": s.min() if num else "", "max": s.max() if num else "",
                     "top-5": ", ".join(f"{k}({v})" for k, v in
                                        s.value_counts().head(5).items())[:70]})
    return pd.DataFrame(rows)
prof = profile(raw)
print(prof.to_string(index=False))

# %% [Р2.2] Приховані пропуски (заглушки) -> NaN
df = raw.copy()
desc = df.Description.astype("string").str.strip()
# у довіднику всі назви товарів — ВЕЛИКИМИ літерами; рядки лише з малих літер ('check',
# 'damaged', 'wet?') чи зі знаків '?' — службові нотатки складу, тобто приховані пропуски
stub_desc = (desc.str.match(r"^\?+$") | (desc == desc.str.lower())).fillna(False).astype(bool)
stubs = {"UnitPrice == 0": (df.UnitPrice == 0).sum(),
         "Description-заглушки ('?', нотатки малими літерами)": stub_desc.sum(),
         "Country == 'Unspecified'": (df.Country == "Unspecified").sum(),
         "CustomerID порожній": df.CustomerID.isna().sum()}
print(pd.Series(stubs, name="к-сть").to_string())
print("Приклади заглушок Description:", df.loc[stub_desc, "Description"].value_counts().head(8).to_dict())
df.loc[df.UnitPrice == 0, "UnitPrice"] = np.nan
df.loc[stub_desc, "Description"] = np.nan
df.loc[df.Country == "Unspecified", "Country"] = np.nan
print("\nЧастка пропусків після заміни заглушок, %:")
print((df.isna().mean() * 100).round(2).to_string())

# %% [Р2.3] Карта пропусків
smp = df.sort_values("InvoiceDate").iloc[::200]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})
sns.heatmap(smp.isna().T, cbar=False, cmap=["#EEEEEE", "#C44E52"], ax=axs[0])
axs[0].set_xticks([]); axs[0].set_xlabel("записи (кожен 200-й, за часом)")
axs[0].set_title("Карта пропусків: CustomerID — основне проблемне поле")
(df.isna().mean() * 100).sort_values().plot.barh(ax=axs[1], color="#C44E52")
axs[1].set_xlabel("% пропусків"); axs[1].set_title("Частка пропусків за стовпцями")
fig.tight_layout(); fig.savefig(FIG / "l4_missing.png", dpi=150)
row_na = df.isna().sum(axis=1)
print("Записів з ≥1 пропуском:", (row_na > 0).sum(), f"({(row_na > 0).mean():.1%})",
      "| з ≥2:", (row_na >= 2).sum())

# %% [Р3.1] Метрики якості за вимірами
df["is_cancel"] = df.InvoiceNo.str.startswith("C")
REF_DATE = df.InvoiceDate.max(); VOLATILITY = 365
def metrics(d):
    sales = d[~d.is_cancel]
    age = (REF_DATE - d.InvoiceDate).dt.days
    return pd.Series({
        "Повнота CustomerID": d.CustomerID.notna().mean(),
        "Повнота Description": d.Description.notna().mean(),
        "Унікальність записів": 1 - d.duplicated(subset=raw.columns.tolist()).mean(),
        "Валідність Quantity": (sales.Quantity > 0).mean(),
        "Валідність UnitPrice": (d.UnitPrice > 0).mean(),
        "Валідність StockCode": d.StockCode.astype(str).str.match(r"^\d{5}[A-Za-z]{0,2}$").mean(),
        "Узгодженість C⇔Q<0": (d.is_cancel == (d.Quantity < 0)).mean(),
        "Актуальність (Піпіно)": np.maximum(0, 1 - age / VOLATILITY).mean()})
m_all = metrics(df)
tab = REQ.copy(); tab["значення"] = m_all.values.round(4)
tab["статус"] = np.where(tab.значення >= tab.поріг, "OK", "НЕ ОК")
print(tab.to_string(index=False))
print("\nНевалідні StockCode:", df.loc[~df.StockCode.astype(str).str.match(r"^\d{5}[A-Za-z]{0,2}$"),
                                     "StockCode"].value_counts().head(8).to_dict())

# %% [Р3.2] Метрики у зрізах: за країною та за місяцем
df["month"] = df.InvoiceDate.dt.to_period("M").astype(str)
top_c = df.Country.value_counts().head(8).index
def by_slice(d, col):   # метрики для кожного значення зрізу
    return pd.DataFrame({k: metrics(g) for k, g in d.groupby(col)}).T
by_country = by_slice(df[df.Country.isin(top_c)], "Country")
by_month = by_slice(df, "month")
show = ["Повнота CustomerID", "Унікальність записів", "Валідність UnitPrice", "Узгодженість C⇔Q<0"]
print(by_country[show].round(3).to_string())
print(); print(by_month[show].round(3).to_string())
fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
by_month["Повнота CustomerID"].plot(ax=axs[0], marker="o", label="за місяцями")
axs[0].axhline(m_all["Повнота CustomerID"], color="k", ls="--", label="середнє")
axs[0].axhline(0.95, color="r", ls=":", label="поріг 0.95")
axs[0].set_ylim(0, 1.02); axs[0].set_ylabel("повнота CustomerID"); axs[0].legend()
axs[0].set_title("Провал повноти у грудні 2010 – січні 2011 (≈ 62 %)")
axs[0].tick_params(axis="x", rotation=45)
by_country["Повнота CustomerID"].sort_values().plot.barh(ax=axs[1], color="#4C72B0")
axs[1].axvline(0.95, color="r", ls=":"); axs[1].set_xlim(0, 1.02)
axs[1].set_xlabel("повнота CustomerID"); axs[1].set_title("UK (73 %) — головне джерело пропусків CustomerID")
fig.tight_layout(); fig.savefig(FIG / "l4_slices.png", dpi=150)

# %% [Р4.1] Автоматизовані перевірки: 12 правил на трьох рівнях
CODE_RE = r"^\d{5}[A-Za-z]{0,2}$"
RULES = [  # (id, рівень, опис, критичність, функція -> маска порушень або bool)
    ("F1", "поле", "InvoiceNo: 6 цифр, можливо з префіксом C/A", "критичне",
     lambda d: ~d.InvoiceNo.str.match(r"^[CA]?\d{6}$")),
    ("F2", "поле", "StockCode відповідає формату товарного коду", "некритичне",
     lambda d: ~d.StockCode.astype(str).str.match(CODE_RE)),
    ("F3", "поле", "UnitPrice > 0 і < 10000", "критичне",
     lambda d: ~d.UnitPrice.between(0.001, 10000)),
    ("F4", "поле", "|Quantity| < 10000", "критичне", lambda d: d.Quantity.abs() >= 10000),
    ("F5", "поле", "CustomerID не порожній", "некритичне", lambda d: d.CustomerID.isna()),
    ("R1", "запис", "скасування (C) ⇔ Quantity < 0", "критичне",
     lambda d: d.InvoiceNo.str.startswith("C") != (d.Quantity < 0)),
    ("R2", "запис", "продаж з Quantity>0 має Description", "некритичне",
     lambda d: (d.Quantity > 0) & d.Description.isna()),
    ("R3", "запис", "InvoiceDate у межах періоду вивантаження", "критичне",
     lambda d: ~d.InvoiceDate.between("2010-12-01", "2011-12-10")),
    ("S1", "набір", "немає повних дублікатів записів", "критичне",
     lambda d: d.duplicated(subset=raw.columns.tolist())),
    ("S2", "набір", "одна накладна — один клієнт і одна країна", "критичне",
     lambda d: d.InvoiceNo.map(d.groupby("InvoiceNo").Country.nunique()) > 1),
    ("S3", "набір", "обсяг набору ≥ 500 000 записів", "критичне",
     lambda d: pd.Series([len(d) < 500_000])),
    ("S4", "набір", "частка UK у межах 85–95 % (розподіл)", "некритичне",
     lambda d: pd.Series([not 0.85 <= (d.Country == "United Kingdom").mean() <= 0.95])),
]
def run_checks(d):
    rep = []
    for rid, lvl, desc, crit, f in RULES:
        mask = f(d).fillna(False).astype(bool)
        n = int(mask.sum()); share = n / len(mask)
        ex = d.loc[mask.values, ["InvoiceNo", "StockCode", "Quantity", "UnitPrice"]].head(2) \
            .apply(lambda r: "|".join(map(str, r)), axis=1).tolist() if len(mask) == len(d) and n else []
        rep.append([rid, lvl, desc, crit, n, round(share, 4), "; ".join(ex)[:55]])
    return pd.DataFrame(rep, columns=["id", "рівень", "правило", "критичність",
                                      "порушень", "частка", "приклади"])
report = run_checks(df)
print(report.to_string(index=False))

# %% [Р4.2] Поведінка конвеєра: критичні правила зупиняють, некритичні — попереджають
class DataQualityError(Exception): pass
def gate(rep, tolerance=0.0):
    crit = rep[(rep.критичність == "критичне") & (rep.частка > tolerance)]
    warn = rep[(rep.критичність == "некритичне") & (rep.порушень > 0)]
    for _, w in warn.iterrows():
        print(f"WARNING [{w.id}] {w.правило}: {w.порушень} порушень")
    if len(crit):
        raise DataQualityError("STOP: порушено критичні правила " + ", ".join(crit.id))
    print("PASS: конвеєр продовжує роботу")
try:
    gate(report)
except DataQualityError as e:
    print(e)
# після очищення: дублікати, ціни <= 0, екстремальні кількості, неузгоджені скасування
clean = df.drop_duplicates(subset=raw.columns.tolist())
clean = clean[clean.UnitPrice.between(0.001, 10000) & (clean.Quantity.abs() < 10000)]
clean = clean[clean.InvoiceNo.str.startswith("C") == (clean.Quantity < 0)]
print(f"\nПісля очищення: {len(clean)} записів ({len(clean)/len(df):.1%})")
try:
    gate(run_checks(clean))
except DataQualityError as e:
    print(e)

# %% [Р5] Матриця якості: вимір — метрика — поріг — факт — статус
m_clean = metrics(clean)
matrix = REQ.copy()
matrix["факт (сирі)"] = m_all.values.round(4); matrix["факт (очищені)"] = m_clean.values.round(4)
matrix["статус"] = np.where(matrix["факт (очищені)"] >= matrix.поріг, "OK", "НЕ ОК")
print(matrix.to_string(index=False))
print("\nУнікальних клієнтів:", clean.CustomerID.nunique(),
      "| частка виручки без CustomerID:",
      round((clean.Quantity * clean.UnitPrice)[clean.CustomerID.isna()].sum()
            / (clean.Quantity * clean.UnitPrice).sum(), 3))
