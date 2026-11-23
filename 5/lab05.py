# %% [Р1.1] Імпорт, дані Hotel Booking Demand, вибір цілі та метрики
import sys, time, warnings
import numpy as np, pandas as pd, matplotlib, sklearn
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
RANDOM_STATE = 42
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
pd.set_option("display.width", 150); pd.set_option("display.max_columns", 20)
raw = pd.read_csv("../data/hotels.csv", na_values=["NULL", "NA"])
print("Python", sys.version.split()[0], "| sklearn", sklearn.__version__, "| shape:", raw.shape)
# Витік: reservation_status(_date) містить результат (Canceled/Check-Out), assigned_room_type
# призначається при заселенні -> вилучаємо одразу
raw = raw.drop(columns=["reservation_status", "reservation_status_date", "assigned_room_type"])
print("Ціль is_canceled:", raw.is_canceled.value_counts(normalize=True).round(3).to_dict())
print("Метрика: ROC-AUC (основна) та F1 класу 'скасовано'")
print((raw.isna().mean() * 100).round(2)[lambda s: s > 0].to_string())

# %% [Р1.2] Механізм пропусків: порівняння розподілів у групах «є / немає пропуску»
def mech(col, cmp_num=("lead_time", "adr", "is_canceled"), cmp_cat="distribution_channel"):
    miss = raw[col].isna()
    t = raw.groupby(miss)[list(cmp_num)].mean().round(3)
    t.index = [f"{col}: є значення", f"{col}: ПРОПУСК"]
    top = raw.groupby(miss)[cmp_cat].agg(lambda s: s.value_counts(normalize=True).head(2).round(2).to_dict())
    t[f"топ {cmp_cat}"] = top.values
    return t
for c in ["agent", "company", "country"]:
    print(mech(c).to_string(), "\n")

# %% [Р1.3] Дублікати: точні й нечіткі
before = len(raw)
exact = raw.duplicated().sum()
df = raw.drop_duplicates().reset_index(drop=True)
key = ["hotel", "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month",
       "lead_time", "adults", "children", "country", "market_segment", "reserved_room_type",
       "is_canceled", "stays_in_week_nights", "stays_in_weekend_nights"]
df["adr_r"] = df.adr.round(-1)            # нечіткість: ціна з точністю до 10
fuzzy = df.duplicated(subset=key + ["adr_r"]).sum()
df = df.drop_duplicates(subset=key + ["adr_r"]).drop(columns="adr_r").reset_index(drop=True)
print(f"Було {before}; точних дублікатів {exact}; нечітких (ціна ±5, ті самі ключові поля) "
      f"{fuzzy}; залишилось {len(df)}")
# дата заїзду -> сортування за часом; підвибірка 20 000 для швидкості KNN/t-SNE
df["arrival"] = pd.to_datetime(df.arrival_date_year.astype(str) + "-" + df.arrival_date_month
                               + "-" + df.arrival_date_day_of_month.astype(str), format="%Y-%B-%d")
df = df.sample(20000, random_state=RANDOM_STATE).sort_values("arrival").reset_index(drop=True)
print("Підвибірка:", df.shape, "| період", df.arrival.min().date(), "—", df.arrival.max().date())

# %% [Р1.4] Дві стратегії імпутації в Pipeline: медіана+індикатор vs KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_validate
y = df["is_canceled"]
# приховані пропуски: adr = 0 при некомплементарному сегменті -> NaN
df.loc[(df.adr <= 0) & (df.market_segment != "Complementary"), "adr"] = np.nan
df["has_agent"] = df.agent.notna().astype(int); df["has_company"] = df.company.notna().astype(int)
NUM = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
       "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
       "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces",
       "total_of_special_requests", "has_agent", "has_company"]
CAT = ["hotel", "meal", "country", "market_segment", "distribution_channel",
       "reserved_room_type", "deposit_type", "customer_type"]
print("Пропуски у NUM:", df[NUM].isna().sum()[lambda s: s > 0].to_dict())
CV = TimeSeriesSplit(n_splits=5)       # розбиття за часом (обґрунтування — Результат 3)
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))])
def model(num_imputer):
    pre = ColumnTransformer([("num", Pipeline([("imp", num_imputer), ("sc", StandardScaler())]), NUM),
                             ("cat", cat_pipe, CAT)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
imp_res = {}
for name, imp in [("медіана + індикатор", SimpleImputer(strategy="median", add_indicator=True)),
                  ("KNNImputer (k=5)", KNNImputer(n_neighbors=5, add_indicator=True))]:
    t0 = time.time(); s = cross_val_score(model(imp), df[NUM + CAT], y, cv=CV, scoring="roc_auc")
    imp_res[name] = s
    print(f"{name:22s} ROC-AUC = {s.mean():.4f} ± {s.std():.4f}  ({time.time()-t0:.1f} с)")

# %% [Р2.1] Викиди трьома способами: IQR, z-оцінка, IsolationForest
from sklearn.ensemble import IsolationForest
OUT_COLS = ["lead_time", "adr", "stays_in_week_nights", "stays_in_weekend_nights"]
X_o = df[OUT_COLS].fillna(df[OUT_COLS].median())
q1, q3 = X_o.quantile(.25), X_o.quantile(.75); iqr = q3 - q1
m_iqr = ((X_o < q1 - 1.5 * iqr) | (X_o > q3 + 1.5 * iqr)).any(axis=1)
m_z = (((X_o - X_o.mean()) / X_o.std()).abs() > 3).any(axis=1)
m_if = pd.Series(IsolationForest(contamination=0.02, random_state=RANDOM_STATE)
                 .fit_predict(X_o) == -1, index=X_o.index)
ov = pd.DataFrame({"метод": ["IQR", "z > 3", "IsolationForest", "IQR ∩ z", "IQR ∩ IF", "z ∩ IF",
                             "усі три"],
                   "записів": [m_iqr.sum(), m_z.sum(), m_if.sum(), (m_iqr & m_z).sum(),
                               (m_iqr & m_if).sum(), (m_z & m_if).sum(), (m_iqr & m_z & m_if).sum()]})
print(ov.to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].bar(ov.метод[:3], ov.записів[:3], color=["#4C72B0", "#55A868", "#C44E52"])
axs[0].set_ylabel("к-сть записів"); axs[0].set_title("IQR позначає найбільше, IF — найменше")
for m_, c, lbl in [(m_iqr & ~m_z, "#4C72B0", "лише IQR"), (m_z & ~m_if, "#55A868", "z (не IF)"),
                   (m_if, "#C44E52", "IsolationForest")]:
    axs[1].scatter(X_o.lead_time[m_], X_o.adr[m_], s=8, c=c, label=lbl)
axs[1].scatter(X_o.lead_time[~(m_iqr | m_z | m_if)], X_o.adr[~(m_iqr | m_z | m_if)], s=2,
               c="#BBBBBB", label="норма", zorder=0)
axs[1].set_xlabel("lead_time, дні"); axs[1].set_ylabel("adr, EUR/ніч"); axs[1].legend()
axs[1].set_title("Викиди у просторі lead_time–adr")
fig.tight_layout(); fig.savefig(FIG / "l5_outliers.png", dpi=150)

# %% [Р2.2] Аналіз 10 прикладів викидів
ex = df.loc[m_iqr & m_z & m_if, ["hotel", "lead_time", "adr", "stays_in_week_nights",
                                  "stays_in_weekend_nights", "adults", "market_segment",
                                  "customer_type", "is_canceled"]]
print(pd.concat([ex.nlargest(5, "adr"), ex.nlargest(5, "lead_time")]).to_string())

# %% [Р2.3] Стратегія: winsorizing (межі з train) усередині Pipeline + вимірювання ефекту
from sklearn.base import BaseEstimator, TransformerMixin
class Winsorizer(BaseEstimator, TransformerMixin):
    """Обрізає значення за перцентилями, обчисленими ЛИШЕ на train (fit)."""
    def __init__(self, low=0.01, high=0.99): self.low, self.high = low, high
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_ = np.nanquantile(X, self.low, axis=0); self.hi_ = np.nanquantile(X, self.high, axis=0)
        return self
    def transform(self, X): return np.clip(np.asarray(X, float), self.lo_, self.hi_)
def model_w(winsor=True):
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if winsor: steps.append(("win", Winsorizer()))
    steps.append(("sc", StandardScaler()))
    pre = ColumnTransformer([("num", Pipeline(steps), NUM), ("cat", cat_pipe, CAT)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
for w in [False, True]:
    s = cross_val_score(model_w(w), df[NUM + CAT], y, cv=CV, scoring="roc_auc")
    print(f"winsorizing={w!s:5s}: ROC-AUC = {s.mean():.4f} ± {s.std():.4f}")

# %% [Р3.1] Нові ознаки та повний Pipeline без витоку
from sklearn.decomposition import PCA
def add_features(d):
    d = d.copy()
    month = d.arrival.dt.month
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    d["arrival_weekday"] = d.arrival.dt.weekday
    d["total_nights"] = d.stays_in_week_nights + d.stays_in_weekend_nights
    d["total_guests"] = d.adults + d.children.fillna(0) + d.babies
    d["adr_per_guest"] = d.adr / d.total_guests.replace(0, np.nan)
    d["weekend_ratio"] = d.stays_in_weekend_nights / d.total_nights.replace(0, np.nan)
    d["prev_cancel_rate"] = d.previous_cancellations / (d.previous_cancellations
                                                        + d.previous_bookings_not_canceled + 1)
    d["log_lead_time"] = np.log1p(d.lead_time)
    return d
NEW = ["month_sin", "month_cos", "arrival_weekday", "total_nights", "total_guests",
       "adr_per_guest", "weekend_ratio", "prev_cancel_rate", "log_lead_time"]
feat_tf = FunctionTransformer(add_features)       # без параметрів, що навчаються -> без витоку
def full_pipeline(impute="median", winsor=True, features=True, scale=True, pca=None, encode=True):
    num = NUM + (NEW if features else [])
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True) if impute == "median"
              else KNNImputer(add_indicator=True))]
    if winsor: steps.append(("win", Winsorizer()))
    if scale: steps.append(("sc", StandardScaler()))
    tr = [("num", Pipeline(steps), num)]
    if encode: tr.append(("cat", cat_pipe, CAT))
    st = [("feat", feat_tf), ("pre", ColumnTransformer(tr, sparse_threshold=0))]
    if pca: st.append(("pca", PCA(pca, random_state=RANDOM_STATE)))
    st.append(("clf", LogisticRegression(max_iter=5000)))
    return Pipeline(st)
X = df[NUM + CAT + ["arrival"]]
pipe = full_pipeline()
print(pipe)

# %% [Р3.2] Схема розбиття за часом і крос-валідація повного pipeline
for i, (tr_i, te_i) in enumerate(CV.split(X)):
    print(f"fold {i+1}: train {df.arrival[tr_i].min().date()}…{df.arrival[tr_i].max().date()} "
          f"({len(tr_i)}) -> test {df.arrival[te_i].min().date()}…{df.arrival[te_i].max().date()}")
cvr = cross_validate(pipe, X, y, cv=CV, scoring=["roc_auc", "f1"])
print(f"\nПовний pipeline: ROC-AUC = {cvr['test_roc_auc'].mean():.4f} ± {cvr['test_roc_auc'].std():.4f}"
      f" | F1 = {cvr['test_f1'].mean():.4f} ± {cvr['test_f1'].std():.4f}")

# %% [Р4.1] PCA: пояснена дисперсія, вибір k, навантаження
from sklearn.decomposition import PCA
Xn = Pipeline([("feat", feat_tf)]).fit_transform(X)[NUM + NEW]
Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(Xn))
pca = PCA().fit(Xs); cum = np.cumsum(pca.explained_variance_ratio_)
k95 = int(np.argmax(cum >= 0.95) + 1)
print(f"Числових ознак: {Xs.shape[1]}; k для 95 % дисперсії = {k95}")
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, label="частка компоненти")
ax.plot(range(1, len(cum) + 1), cum, "ro-", label="накопичена"); ax.axhline(.95, ls="--", c="k")
ax.axvline(k95, ls=":", c="g", label=f"k = {k95}"); ax.set_xlabel("номер компоненти")
ax.set_ylabel("частка дисперсії"); ax.legend(); ax.set_title("Scree-графік: дисперсія розподілена рівномірно")
fig.tight_layout(); fig.savefig(FIG / "l5_scree.png", dpi=150)
load = pd.DataFrame(pca.components_[:2].T, index=NUM + NEW, columns=["PC1", "PC2"])
for pc in ["PC1", "PC2"]:
    print(pc, "топ-4 навантаження:", load[pc].abs().nlargest(4).index.tolist(),
          load.loc[load[pc].abs().nlargest(4).index, pc].round(3).tolist())

# %% [Р4.2] t-SNE (perplexity 5/30/50) та UMAP (n_neighbors 15/50)
from sklearn.manifold import TSNE
import umap
idx = np.random.default_rng(RANDOM_STATE).choice(len(Xs), 3000, replace=False)
fig, axs = plt.subplots(1, 5, figsize=(22, 4.3))
for ax, p in zip(axs[:3], [5, 30, 50]):
    emb = TSNE(perplexity=p, random_state=RANDOM_STATE, init="pca").fit_transform(Xs[idx])
    ax.scatter(emb[:, 0], emb[:, 1], c=y.values[idx], cmap="coolwarm", s=3)
    ax.set_title(f"t-SNE, perplexity={p}"); ax.set_xticks([]); ax.set_yticks([])
for ax, nn in zip(axs[3:], [15, 50]):
    emb = umap.UMAP(n_neighbors=nn, random_state=RANDOM_STATE).fit_transform(Xs[idx])
    ax.scatter(emb[:, 0], emb[:, 1], c=y.values[idx], cmap="coolwarm", s=3)
    ax.set_title(f"UMAP, n_neighbors={nn}"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Нелінійні проєкції (червоний — скасовано, синій — ні): кластери змішані")
fig.tight_layout(); fig.savefig(FIG / "l5_tsne_umap.png", dpi=130)
print("Побудовано 5 проєкцій на 3000 об'єктах")

# %% [Р4.3] Модель на повних ознаках vs після PCA: метрика й час
rows = []
for name, p in [("повні ознаки", None), ("PCA 95 %", 0.95), ("PCA 10 компонент", 10)]:
    t0 = time.time()
    r = cross_validate(full_pipeline(pca=p), X, y, cv=CV, scoring="roc_auc")
    rows.append([name, r["test_score"].mean(), r["test_score"].std(), r["fit_time"].mean(),
                 time.time() - t0])
print(pd.DataFrame(rows, columns=["конфігурація", "ROC-AUC", "σ", "час fit, с", "час CV, с"])
      .round(4).to_string(index=False))

# %% [Р5.1] Абляція: внесок кожного кроку (однакові фолди TimeSeriesSplit)
def raw_pipeline():  # «сирі дані»: лише числові ознаки, обов'язкова мінімальна імпутація
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=5000))])
configs = {
    "0. сирі дані (числові, без обробки)": raw_pipeline(),
    "1. + масштабування": full_pipeline(winsor=False, features=False, encode=False),
    "2. + one-hot категорій": full_pipeline(winsor=False, features=False),
    "3. + winsorizing викидів": full_pipeline(features=False),
    "4. + нові ознаки (усі кроки)": full_pipeline(),
    "5. + PCA 95 %": full_pipeline(pca=0.95),
    "усі кроки − масштабування": full_pipeline(scale=False),
    "усі кроки − one-hot": full_pipeline(encode=False),
    "усі кроки − winsorizing": full_pipeline(winsor=False),
    "усі кроки − нові ознаки": full_pipeline(features=False),
    "усі кроки, KNN-імпутація": full_pipeline(impute="knn"),
}
abl = []
for name, p in configs.items():
    r = cross_validate(p, X, y, cv=CV, scoring=["roc_auc", "f1"])
    abl.append([name, r["test_roc_auc"].mean(), r["test_roc_auc"].std(), r["test_f1"].mean(),
                r["test_f1"].std()])
abl = pd.DataFrame(abl, columns=["конфігурація", "ROC-AUC", "σ_AUC", "F1", "σ_F1"]).round(4)
print(abl.to_string(index=False))
seq = abl.iloc[:6]
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.errorbar(range(6), seq["ROC-AUC"], yerr=seq["σ_AUC"], fmt="o-", capsize=5, lw=2)
ax.set_xticks(range(6)); ax.set_xticklabels([s.split(". ")[1] for s in seq.конфігурація],
                                            rotation=15, ha="right")
for i, v in enumerate(seq["ROC-AUC"]): ax.annotate(f"{v:.3f}", (i, v), xytext=(6, -14),
                                                   textcoords="offset points")
ax.set_ylabel("ROC-AUC (середнє ± σ, 5 фолдів)")
ax.set_title("Абляція: найбільший приріст дають one-hot категорій і нові ознаки")
fig.tight_layout(); fig.savefig(FIG / "l5_ablation.png", dpi=150)
