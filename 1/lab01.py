# %% [Р1.1] Імпорт бібліотек, версії, seed
import sys, warnings
import numpy as np, pandas as pd, matplotlib, sklearn
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
RANDOM_STATE = 42
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 20)
print("Python", sys.version.split()[0], "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scikit-learn", sklearn.__version__,
      "| matplotlib", matplotlib.__version__)

# %% [Р1.2] Завантаження даних (UCI Wine Quality: червоні + білі вина)
red = pd.read_csv("../data/winequality-red.csv", sep=";")
white = pd.read_csv("../data/winequality-white.csv", sep=";")
red["type"], white["type"] = "red", "white"
df = pd.concat([red, white], ignore_index=True)
df.columns = [c.replace(" ", "_") for c in df.columns]
print("shape:", df.shape)
df.info()
print(df.describe(include="all").T.round(3))

# %% [Р1.3] Таблиця ознак
targets = {"quality_label": "ціль (класифікація)", "alcohol": "ціль (регресія)"}
feat = pd.DataFrame({
    "ознака": df.columns,
    "тип": [str(t) for t in df.dtypes],
    "шкала": ["номінальна" if c == "type" else ("порядкова" if c == "quality" else "відношень")
              for c in df.columns],
    "приклад": [df[c].iloc[0] for c in df.columns],
    "роль": [targets.get(c, "ознака") for c in df.columns]})
feat.loc[feat["ознака"] == "alcohol", "роль"] = "ціль (регресія)"
feat.loc[feat["ознака"] == "quality", "роль"] = "основа для цілі класифікації"
print(feat.to_string(index=False))

# %% [Р1.4] Перевірка якості: пропуски, дублікати, викиди IQR
print("Частка пропусків (макс.):", df.isna().mean().max())
print("Повних дублікатів:", df.duplicated().sum())
num_cols = df.select_dtypes("number").columns.drop("quality")
q1, q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
iqr = q3 - q1
out = ((df[num_cols] < q1 - 1.5 * iqr) | (df[num_cols] > q3 + 1.5 * iqr)).sum()
print("Викиди за IQR (к-сть):"); print(out.sort_values(ascending=False).to_string())
print("Нульових значень citric_acid:", (df.citric_acid == 0).sum())

# %% [Р1.5] Очищення: дублікати, ціль класифікації, one-hot
df = df.drop_duplicates().reset_index(drop=True)
df["quality_label"] = pd.cut(df["quality"], bins=[0, 5, 6, 10],
                             labels=["low", "medium", "high"])
df = pd.get_dummies(df, columns=["type"], drop_first=True, dtype=int)
print("Після видалення дублікатів:", df.shape)
print(df[["quality", "quality_label", "type_white"]].head())

# %% [Р1.6] Розбиття train/test ДО масштабування
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_cls = df.drop(columns=["quality", "quality_label"])
y_cls = df["quality_label"].astype(str)
Xtr, Xte, ytr, yte = train_test_split(X_cls, y_cls, test_size=0.3,
                                      stratify=y_cls, random_state=RANDOM_STATE)
scaler = StandardScaler().fit(Xtr)          # підганяємо лише на train
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
print("train:", Xtr.shape, "test:", Xte.shape)
print("Середнє train після масштабування:", Xtr_s.mean().round(3),
      "| середнє test:", Xte_s.mean().round(3))

# %% [Р2.1] Баланс класів
print((y_cls.value_counts(normalize=True) * 100).round(1).to_string())

# %% [Р2.2] Моделі: baseline, логістична регресія, дерево, випадковий ліс
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, classification_report, ConfusionMatrixDisplay)
models = {
    "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                           n_jobs=-1)}
rows = []
for name, m in models.items():
    m.fit(Xtr_s, ytr)
    p = m.predict(Xte_s); proba = m.predict_proba(Xte_s)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="macro", zero_division=0)
    rows.append([name, accuracy_score(yte, p), pr, rc, f1,
                 roc_auc_score(yte, proba, multi_class="ovr", average="macro")])
res_cls = pd.DataFrame(rows, columns=["модель", "accuracy", "precision_macro",
                                      "recall_macro", "F1_macro", "ROC-AUC_ovr"]).round(3)
print(res_cls.to_string(index=False))

# %% [Р2.3] Звіт і матриця помилок найкращої моделі
best = models["RandomForest"]
print(classification_report(yte, best.predict(Xte_s), digits=3))
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_estimator(best, Xte_s, yte, labels=["low", "medium", "high"],
                                      cmap="Blues", ax=ax)
ax.set_title("Матриця помилок — RandomForest")
fig.tight_layout(); fig.savefig(FIG / "l1_confusion.png", dpi=150)

# %% [Р3.1] Регресія: ціль alcohol, розподіл
X_reg = df.drop(columns=["alcohol", "quality", "quality_label"])
y_reg = df["alcohol"]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(y_reg, bins=40, color="#4C72B0", edgecolor="white")
ax.set_xlabel("alcohol, % об."); ax.set_ylabel("кількість вин")
ax.set_title("Розподіл цільової змінної alcohol")
fig.tight_layout(); fig.savefig(FIG / "l1_alcohol_hist.png", dpi=150)
print("Асиметрія alcohol:", round(y_reg.skew(), 3), "→ логарифм не потрібен")

# %% [Р3.2] Моделі регресії: baseline, лінійна, випадковий ліс
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_reg, y_reg, test_size=0.3,
                                              random_state=RANDOM_STATE)
sc_r = StandardScaler().fit(Xtr_r)
Xtr_rs, Xte_rs = sc_r.transform(Xtr_r), sc_r.transform(Xte_r)
regs = {"Dummy (mean)": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                                       random_state=RANDOM_STATE)}
rows, preds = [], {}
for name, m in regs.items():
    m.fit(Xtr_rs, ytr_r); p = m.predict(Xte_rs); preds[name] = p
    rows.append([name, mean_absolute_error(yte_r, p),
                 mean_squared_error(yte_r, p) ** 0.5, r2_score(yte_r, p)])
res_reg = pd.DataFrame(rows, columns=["модель", "MAE", "RMSE", "R2"]).round(3)
print(res_reg.to_string(index=False))

# %% [Р3.3] Аналіз залишків і «факт vs прогноз»
p = preds["RandomForestRegressor"]; resid = yte_r - p
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].scatter(p, resid, s=8, alpha=0.5); axs[0].axhline(0, color="red")
axs[0].set_xlabel("прогноз, % об."); axs[0].set_ylabel("залишок, % об.")
axs[0].set_title("Прогноз vs залишок (RF)")
axs[1].scatter(yte_r, p, s=8, alpha=0.5)
lim = [y_reg.min(), y_reg.max()]; axs[1].plot(lim, lim, "r--")
axs[1].set_xlabel("факт, % об."); axs[1].set_ylabel("прогноз, % об.")
axs[1].set_title("Факт vs прогноз (RF)")
fig.tight_layout(); fig.savefig(FIG / "l1_residuals.png", dpi=150)
print("Середній залишок:", round(resid.mean(), 4), "| кор(прогноз, залишок):",
      round(np.corrcoef(p, resid)[0, 1], 3))

# %% [Р4.1] Кластеризація: метод ліктя і силует для k=2..8
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
clu_cols = ["alcohol", "residual_sugar", "volatile_acidity", "total_sulfur_dioxide",
            "chlorides", "density", "pH", "sulphates"]
Z = StandardScaler().fit_transform(df[clu_cols])
ks, inertia, sil = range(2, 9), [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
print(pd.DataFrame({"k": list(ks), "inertia": np.round(inertia, 0),
                    "silhouette": np.round(sil, 3)}).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
axs[0].plot(ks, inertia, "o-"); axs[0].set_title("Метод ліктя")
axs[0].set_xlabel("k"); axs[0].set_ylabel("інерція")
axs[1].plot(ks, sil, "o-", color="green"); axs[1].set_title("Силуетний коефіцієнт")
axs[1].set_xlabel("k"); axs[1].set_ylabel("silhouette")
fig.tight_layout(); fig.savefig(FIG / "l1_elbow_silhouette.png", dpi=150)

# %% [Р4.2] Профіль кластерів (k=3), PCA, стабільність
K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Z)
df["cluster"] = km.labels_
prof = df.groupby("cluster")[clu_cols + ["type_white", "quality"]].mean().round(2)
prof["розмір"] = df.cluster.value_counts().sort_index()
print(prof.T.to_string())
pcs = PCA(n_components=2, random_state=42).fit_transform(Z)
fig, ax = plt.subplots(figsize=(6, 4.5))
sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=df.cluster, cmap="viridis", s=6, alpha=0.6)
ax.legend(*sc.legend_elements(), title="кластер")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Кластери k-means у просторі PCA")
fig.tight_layout(); fig.savefig(FIG / "l1_pca_clusters.png", dpi=150)
km2 = KMeans(n_clusters=K, n_init=10, random_state=7).fit(Z)
from sklearn.metrics import adjusted_rand_score
print("Розміри (rs=42):", sorted(np.bincount(km.labels_).tolist()),
      "| розміри (rs=7):", sorted(np.bincount(km2.labels_).tolist()),
      "| ARI =", round(adjusted_rand_score(km.labels_, km2.labels_), 3))

# %% [Р5] Підсумкова таблиця KDD
summary = pd.DataFrame([
    ["Класифікація (quality_label)", "RandomForest", "F1_macro",
     res_cls.F1_macro.iloc[0], res_cls.F1_macro.iloc[3]],
    ["Регресія (alcohol)", "RandomForestRegressor", "R2",
     res_reg.R2.iloc[0], res_reg.R2.iloc[2]],
    ["Кластеризація", f"KMeans k={K}", "silhouette", 0.0, round(sil[K - 2], 3)]],
    columns=["задача", "модель", "метрика", "baseline", "модель_знач"])
summary["приріст"] = (summary.модель_знач - summary.baseline).round(3)
print(summary.to_string(index=False))
