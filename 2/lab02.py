# %% [Р1.1] Імпорт, версії, завантаження UCI Adult
import sys, warnings
import numpy as np, pandas as pd, matplotlib, sklearn, fairlearn
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
RANDOM_STATE = 42
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
pd.set_option("display.width", 140)
cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"]
tr = pd.read_csv("../data/adult.data", names=cols, skipinitialspace=True, na_values="?")
te = pd.read_csv("../data/adult.test", names=cols, skipinitialspace=True, na_values="?",
                 skiprows=1)
df = pd.concat([tr, te], ignore_index=True)
df["income"] = df["income"].str.rstrip(".")
df["y"] = (df["income"] == ">50K").astype(int)
print("Python", sys.version.split()[0], "| sklearn", sklearn.__version__,
      "| fairlearn", fairlearn.__version__)
print("shape:", df.shape)
print(df.head(3).T)

# %% [Р1.2] Розподіли чутливих атрибутів і базові частоти P(y=1 | група)
for attr in ["sex", "race"]:
    t = df.groupby(attr)["y"].agg(n="size", base_rate="mean")
    t["частка_групи"] = t["n"] / len(df)
    print(t.round(3).sort_values("n", ascending=False), "\n")
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
df["sex"].value_counts().plot.bar(ax=axs[0], color=["#4C72B0", "#DD8452"])
axs[0].set_title("Кількість записів за статтю"); axs[0].set_ylabel("записів")
df.groupby("sex")["y"].mean().plot.bar(ax=axs[1], color=["#DD8452", "#4C72B0"])
axs[1].set_title("P(income>50K | стать)"); axs[1].set_ylim(0, 1)
df.groupby("race")["y"].mean().sort_values().plot.barh(ax=axs[2], color="#55A868")
axs[2].set_title("P(income>50K | раса)"); axs[2].set_xlim(0, 1)
for a in axs: a.tick_params(axis="x", rotation=0)
fig.tight_layout(); fig.savefig(FIG / "l2_groups.png", dpi=150)

# %% [Р1.3] Репрезентативність: порівняння з довідковими даними (US Census / BLS, 1994)
ref = pd.DataFrame({"Adult": df["sex"].value_counts(normalize=True),
                    "Населення США 1994": pd.Series({"Female": 0.512, "Male": 0.488}),
                    "Робоча сила США 1994 (BLS)": pd.Series({"Female": 0.460, "Male": 0.540})})
print(ref.round(3))

# %% [Р1.4] Proxy-ознаки: залежність від статі (mutual information, χ², Крамер V)
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency
feats = [c for c in cols if c not in ("sex", "income", "fnlwgt")]
d = df.dropna(subset=feats).copy()
s = (d["sex"] == "Male").astype(int)
X_enc = d[feats].apply(lambda c: c.astype("category").cat.codes if c.dtype == object
                       or str(c.dtype) in ("str", "string") else c)
disc = [not pd.api.types.is_numeric_dtype(d[c]) for c in feats]
mi = mutual_info_classif(X_enc, s, discrete_features=disc, random_state=RANDOM_STATE)
rows = []
for c, m in zip(feats, mi):
    x = d[c] if not pd.api.types.is_numeric_dtype(d[c]) else pd.qcut(d[c], 5, duplicates="drop")
    if x.nunique() < 2:                       # capital_gain/loss: здебільшого нулі
        x = d[c] > 0
    ct = pd.crosstab(x, d["sex"]); chi2, p, *_ = chi2_contingency(ct)
    v = np.sqrt(chi2 / (len(d) * (min(ct.shape) - 1)))
    rows.append([c, round(m, 4), round(chi2, 0), p, round(v, 3)])
proxy = pd.DataFrame(rows, columns=["ознака", "MI", "chi2", "p", "Cramer_V"]) \
    .sort_values("MI", ascending=False)
print(proxy.to_string(index=False))
print("\nТоп-5 proxy:", proxy.head(5)["ознака"].tolist())

# %% [Р2.1] Підготовка і стратифіковане розбиття за ціллю та статтю
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
num = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
cat = ["workclass", "marital_status", "occupation", "relationship", "race", "native_country"]
X = df[num + cat + ["sex"]]; y = df["y"]; A = df["sex"]
strat = y.astype(str) + "_" + A
Xtr, Xte, ytr, yte, Atr, Ate = train_test_split(X, y, A, test_size=0.3, stratify=strat,
                                                random_state=RANDOM_STATE)
print("train:", Xtr.shape, "test:", Xte.shape)
print(pd.crosstab(Ate, yte, normalize="index").round(3))

def make_model(with_sex):
    c = cat + (["sex"] if with_sex else [])
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), c)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])

# %% [Р2.2] Логістична регресія з чутливим атрибутом і без нього
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
res, models, preds = [], {}, {}
for name, ws in [("LR з атрибутом sex", True), ("LR без атрибута sex", False)]:
    m = make_model(ws).fit(Xtr, ytr); models[name] = m
    p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]; preds[name] = (p, pr)
    res.append([name, accuracy_score(yte, p), precision_score(yte, p),
                recall_score(yte, p), roc_auc_score(yte, pr)])
print(pd.DataFrame(res, columns=["модель", "accuracy", "precision", "recall", "ROC-AUC"])
      .round(4).to_string(index=False))

# %% [Р2.3] Метрики за групами (fairlearn MetricFrame) + bootstrap ДІ
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate,
                               false_positive_rate, count)
mets = {"n": count, "selection_rate": selection_rate, "TPR": true_positive_rate,
        "FPR": false_positive_rate, "accuracy": accuracy_score}
frames = {}
for name, (p, _) in preds.items():
    mf = MetricFrame(metrics=mets, y_true=yte, y_pred=p, sensitive_features=Ate)
    frames[name] = mf
    print(name); print(mf.by_group.round(3).to_string(), "\n")

rng = np.random.default_rng(RANDOM_STATE)
p = preds["LR без атрибута sex"][0]
yv, av = yte.to_numpy(), Ate.to_numpy()
for g in ["Female", "Male"]:
    idx = np.where(av == g)[0]; bs = {"selection_rate": [], "TPR": []}
    for _ in range(1000):
        b = rng.choice(idx, len(idx), replace=True)
        bs["selection_rate"].append(p[b].mean())
        pos = b[yv[b] == 1]; bs["TPR"].append(p[pos].mean())
    print(g, {k: np.round(np.percentile(v, [2.5, 97.5]), 3).tolist() for k, v in bs.items()})

# %% [Р3.1] Розриви справедливості та графік метрик за групами
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
gaps = []
for name, mf in frames.items():
    bg = mf.by_group
    gaps.append([name, bg.selection_rate.max() - bg.selection_rate.min(),
                 bg.TPR.max() - bg.TPR.min(), bg.FPR.max() - bg.FPR.min()])
gaps = pd.DataFrame(gaps, columns=["модель", "DP diff (selection)", "EO diff (TPR)",
                                   "FPR diff"]).round(4)
print(gaps.to_string(index=False))
bg = frames["LR без атрибута sex"].by_group[["selection_rate", "TPR", "FPR", "accuracy"]]
ax = bg.T.plot.bar(figsize=(8, 4), color=["#DD8452", "#4C72B0"], rot=0)
ax.set_ylim(0, 1); ax.set_ylabel("значення метрики")
ax.set_title("Метрики LR (без sex) за статтю: жінкам рідше прогнозують >50K")
plt.tight_layout(); plt.savefig(FIG / "l2_group_metrics.png", dpi=150)
print("\nБазові частоти:", df.groupby("sex")["y"].mean().round(3).to_dict())

# %% [Р4.1] Пом'якшення: ThresholdOptimizer та ExponentiatedGradient(DemographicParity)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
base = models["LR без атрибута sex"]
to = ThresholdOptimizer(estimator=base, constraints="demographic_parity",
                        objective="accuracy_score", prefit=True, predict_method="predict_proba")
to.fit(Xtr, ytr, sensitive_features=Atr)
p_to = to.predict(Xte, sensitive_features=Ate, random_state=RANDOM_STATE)
pre = base.named_steps["pre"]
Xtr_t, Xte_t = pre.transform(Xtr), pre.transform(Xte)
eg = ExponentiatedGradient(LogisticRegression(max_iter=3000), DemographicParity(difference_bound=0.02))
eg.fit(Xtr_t, ytr, sensitive_features=Atr)
p_eg = eg.predict(Xte_t, random_state=RANDOM_STATE)
pr_eg = eg._pmf_predict(Xte_t)[:, 1]

def summary(name, p, pr=None):
    mf = MetricFrame(metrics={"sr": selection_rate, "tpr": true_positive_rate,
                              "fpr": false_positive_rate}, y_true=yte, y_pred=p,
                     sensitive_features=Ate)
    d = mf.difference()
    return [name, accuracy_score(yte, p), roc_auc_score(yte, pr) if pr is not None else np.nan,
            d["sr"], d["tpr"], d["fpr"]]
ba = pd.DataFrame([summary("До: LR без sex", *preds["LR без атрибута sex"]),
                   summary("Після: ThresholdOptimizer(DP)", p_to),
                   summary("Після: ExpGradient(DP, 0.02)", p_eg, pr_eg)],
                  columns=["модель", "accuracy", "ROC-AUC", "DP diff", "TPR diff", "FPR diff"])
print(ba.round(4).to_string(index=False))

# %% [Р4.2] Компроміс «точність — розрив» для різних обмежень
curve = []
for eps in [0.005, 0.01, 0.02, 0.04, 0.07, 0.10, 0.15]:
    e = ExponentiatedGradient(LogisticRegression(max_iter=3000),
                              DemographicParity(difference_bound=eps)).fit(
        Xtr_t, ytr, sensitive_features=Atr)
    pp = e.predict(Xte_t, random_state=RANDOM_STATE)
    curve.append([eps, accuracy_score(yte, pp), demographic_parity_difference(
        yte, pp, sensitive_features=Ate)])
curve = pd.DataFrame(curve, columns=["eps", "accuracy", "DP_diff"])
print(curve.round(4).to_string(index=False))
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(curve.DP_diff, curve.accuracy, "o-", label="ExpGradient (різні eps)")
b0 = ba.iloc[0]; b1 = ba.iloc[1]
ax.scatter(b0["DP diff"], b0["accuracy"], c="red", s=70, label="LR без пом'якшення")
ax.scatter(b1["DP diff"], b1["accuracy"], c="green", s=70, marker="s", label="ThresholdOptimizer")
ax.set_xlabel("розрив demographic parity"); ax.set_ylabel("accuracy")
ax.set_title("Компроміс: усунення розриву коштує ~2 п.п. точності"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "l2_tradeoff.png", dpi=150)
