# %% [Р0] Імпорт і підготовка очищеного набору (як у ЛР 5)
import sys, warnings
import numpy as np, pandas as pd, matplotlib, sklearn, seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
RANDOM_STATE = 42
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
pd.set_option("display.width", 150)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]  # ≤ 6 кольорів
sns.set_theme(style="whitegrid", palette=PALETTE)
df = pd.read_csv("../data/hotels.csv", na_values=["NULL", "NA"])
df = df.drop(columns=["reservation_status", "reservation_status_date", "assigned_room_type"])
df = df.drop_duplicates().reset_index(drop=True)
df["arrival"] = pd.to_datetime(df.arrival_date_year.astype(str) + "-" + df.arrival_date_month
                               + "-" + df.arrival_date_day_of_month.astype(str), format="%Y-%B-%d")
df = df[(df.adr > 0) & (df.adr < 1000) & (df.adults > 0)].copy()     # прибрати некоректні ціни
df["nights"] = df.stays_in_week_nights + df.stays_in_weekend_nights
df["has_agent"] = df.agent.notna().astype(int)
df = df.sort_values("arrival").reset_index(drop=True)
print("Python", sys.version.split()[0], "| sklearn", sklearn.__version__, "| seaborn", sns.__version__)
print("Очищений набір:", df.shape, "| частка скасувань:", round(df.is_canceled.mean(), 3))

# %% [Р1.1] Панель: 6 типів діаграм із заголовками-висновками
fig, ax = plt.subplots(figsize=(7, 4))
sns.violinplot(data=df, x="hotel", y="adr", hue="hotel", cut=0, ax=ax, legend=False)
ax.set(ylim=(0, 400), xlabel="тип готелю", ylabel="середня ціна за ніч (ADR), EUR",
       title="Ціни в міському готелі вищі й стабільніші, курортний має довгий хвіст")
fig.tight_layout(); fig.savefig(FIG / "l6_1_violin.png", dpi=150); plt.close()

rate = df.groupby("market_segment").is_canceled.mean().sort_values()
rate = rate[df.market_segment.value_counts()[rate.index] > 100]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(rate.index, rate.values * 100, color=PALETTE[0]); ax.set_xlim(0, 100)
ax.set(xlabel="частка скасувань, %", ylabel="сегмент ринку",
       title="Найчастіше скасовують бронювання через онлайн-агентства")
for i, v in enumerate(rate.values): ax.text(v * 100 + 1, i, f"{v:.0%}", va="center")
fig.tight_layout(); fig.savefig(FIG / "l6_2_bar.png", dpi=150); plt.close()

smp = df.sample(4000, random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(7, 4))
sns.scatterplot(data=smp, x="lead_time", y="adr", hue=smp.is_canceled.map({0: "ні", 1: "так"}),
                s=10, alpha=.5, ax=ax)
ax.set(xlabel="завчасність бронювання (lead time), дні", ylabel="ADR, EUR",
       title="Далекі бронювання частіше скасовують; ціна від lead time майже не залежить")
ax.legend(title="скасовано"); fig.tight_layout(); fig.savefig(FIG / "l6_3_scatter.png", dpi=150); plt.close()

ts = df.set_index("arrival").groupby("hotel").resample("MS").size().unstack(0)
fig, ax = plt.subplots(figsize=(8, 4))
ts.plot(ax=ax, marker="o", ms=3); ax.set_ylim(0, None)
ax.set(xlabel="місяць заїзду", ylabel="кількість бронювань, шт.",
       title="Сезонність: пік бронювань — літо, мінімум — січень")
fig.tight_layout(); fig.savefig(FIG / "l6_4_line.png", dpi=150); plt.close()

cols = ["is_canceled", "lead_time", "adr", "nights", "adults", "previous_cancellations",
        "booking_changes", "total_of_special_requests", "required_car_parking_spaces", "has_agent"]
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(df[cols].corr(method="spearman"), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1,
            vmax=1, ax=ax, annot_kws={"size": 8})
ax.set_title("Спецзапити й паркування знижують ризик скасування, lead time — підвищує")
fig.tight_layout(); fig.savefig(FIG / "l6_5_heatmap.png", dpi=150); plt.close()

g = sns.pairplot(smp.sample(1500, random_state=1)[["lead_time", "adr", "nights",
                 "total_of_special_requests", "is_canceled"]], hue="is_canceled",
                 plot_kws={"s": 8, "alpha": .5}, corner=True, height=1.9)
g.figure.suptitle("Pairplot: скасування зосереджені у великих lead time і без спецзапитів", y=1.02)
g.savefig(FIG / "l6_6_pairplot.png", dpi=150); plt.close()
print("Збережено 6 діаграм:", sorted(p.name for p in FIG.glob("l6_[1-6]_*.png")))

# %% [Р1.2] Погана діаграма та її виправлення
dep = df.groupby("deposit_type").is_canceled.mean()
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].bar(dep.index, dep.values, color=["red", "yellow", "lime"])
axs[0].set_ylim(0.2, 1.0)                     # погано: вісь не з нуля
axs[0].set_title("ПОГАНО: вісь з 0.2, кислотні кольори,\nбез одиниць і висновку", color="#C44E52")
axs[1].bar(dep.index, dep.values * 100, color=PALETTE[0]); axs[1].set_ylim(0, 100)
for i, v in enumerate(dep.values): axs[1].text(i, v * 100 + 2, f"{v:.0%}", ha="center")
axs[1].set(xlabel="тип депозиту", ylabel="частка скасувань, %")
axs[1].set_title("ДОБРЕ: «Non Refund» скасовують у 95 % випадків", color="#2E7D32")
fig.tight_layout(); fig.savefig(FIG / "l6_bad_good.png", dpi=150); plt.close()
print((dep * 100).round(1).to_string())

# %% [Р2.1] Розбиття train/validation/test за часом (60/20/20) і моделі класифікації
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)
NUM = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
       "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
       "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces",
       "total_of_special_requests", "has_agent", "arrival_date_week_number"]
CAT = ["hotel", "meal", "country", "market_segment", "distribution_channel",
       "reserved_room_type", "deposit_type", "customer_type"]
n = len(df); i1, i2 = int(n * .6), int(n * .8)
tr, va, te = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
for name, d in [("train", tr), ("valid", va), ("test", te)]:
    print(f"{name}: {len(d):6d} | {d.arrival.min().date()} — {d.arrival.max().date()} | "
          f"скасувань {d.is_canceled.mean():.3f}")
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=30,
                                           sparse_output=False))]), CAT)])
models = {"Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
          "LogisticRegression": LogisticRegression(max_iter=3000),
          "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE)}
fitted, rows = {}, []
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, m) in zip(axs, models.items()):
    p = Pipeline([("pre", pre), ("clf", m)]).fit(tr[NUM + CAT], tr.is_canceled)
    fitted[name] = p; yp = p.predict(te[NUM + CAT]); yt = te.is_canceled
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    rows.append([name, tn, fp, fn, tp, accuracy_score(yt, yp), precision_score(yt, yp, zero_division=0),
                 recall_score(yt, yp), f1_score(yt, yp)])
    ConfusionMatrixDisplay.from_predictions(yt, yp, ax=ax, cmap="Blues", colorbar=False,
                                            display_labels=["не скас.", "скас."])
    ax.set_title(name)
fig.tight_layout(); fig.savefig(FIG / "l6_confusion.png", dpi=150); plt.close()
res = pd.DataFrame(rows, columns=["модель", "TN", "FP", "FN", "TP", "accuracy", "precision",
                                  "recall", "F1"]).round(3)
print(res.to_string(index=False))

# %% [Р3.1] ROC- та PR-криві, калібрування
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score)
from sklearn.calibration import CalibrationDisplay
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
for name in ["LogisticRegression", "HistGradientBoosting"]:
    pr = fitted[name].predict_proba(te[NUM + CAT])[:, 1]
    fpr, tpr, _ = roc_curve(te.is_canceled, pr); auc = roc_auc_score(te.is_canceled, pr)
    pc, rc, _ = precision_recall_curve(te.is_canceled, pr)
    ap = average_precision_score(te.is_canceled, pr)
    axs[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    axs[1].plot(rc, pc, label=f"{name} (AP={ap:.3f})")
    CalibrationDisplay.from_predictions(te.is_canceled, pr, n_bins=10, ax=axs[2], name=name)
    print(f"{name:22s} ROC-AUC = {auc:.4f} | AP = {ap:.4f}")
axs[0].plot([0, 1], [0, 1], "k--", label="випадкова"); axs[0].legend()
axs[0].set(xlabel="FPR", ylabel="TPR", title="ROC: обидві моделі майже рівні (AUC ≈ 0.80)")
base = te.is_canceled.mean()
axs[1].axhline(base, ls="--", c="k", label=f"baseline = {base:.2f}"); axs[1].legend()
axs[1].set(xlabel="recall", ylabel="precision", title="PR-криві")
axs[2].set_title("Калібрування: моделі занижують ризик при p = 0.1–0.6")
fig.tight_layout(); fig.savefig(FIG / "l6_roc_pr_cal.png", dpi=150); plt.close()

# %% [Р4.1] Поріг за вартістю помилок (вибір на validation, оцінка на test)
rev = (tr.adr * tr.nights.clip(lower=1)).median()
C_FN = round(rev)          # пропущене скасування -> порожній номер, втрачена виручка
C_FP = 150                 # хибна тривога -> зайвий овербукінг: переселення + компенсація
print(f"C_FN = {C_FN} EUR (медіанна виручка бронювання), C_FP = {C_FP} EUR")
best = fitted["HistGradientBoosting"]
pv = best.predict_proba(va[NUM + CAT])[:, 1]; yv = va.is_canceled.to_numpy()
ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
def cost(y, p, t):
    yp = (p >= t).astype(int)
    return (C_FP * ((yp == 1) & (y == 0)).sum() + C_FN * ((yp == 0) & (y == 1)).sum()) / len(y)
cv_ = [cost(yv, pv, t) for t in ths]
t_best = ths[int(np.argmin(cv_))]
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(ths, cv_, "o-"); ax.axvline(t_best, c="g", ls="--", label=f"мінімум: поріг {t_best}")
ax.axvline(0.5, c="r", ls=":", label="поріг 0.5"); ax.set_ylim(0, None)
ax.set(xlabel="поріг класифікації", ylabel="очікувані витрати, EUR на бронювання",
       title="Оптимальний поріг нижчий за 0.5, бо FN дорожчий за FP"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "l6_cost.png", dpi=150); plt.close()
pt = best.predict_proba(te[NUM + CAT])[:, 1]; yt = te.is_canceled.to_numpy()
out = []
for t in [0.5, t_best]:
    yp = (pt >= t).astype(int)
    out.append([t, cost(yt, pt, t), cost(yt, pt, t) * len(yt), precision_score(yt, yp),
                recall_score(yt, yp), ((yp == 1) & (yt == 0)).sum(), ((yp == 0) & (yt == 1)).sum()])
out = pd.DataFrame(out, columns=["поріг", "EUR/бронювання", "EUR усього", "precision", "recall",
                                 "FP", "FN"]).round(3)
print(out.to_string(index=False))
print(f"Економія на тесті: {out['EUR усього'][0] - out['EUR усього'][1]:,.0f} EUR")

# %% [Р5.1] Регресія ADR: метрики на test vs baseline, 5-кратна CV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, KFold, learning_curve, validation_curve
NUM_R = [c for c in NUM if c != "adr"] + ["is_canceled"]
pre_r = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM_R),
                           ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                                             ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=30,
                                                                  sparse_output=False))]), CAT)])
trv = pd.concat([tr, va])
regs = {"Dummy (mean)": DummyRegressor(),
        "HistGBRegressor": HistGradientBoostingRegressor(random_state=RANDOM_STATE)}
rows = []
for name, m in regs.items():
    p = Pipeline([("pre", pre_r), ("reg", m)]).fit(trv[NUM_R + CAT], trv.adr)
    yp = p.predict(te[NUM_R + CAT])
    rows.append([name, mean_absolute_error(te.adr, yp), mean_squared_error(te.adr, yp) ** .5,
                 r2_score(te.adr, yp)])
print(pd.DataFrame(rows, columns=["модель", "MAE", "RMSE", "R2"]).round(3).to_string(index=False))
reg_pipe = Pipeline([("pre", pre_r), ("reg", HistGradientBoostingRegressor(random_state=RANDOM_STATE))])
cvr = cross_validate(reg_pipe, trv[NUM_R + CAT], trv.adr, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
                     scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error",
                              "R2": "r2"}, return_train_score=True)
for k in ["MAE", "RMSE", "R2"]:
    s = cvr[f"test_{k}"] * (-1 if k != "R2" else 1); s_tr = cvr[f"train_{k}"] * (-1 if k != "R2" else 1)
    print(f"CV {k:4s}: test {s.mean():.3f} ± {s.std():.3f} | train {s_tr.mean():.3f}")

# %% [Р5.2] Криві навчання та валідаційна крива (складність = max_depth)
sub = trv.sample(20000, random_state=RANDOM_STATE)
Xs, ys = sub[NUM_R + CAT], sub.adr
cv5 = KFold(5, shuffle=True, random_state=RANDOM_STATE)
sizes, tr_sc, va_sc = learning_curve(reg_pipe, Xs, ys, cv=cv5, scoring="neg_root_mean_squared_error",
                                     train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
depths = [1, 2, 3, 4, 6, 8, 12, 16, None]
vtr, vva = validation_curve(reg_pipe, Xs, ys, param_name="reg__max_depth", param_range=depths,
                            cv=cv5, scoring="neg_root_mean_squared_error", n_jobs=-1)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
for sc, lbl in [(-tr_sc, "train"), (-va_sc, "validation")]:
    axs[0].plot(sizes, sc.mean(1), "o-", label=lbl)
    axs[0].fill_between(sizes, sc.mean(1) - sc.std(1), sc.mean(1) + sc.std(1), alpha=.2)
axs[0].set(xlabel="розмір навчальної вибірки", ylabel="RMSE, EUR",
           title="Крива навчання: розрив зменшується з обсягом даних"); axs[0].legend()
xs = [str(d) for d in depths]
for sc, lbl in [(-vtr, "train"), (-vva, "validation")]:
    axs[1].plot(xs, sc.mean(1), "o-", label=lbl)
    axs[1].fill_between(xs, sc.mean(1) - sc.std(1), sc.mean(1) + sc.std(1), alpha=.2)
axs[1].set(xlabel="max_depth дерев бустингу", ylabel="RMSE, EUR",
           title="Валідаційна крива: недонавчання при depth ≤ 3"); axs[1].legend()
fig.tight_layout(); fig.savefig(FIG / "l6_curves.png", dpi=150); plt.close()
print("learning curve RMSE (train/valid):")
print(pd.DataFrame({"n": sizes, "train": -tr_sc.mean(1), "valid": -va_sc.mean(1)}).round(2).to_string(index=False))
print("validation curve RMSE (train/valid):")
print(pd.DataFrame({"max_depth": xs, "train": -vtr.mean(1), "valid": -vva.mean(1)}).round(2).to_string(index=False))
