# %% [Р1.1] Імпорт, дані Medical Cost (insurance.csv), описова статистика
import sys, warnings
import numpy as np, pandas as pd, matplotlib, scipy, statsmodels
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from pathlib import Path
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
RANDOM_STATE = 42
rng = np.random.default_rng(42)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
pd.set_option("display.width", 140)
print("Python", sys.version.split()[0], "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| statsmodels", statsmodels.__version__)
df = pd.read_csv("../data/insurance.csv")
print("shape:", df.shape); print(df.head())
num = ["age", "bmi", "children", "charges"]
desc = pd.DataFrame({
    "mean": df[num].mean(), "median": df[num].median(), "std": df[num].std(),
    "IQR": df[num].quantile(.75) - df[num].quantile(.25),
    "skew": df[num].apply(stats.skew), "kurtosis": df[num].apply(stats.kurtosis)})
print(desc.round(3))

# %% [Р1.2] Гістограми, boxplot, викиди за Тьюкі
fig, axs = plt.subplots(2, 4, figsize=(15, 6))
units = {"age": "роки", "bmi": "кг/м²", "children": "осіб", "charges": "USD"}
for i, c in enumerate(num):
    axs[0, i].hist(df[c], bins=30, color="#4C72B0", edgecolor="white")
    axs[0, i].set_title(f"Гістограма {c}"); axs[0, i].set_xlabel(units[c])
    axs[1, i].boxplot(df[c], vert=False, whis=1.5)
    axs[1, i].set_title(f"Boxplot {c}"); axs[1, i].set_xlabel(units[c])
fig.tight_layout(); fig.savefig(FIG / "l3_hist_box.png", dpi=150)
for c in num:
    q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
    n_out = ((df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)).sum()
    print(f"{c:9s} викидів за Тьюкі: {n_out}")

# %% [Р1.3] Нормальність: Q–Q графіки та тест Шапіро–Вілка (вибірка 500)
sample = df.sample(500, random_state=RANDOM_STATE)
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for i, c in enumerate(num):
    stats.probplot(sample[c], dist="norm", plot=axs[i]); axs[i].set_title(f"Q–Q: {c}")
    W, p = stats.shapiro(sample[c])
    print(f"{c:9s} Shapiro W={W:.4f}, p={p:.2e} -> {'нормальний' if p > .05 else 'НЕ нормальний'}")
fig.tight_layout(); fig.savefig(FIG / "l3_qq.png", dpi=150)

# %% [Р1.4] Перетворення log1p для асиметричної ознаки charges
df["log_charges"] = np.log1p(df["charges"])
print("Асиметрія charges до:", round(stats.skew(df.charges), 3),
      "| після log1p:", round(stats.skew(df.log_charges), 3))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
axs[0].hist(df.charges, bins=40, color="#C44E52"); axs[0].set_title("charges (сильна правостороння асиметрія)")
axs[0].set_xlabel("USD")
axs[1].hist(df.log_charges, bins=40, color="#55A868"); axs[1].set_title("log1p(charges): асиметрія майже зникла")
axs[1].set_xlabel("ln(USD+1)")
fig.tight_layout(); fig.savefig(FIG / "l3_log.png", dpi=150)

# %% [Р2.1] 95 % ДІ середнього bmi: формула t та bootstrap
x = df["bmi"].to_numpy(); n = len(x)
m, s = x.mean(), x.std(ddof=1); t_crit = stats.t.ppf(0.975, n - 1)
ci_t = (m - t_crit * s / np.sqrt(n), m + t_crit * s / np.sqrt(n))
boot = np.array([rng.choice(x, n, replace=True).mean() for _ in range(5000)])
ci_b = np.percentile(boot, [2.5, 97.5])
print(f"x̄ = {m:.3f}, s = {s:.3f}, n = {n}, t* = {t_crit:.4f}")
print(f"ДІ (формула t):  [{ci_t[0]:.3f}; {ci_t[1]:.3f}]  ширина {ci_t[1]-ci_t[0]:.3f}")
print(f"ДІ (bootstrap):  [{ci_b[0]:.3f}; {ci_b[1]:.3f}]  ширина {ci_b[1]-ci_b[0]:.3f}")

# %% [Р2.2] Перевірка гіпотези: чи відрізняються витрати курців і некурців (Welch t-test)
g1 = df.loc[df.smoker == "yes", "charges"].to_numpy()
g0 = df.loc[df.smoker == "no", "charges"].to_numpy()
t, p = stats.ttest_ind(g1, g0, equal_var=False)
sp = np.sqrt(((len(g1) - 1) * g1.var(ddof=1) + (len(g0) - 1) * g0.var(ddof=1)) / (len(g1) + len(g0) - 2))
d = (g1.mean() - g0.mean()) / sp
diff = np.array([rng.choice(g1, len(g1)).mean() - rng.choice(g0, len(g0)).mean() for _ in range(5000)])
print(f"n курців = {len(g1)}, n некурців = {len(g0)}")
print(f"Середні: курці {g1.mean():.0f} USD, некурці {g0.mean():.0f} USD")
print(f"Welch t = {t:.2f}, p = {p:.2e}")
print(f"d Коена = {d:.2f}")
print(f"Різниця середніх = {g1.mean()-g0.mean():.0f} USD, bootstrap 95% ДІ: "
      f"[{np.percentile(diff, 2.5):.0f}; {np.percentile(diff, 97.5):.0f}]")
# додатково — порівняння bmi за статтю (менший ефект)
t2, p2 = stats.ttest_ind(df.bmi[df.sex == "male"], df.bmi[df.sex == "female"], equal_var=False)
print(f"bmi чоловіки vs жінки: t = {t2:.2f}, p = {p2:.3f}")

# %% [Р3.1] Матриці кореляцій Пірсона і Спірмена
dd = df.assign(smoker_yes=(df.smoker == "yes").astype(int), male=(df.sex == "male").astype(int))
cc = ["age", "bmi", "children", "smoker_yes", "male", "charges"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, meth in zip(axs, ["pearson", "spearman"]):
    sns.heatmap(dd[cc].corr(method=meth), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax); ax.set_title(f"Кореляція {meth.capitalize()}")
fig.tight_layout(); fig.savefig(FIG / "l3_corr.png", dpi=150)
print(dd[cc].corr(method="pearson")["charges"].round(3).to_string())

# %% [Р3.2] OLS: charges ~ age + bmi + children + smoker
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
tr, te = train_test_split(dd, test_size=0.2, random_state=RANDOM_STATE)
preds = ["age", "bmi", "children", "smoker_yes"]
Xtr = sm.add_constant(tr[preds]); ols = sm.OLS(tr["charges"], Xtr).fit()
print(ols.summary())

# %% [Р3.3] VIF та діагностика залишків
vif = pd.Series([variance_inflation_factor(Xtr.values, i) for i in range(1, Xtr.shape[1])],
                index=preds, name="VIF")
print(vif.round(3).to_string())
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].scatter(ols.fittedvalues, ols.resid, s=8, alpha=.5); axs[0].axhline(0, color="r")
axs[0].set_xlabel("прогноз, USD"); axs[0].set_ylabel("залишок, USD")
axs[0].set_title("Залишки vs прогноз: видно 3 «смуги» — нелінійність")
stats.probplot(ols.resid, plot=axs[1]); axs[1].set_title("Q–Q залишків: важкі хвости")
fig.tight_layout(); fig.savefig(FIG / "l3_resid1.png", dpi=150)
from statsmodels.stats.diagnostic import het_breuschpagan
print("Breusch–Pagan p =", f"{het_breuschpagan(ols.resid, Xtr)[1]:.2e}",
      "| Jarque–Bera p =", f"{stats.jarque_bera(ols.resid).pvalue:.2e}")

# %% [Р3.4] Покращення: log(charges) + ознаки obese та взаємодія smoker×obese
for d_ in (tr, te):
    d_["obese"] = (d_.bmi >= 30).astype(int); d_["smoker_obese"] = d_.smoker_yes * d_.obese
    d_["age2"] = d_.age ** 2
preds2 = ["age", "age2", "bmi", "children", "smoker_yes", "obese", "smoker_obese"]
ols2 = sm.OLS(tr["charges"], sm.add_constant(tr[preds2])).fit()
ols3 = sm.OLS(np.log(tr["charges"]), sm.add_constant(tr[preds2])).fit()
def evaluate(name, pred):
    rmse = np.sqrt(np.mean((te.charges - pred) ** 2))
    r2 = 1 - np.sum((te.charges - pred) ** 2) / np.sum((te.charges - te.charges.mean()) ** 2)
    return [name, round(r2, 4), round(rmse, 1)]
smear = np.mean(np.exp(ols3.resid))   # поправка Дуана при зворотному перетворенні
res = pd.DataFrame([
    evaluate("M1: OLS базова", ols.predict(sm.add_constant(te[preds]))),
    evaluate("M2: + age², obese, smoker×obese", ols2.predict(sm.add_constant(te[preds2]))),
    evaluate("M3: log(charges) + ті ж ознаки", np.exp(ols3.predict(sm.add_constant(te[preds2]))) * smear)],
    columns=["модель", "R2_test", "RMSE_test"])
print(res.to_string(index=False))
print(ols2.params.round(1).to_string())
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(ols2.fittedvalues, ols2.resid, s=8, alpha=.5); ax.axhline(0, color="r")
ax.set_xlabel("прогноз, USD"); ax.set_ylabel("залишок, USD"); ax.set_title("Залишки M2: структура зникла")
fig.tight_layout(); fig.savefig(FIG / "l3_resid2.png", dpi=150)

# %% [Р4.1] Часовий ряд CO2 (Мауна-Лоа), місячні середні; декомпозиція
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import seasonal_decompose
ts = co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
print("Точок:", len(ts), "| період:", ts.index.min().date(), "—", ts.index.max().date())
dec = seasonal_decompose(ts, model="additive", period=12)
fig = dec.plot(); fig.set_size_inches(11, 7)
fig.axes[0].set_title("Декомпозиція CO2: зростаючий тренд + річна сезонність (ppm)")
fig.tight_layout(); fig.savefig(FIG / "l3_decompose.png", dpi=150)

# %% [Р4.2] ADF-тест, диференціювання, ACF/PACF
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
for name, s_ in [("вихідний", ts), ("d=1", ts.diff().dropna()),
                 ("d=1, D=1 (lag 12)", ts.diff().diff(12).dropna())]:
    st_, p_, *_ = adfuller(s_)
    print(f"ADF {name:18s}: stat = {st_:7.3f}, p = {p_:.4f} -> "
          f"{'стаціонарний' if p_ < .05 else 'НЕстаціонарний'}")
sd = ts.diff().diff(12).dropna()
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
plot_acf(sd, lags=36, ax=axs[0]); plot_pacf(sd, lags=36, ax=axs[1], method="ywm")
axs[0].set_title("ACF ряду (1−B)(1−B¹²)CO2"); axs[1].set_title("PACF ряду (1−B)(1−B¹²)CO2")
fig.tight_layout(); fig.savefig(FIG / "l3_acf.png", dpi=150)

# %% [Р4.3] Розбиття 80/20 за часом, наївний прогноз і SARIMA (підбір за AIC)
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.simplefilter("ignore")  # приховати попередження про стартові параметри
split = int(len(ts) * 0.8); train, test = ts[:split], ts[split:]
print("train:", train.index[0].date(), "—", train.index[-1].date(), f"({len(train)})",
      "| test:", test.index[0].date(), "—", test.index[-1].date(), f"({len(test)})")
aic = []
for p_ in (0, 1, 2):
    for q_ in (0, 1, 2):
        mdl = SARIMAX(train, order=(p_, 1, q_), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
        aic.append([(p_, 1, q_), round(mdl.aic, 1)])
aic = pd.DataFrame(aic, columns=["order", "AIC"]).sort_values("AIC")
print(aic.head(5).to_string(index=False))
best_order = aic.iloc[0, 0]
fit = SARIMAX(train, order=best_order, seasonal_order=(0, 1, 1, 12)).fit(disp=False)
fc = fit.get_forecast(len(test)); mean_fc = fc.predicted_mean; ci = fc.conf_int()
naive = pd.Series(train.iloc[-1], index=test.index)
# сезонний наївний: повторюємо значення останнього року навчання
snaive = pd.Series(train.iloc[-12:].to_numpy()[np.arange(len(test)) % 12], index=test.index)
def m_(y, f): return [np.mean(np.abs(y - f)), np.sqrt(np.mean((y - f) ** 2))]
cmp = pd.DataFrame([["Наївний (останнє значення)", *m_(test, naive)],
                    ["Сезонний наївний (рік тому)", *m_(test, snaive)],
                    [f"SARIMA{best_order}x(0,1,1,12)", *m_(test, mean_fc)]],
                   columns=["модель", "MAE, ppm", "RMSE, ppm"]).round(3)
print(cmp.to_string(index=False))
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(train[-120:], label="train"); ax.plot(test, label="test (факт)", color="k")
ax.plot(naive, "--", label="наївний"); ax.plot(mean_fc, label="SARIMA прогноз", color="r")
ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="r", alpha=.15, label="95 % ДІ")
ax.set_ylabel("CO2, ppm"); ax.set_title("SARIMA відтворює тренд і сезонність, наївний — ні"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "l3_forecast.png", dpi=150)

# %% [Р5.1] Монте-Карло: вартість і термін IT-проєкту (10 000 прогонів)
N = 10_000
rng = np.random.default_rng(42)
def simulate(team=5, N=N, rng=rng):
    # обсяг робіт, людино-тижні: трикутний (оптимістично, найімовірніше, песимістично)
    work = rng.triangular(80, 100, 150, N)
    productivity = rng.normal(1.0, 0.12, N).clip(0.6, 1.4)   # частка від норми
    rate = rng.normal(1500, 150, N)                         # USD за людино-тиждень
    risks = rng.poisson(2.0, N)                              # к-сть ризик-подій
    risk_delay = rng.triangular(0.5, 1, 3, (N, 8))           # тижнів на подію
    delay = np.array([risk_delay[i, :k].sum() for i, k in enumerate(np.minimum(risks, 8))])
    coord = 1 + 0.04 * (team - 1)                            # втрати на комунікацію
    duration = work * coord / (team * productivity) + delay
    cost = duration * team * rate + 5000                     # + фіксовані витрати
    return pd.DataFrame(dict(work=work, productivity=productivity, rate=rate,
                             risks=risks, duration=duration, cost=cost))
sim = simulate(); sim.to_csv("mc_results.csv", index=False)
BUDGET, DEADLINE = 240_000, 30   # бюджет (USD) і дедлайн (тижні)
q = sim[["duration", "cost"]].quantile([.05, .5, .95]).round(1)
print(q)
print(f"P(вартість > {BUDGET} USD) = {(sim.cost > BUDGET).mean():.3f}")
print(f"P(термін > {DEADLINE} тиж.) = {(sim.duration > DEADLINE).mean():.3f}")
print(f"P(хоча б одне порушення) = {((sim.cost > BUDGET) | (sim.duration > DEADLINE)).mean():.3f}")
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
axs[0].hist(sim.cost / 1000, bins=60, color="#4C72B0")
for qq, ls in zip(q.cost / 1000, [":", "-", ":"]): axs[0].axvline(qq, color="k", ls=ls)
axs[0].axvline(BUDGET / 1000, color="r", lw=2, label="бюджет"); axs[0].legend()
axs[0].set_xlabel("вартість, тис. USD"); axs[0].set_title("Розподіл вартості (квантилі 5/50/95 %)")
axs[1].hist(sim.duration, bins=60, color="#55A868"); axs[1].axvline(DEADLINE, color="r", lw=2)
axs[1].set_xlabel("тривалість, тижні"); axs[1].set_title("Розподіл тривалості")
fig.tight_layout(); fig.savefig(FIG / "l3_mc_hist.png", dpi=150)

# %% [Р5.2] Аналіз чутливості та вибір параметрів рішення
sens = sim[["work", "productivity", "rate", "risks"]].corrwith(sim.cost, method="spearman")
sens_d = sim[["work", "productivity", "rate", "risks"]].corrwith(sim.duration, method="spearman")
print(pd.DataFrame({"ρ(вартість)": sens, "ρ(термін)": sens_d}).round(3))
fig, ax = plt.subplots(figsize=(6, 3.2))
sens.sort_values().plot.barh(ax=ax, color="#8172B2"); ax.axvline(0, color="k")
ax.set_xlabel("ρ Спірмена з вартістю"); ax.set_title("Торнадо-діаграма: головний драйвер — обсяг робіт")
fig.tight_layout(); fig.savefig(FIG / "l3_tornado.png", dpi=150)
rows = []
for team in range(3, 9):
    s_ = simulate(team, rng=np.random.default_rng(42))
    rows.append([team, s_.duration.median(), s_.cost.median(), (s_.duration > DEADLINE).mean(),
                 (s_.cost > BUDGET).mean(),
                 ((s_.cost > BUDGET) | (s_.duration > DEADLINE)).mean()])
print(pd.DataFrame(rows, columns=["команда", "медіана_тиж", "медіана_USD", "P(зрив терміну)",
                                  "P(перевитрата)", "P(будь-яке)"]).round(3).to_string(index=False))
ok = sim[(sim.cost <= BUDGET) & (sim.duration <= DEADLINE)]
print("Діапазони входів у «успішних» сценаріях (5–95 %):")
print(ok[["work", "productivity", "risks"]].quantile([.05, .95]).round(2))
