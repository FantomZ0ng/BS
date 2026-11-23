# %% [Р1.1] Економічна цінність: модель з явними припущеннями
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
pd.set_option("display.width", 160)
Path("figures").mkdir(exist_ok=True)
A = {  # припущення (джерела: регламент KB-16, журнали ЛР 14, оцінки служби)
    "звернень на місяць": 900,               # ~45 на робочий день у службі 1-ї лінії
    "частка, яку покриває асистент": 0.55,   # частка типових діагностичних питань (тип simple/multi в еталоні — 69 %)
    "хв на пошук відповіді без асистента": 12,
    "хв з асистентом (питання + перевірка)": 4,
    "вартість години інженера, грн": 220,
    "ескалацій на 2-гу лінію на місяць": 120,
    "частка ескалацій, яких уникнуто": 0.15,
    "вартість ескалації, грн": 350,
    "LLM, $ / 1000 запитів (ЛР 15)": 0.262,
    "курс, грн/$": 41.5,
    "електроенергія та обслуговування сервера, грн/міс": 900,
    "розробка й підтримка, грн/міс (0.1 ставки)": 3500,
}
q = A["звернень на місяць"] * A["частка, яку покриває асистент"]
time_saved_h = q * (A["хв на пошук відповіді без асистента"] - A["хв з асистентом (питання + перевірка)"]) / 60
V = {"запитів до асистента / міс": q,
     "зекономлено годин / міс": time_saved_h,
     "економія часу, грн": time_saved_h * A["вартість години інженера, грн"],
     "економія на ескалаціях, грн": A["ескалацій на 2-гу лінію на місяць"] * A["частка ескалацій, яких уникнуто"] * A["вартість ескалації, грн"],
     "витрати LLM, грн": q / 1000 * A["LLM, $ / 1000 запитів (ЛР 15)"] * A["курс, грн/$"],
     "витрати сервер + підтримка, грн": A["електроенергія та обслуговування сервера, грн/міс"] + A["розробка й підтримка, грн/міс (0.1 ставки)"]}
V["чиста вигода, грн / міс"] = V["економія часу, грн"] + V["економія на ескалаціях, грн"] - V["витрати LLM, грн"] - V["витрати сервер + підтримка, грн"]
print(pd.Series(A).to_string()); print(); print(pd.Series(V).round(1).to_string())
# чутливість: частка покриття і зекономлені хвилини
sens = pd.DataFrame({cov: {m: round(A["звернень на місяць"] * cov * m / 60 * A["вартість години інженера, грн"]
                                     + V["економія на ескалаціях, грн"] - V["витрати LLM, грн"] - V["витрати сервер + підтримка, грн"])
                           for m in (4, 8, 12)} for cov in (0.3, 0.55, 0.8)})
sens.index.name = "хв зекономлено / запит"; sens.columns.name = "частка покриття"
print("\nЧутливість чистої вигоди, грн/міс:"); print(sens.to_string())
json.dump({k: float(v) for k, v in V.items()}, open("value.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# %% [Р3.1] Зведений рисунок «до/після» (ЛР 13 → ЛР 15)
after = json.load(open("../15/outputs_after.json", encoding="utf-8"))
keys = ["R@3", "MRR", "faithfulness", "completeness", "refusal_ok"]
B, Aft = after["база (v1.0)"], after["C1+C2 (v2)"]
norm = lambda d, k: d[k] / 5 if k in ("faithfulness", "completeness") else d[k]
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(keys))
ax.bar(x - 0.2, [norm(B, k) for k in keys], 0.4, label="до (v1.0, ЛР 13)", color="#A5A5A5")
ax.bar(x + 0.2, [norm(Aft, k) for k in keys], 0.4, label="після (v1.1, ЛР 15)", color="#2E75B6")
for i, k in enumerate(keys):
    ax.text(i + 0.2, norm(Aft, k) + 0.02, f"{Aft[k]:.2f}", ha="center", fontsize=8)
    ax.text(i - 0.2, norm(B, k) + 0.02, f"{B[k]:.2f}", ha="center", fontsize=8, color="#555")
ax.set_xticks(x, ["recall@3", "MRR", "faithfulness /5", "повнота /5", "правильні\nвідмови"])
ax.set_ylim(0, 1.12); ax.legend(); ax.grid(axis="y", alpha=.3); ax.set_title("Якість системи до і після покращень")
fig.tight_layout(); fig.savefig("figures/l16_before_after.png", dpi=170); plt.close(fig)

# %% [Р4.1] Матриця ризиків
R = [("R1 галюцинації", 3, 4), ("R2 prompt injection", 3, 4), ("R3 витік ПД", 2, 5), ("R4 відмова LLM/БД", 3, 3),
     ("R5 дрейф/застарілі знання", 4, 3), ("R6 надмірна довіра техніків", 3, 3), ("R7 затримка > 30 с", 4, 2)]
fig, ax = plt.subplots(figsize=(6.5, 5))
for i in range(1, 6):
    for j in range(1, 6):
        ax.add_patch(plt.Rectangle((i - .5, j - .5), 1, 1, color=plt.cm.RdYlGn_r((i * j) / 25), alpha=.35))
off = {}
for name, p, imp in R:
    k = (p, imp); off[k] = off.get(k, 0) + 1
    ax.scatter(p, imp, s=120, c="#1F3864"); ax.annotate(name, (p, imp), xytext=(6, 4 - 13 * (off[k] - 1)),
                                                        textcoords="offset points", fontsize=8)
ax.set_xlim(.5, 5.5); ax.set_ylim(.5, 5.5); ax.set_xlabel("ймовірність (1–5)"); ax.set_ylabel("вплив (1–5)")
ax.set_title("Матриця ризиків (до заходів)")
fig.tight_layout(); fig.savefig("figures/l16_risks.png", dpi=170); plt.close(fig)
print("Збережено figures/l16_before_after.png, figures/l16_risks.png")
