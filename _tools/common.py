"""Спільні частини звітів: схема-конвеєр, README, декларація ШІ."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def flow_diagram(path, steps, title):
    """Горизонтальна схема етапів: steps = [(назва, опис), ...]."""
    n = len(steps)
    fig, ax = plt.subplots(figsize=(2.6 * n, 3.2))
    ax.set_xlim(0, n * 2.6); ax.set_ylim(0, 3.2); ax.axis("off")
    colors = ["#DCEAF7", "#E2F0D9", "#FFF2CC", "#FCE4D6", "#EADCF4", "#DDEBF7", "#E7E6E6"]
    for i, (name, desc) in enumerate(steps):
        x = i * 2.6 + 0.15
        ax.add_patch(FancyBboxPatch((x, 0.25), 2.2, 2.45, boxstyle="round,pad=0.04",
                                    fc=colors[i % len(colors)], ec="#404040", lw=1.2))
        ax.text(x + 1.1, 2.4, name, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x + 1.1, 1.3, desc, ha="center", va="center", fontsize=8.6, wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(x + 2.47, 1.45), xytext=(x + 2.25, 1.45),
                        arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#404040"))
    ax.set_title(title, fontsize=12, weight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def readme_and_ai(r, n, data_desc, extra_deps=""):
    r.h2("README: відтворення результатів")
    r.p(f"Файли роботи: `lab{n:02d}.py` (скрипт із комірками `# %%`), `lab{n:02d}.ipynb` "
        f"(виконаний ноутбук з усіма виводами), папка `figures/` (рисунки PNG, 150 dpi). "
        f"Дані: {data_desc}")
    r.p("Середовище: Python 3.12.16; numpy 2.5, pandas 3.1, scikit-learn 1.9, scipy, statsmodels, "
        f"matplotlib 3.11, seaborn{extra_deps}. Фіксований seed: `RANDOM_STATE = 42` "
        "(для NumPy — `np.random.default_rng(42)`). Порядок запуску:")
    r.bullets(["`uv venv --python 3.12 .venv` та `uv pip install -r requirements.txt` "
               "(або `pip install ...` з переліку вище);",
               f"відкрити `lab{n:02d}.ipynb` у VS Code (розширення Jupyter) і виконати Run All, "
               f"або запустити `python _tools/run_lab.py {n}` з кореневої папки — "
               "скрипт перетворить .py у ноутбук і виконає його з нуля."])
    r.h2("Декларація використання ШІ")
    r.p("Інструмент: Claude (Anthropic), асистент для програмування в середовищі Claude Code. "
        "Призначення: допомога в написанні та налагодженні коду Python, підборі відкритих наборів "
        "даних, оформленні звіту (генерування зображень коду, структурування розділів). "
        "Запити: «виконай лабораторну роботу за завданням, напиши код, запусти, оформи звіт». "
        "Перевірка: увесь код виконано з нуля у чистому середовищі; усі числа у звіті взято з "
        "фактичного виводу ноутбука; висновки звірено з графіками й таблицями; теоретичні "
        "відомості звірено з лекцією та рекомендованими джерелами. Студент ознайомився з кожною "
        "частиною роботи та може пояснити її на захисті.")
