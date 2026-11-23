"""Перетворює labNN.py (комірки '# %%') у labNN.ipynb та виконує ноутбук.
Використання: python _tools/run_lab.py 1"""
import sys, re, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

n = int(sys.argv[1])
folder = Path(__file__).resolve().parent.parent / str(n)
src = (folder / f"lab{n:02d}.py").read_text(encoding="utf-8")
chunks = [c.strip("\n") for c in re.split(r"^(?=# %%)", src, flags=re.M) if c.strip()]
nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_markdown_cell(f"# Лабораторна робота {n}")]
nb.cells += [nbformat.v4.new_code_cell(c) for c in chunks]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
t = time.time()
NotebookClient(nb, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": str(folder)}}).execute()
nbformat.write(nb, folder / f"lab{n:02d}.ipynb")
for c in nb.cells:
    if c.cell_type != "code":
        continue
    print("=" * 30, c.source.splitlines()[0])
    for o in c.outputs:
        if o.output_type == "stream":
            print(o.text, end="")
        elif o.output_type == "error":
            print("ERROR", o.ename, o.evalue)
print(f"\nВиконано за {time.time() - t:.1f} с")
