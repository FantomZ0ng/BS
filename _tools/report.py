"""Бібліотека для формування звітів .docx до лабораторних робіт."""
import re, textwrap, unicodedata
from pathlib import Path
import nbformat
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent.parent
FONT = "C:/Windows/Fonts/consola.ttf"
FONT_B = "C:/Windows/Fonts/consolab.ttf"
FS = 26  # розмір шрифту в пікселях (рендер у 2x для чіткості)

# Кольори теми VS Code «Dark+»
BG, GUT, LN, TXT = "#1E1E1E", "#1E1E1E", "#858585", "#D4D4D4"
COLORS = [
    (Token.Comment, "#6A9955"), (Token.Literal.String, "#CE9178"),
    (Token.Literal.Number, "#B5CEA8"), (Token.Keyword.Namespace, "#C586C0"),
    (Token.Keyword.Constant, "#569CD6"), (Token.Keyword, "#C586C0"),
    (Token.Operator.Word, "#569CD6"), (Token.Name.Builtin, "#DCDCAA"),
    (Token.Name.Function, "#DCDCAA"), (Token.Name.Class, "#4EC9B0"),
    (Token.Name.Namespace, "#4EC9B0"), (Token.Name.Decorator, "#DCDCAA"),
    (Token.Name, "#9CDCFE"), (Token.Operator, "#D4D4D4"), (Token.Punctuation, "#D4D4D4"),
]

from fontTools.ttLib import TTFont as _TT
import matplotlib as _mpl
_CMAP = set(_TT(FONT)["cmap"].getBestCmap())
FALLBACK = str(Path(_mpl.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf")


def _draw(d, xy, text, font, fill):
    """Малює текст Consolas, а символи, яких у ньому немає (⇔, ∩, x̄), — DejaVu Sans Mono."""
    x, y = xy
    fb = ImageFont.truetype(FALLBACK, font.size)
    cw = font.getlength("M")
    for ch in text:
        f = font if ord(ch) in _CMAP else fb
        if unicodedata.combining(ch):        # діакритика (риска над x̄) — над попереднім символом
            d.text((x - cw, y), ch, font=f, fill=fill); continue
        d.text((x, y), ch, font=f, fill=fill)
        x += cw
    return x


def _color(tok):
    for t, c in COLORS:
        if tok in t:
            return c
    return TXT


def _font(bold=False, size=FS):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def _wrap_code(code, width=92):
    out = []
    for line in code.splitlines():
        out.append(line)  # код не переносимо — VS Code показує як є
    return "\n".join(out)


def code_image(code, path, filename, start_line=1):
    """Рендер фрагмента коду у вигляді вікна редактора VS Code (Dark+)."""
    code = code.rstrip("\n")
    lines = code.split("\n")
    f = _font(); fb = _font(True)
    cw = f.getbbox("M")[2]; lh = int(FS * 1.45)
    maxlen = max(len(l) for l in lines)
    gutter = cw * 5 + 20
    title_h, tab_h, status_h = 44, 46, 34
    W = max(gutter + cw * (maxlen + 2), 1100)
    H = title_h + tab_h + lh * len(lines) + 30 + status_h
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # заголовок вікна
    d.rectangle([0, 0, W, title_h], fill="#323233")
    for i, c in enumerate(["#FF5F57", "#FEBC2E", "#28C840"]):
        d.ellipse([16 + i * 30, 13, 34 + i * 30, 31], fill=c)
    t = f"{filename} — Вибрані розділи — Visual Studio Code"
    sf = _font(size=22)
    d.text(((W - sf.getlength(t)) / 2, 10), t, font=sf, fill="#CCCCCC")
    # вкладка
    d.rectangle([0, title_h, W, title_h + tab_h], fill="#252526")
    tabw = sf.getlength(filename) + 70
    d.rectangle([0, title_h, tabw, title_h + tab_h], fill=BG)
    d.rectangle([0, title_h, tabw, title_h + 2], fill="#007ACC")
    d.text((14, title_h + 10), "🐍" if False else "py", font=_font(True, 18), fill="#E5C07B")
    d.text((48, title_h + 11), filename, font=sf, fill="#FFFFFF")
    # код
    y0 = title_h + tab_h + 15
    for i, line in enumerate(lines):
        y = y0 + i * lh
        ln = str(start_line + i)
        d.text((gutter - 20 - f.getlength(ln), y), ln, font=f, fill=LN)
    x, row = gutter, 0
    for tok, val in lex(code, PythonLexer()):
        parts = val.split("\n")
        for j, p in enumerate(parts):
            if j > 0:
                row += 1; x = gutter
            if p:
                x = _draw(d, (x, y0 + row * lh), p, f, _color(tok))
    # рядок стану
    d.rectangle([0, H - status_h, W, H], fill="#007ACC")
    d.text((14, H - status_h + 6), f"Python 3.12.16 ('.venv')   Ln {start_line}, Col 1   UTF-8   LF",
           font=_font(size=19), fill="#FFFFFF")
    img.quantize(64).save(path, optimize=True)   # палітра 64 кольори — менший розмір звіту
    return path


def output_image(text, path, title="ВИХІД КОМІРКИ — Jupyter (VS Code)", width=140):
    """Рендер текстового виводу у вигляді панелі виводу/терміналу VS Code."""
    raw = text.rstrip("\n").split("\n")
    lines = []
    for l in raw:
        l = l.expandtabs(4)
        lines += textwrap.wrap(l, width, drop_whitespace=False, replace_whitespace=False) or [""]
    f = _font(size=24); cw = f.getbbox("M")[2]; lh = int(24 * 1.35)
    head = 44
    W = max(int(cw * (max(len(l) for l in lines) + 4)), 1100)
    H = head + lh * len(lines) + 30
    img = Image.new("RGB", (W, H), "#181818")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, head], fill="#252526")
    d.text((16, 10), title, font=_font(True, 20), fill="#E7E7E7")
    d.rectangle([16, head - 3, 16 + _font(True, 20).getlength(title), head], fill="#007ACC")
    for i, l in enumerate(lines):
        col = "#F48771" if re.match(r"^\s*(\w*Error|Traceback|ERROR|STOP)", l) else "#CCCCCC"
        _draw(d, (20, head + 12 + i * lh), l, f, col)
    img.quantize(64).save(path, optimize=True)   # палітра 64 кольори — менший розмір звіту
    return path


class Report:
    def __init__(self, n, topic):
        self.n, self.topic = n, topic
        self.dir = ROOT / str(n)
        import os
        self.img = Path(os.environ.get("REPORT_IMG_DIR", self.dir / "report_img")) / str(n) if os.environ.get("REPORT_IMG_DIR") else self.dir / "report_img"
        self.img.mkdir(parents=True, exist_ok=True)
        nb = nbformat.read(self.dir / f"lab{n:02d}.ipynb", as_version=4)
        self.cells = {}
        start = 1
        for c in nb.cells:
            if c.cell_type != "code":
                continue
            m = re.match(r"# %% \[(.+?)\]", c.source)
            key = m.group(1) if m else str(len(self.cells))
            out = ""
            for o in c.outputs:
                if o.output_type == "stream":
                    out += o.text
                elif o.output_type == "execute_result" and "text/plain" in o.data:
                    out += o.data["text/plain"] + "\n"
                elif o.output_type == "error":
                    out += f"{o.ename}: {o.evalue}\n"
            self.cells[key] = (c.source, out, start)
            start += len(c.source.split("\n")) + 1
        self.fig_no = 0; self.tab_no = 0
        self.doc = Document()
        self._styles()
        self._title_page()

    # ---------- оформлення ----------
    def _styles(self):
        s = self.doc.sections[0]
        s.page_height, s.page_width = Cm(29.7), Cm(21)
        s.left_margin, s.right_margin = Cm(2.5), Cm(1.5)
        s.top_margin, s.bottom_margin = Cm(2), Cm(2)
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"; st.font.size = Pt(14)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        pf = st.paragraph_format
        pf.line_spacing = 1.5; pf.space_after = Pt(0); pf.space_before = Pt(0)
        for name, size in [("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 14)]:
            h = self.doc.styles[name]
            h.font.name = "Times New Roman"; h.font.size = Pt(size); h.font.bold = True
            h.font.color.rgb = RGBColor(0, 0, 0)
            h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
            rf = h.element.rPr.rFonts
            rf.set(qn("w:ascii"), "Times New Roman"); rf.set(qn("w:hAnsi"), "Times New Roman")
            h.paragraph_format.space_before = Pt(12); h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.keep_with_next = True
            h.paragraph_format.line_spacing = 1.15
        # нумерація сторінок у нижньому колонтитулі (крім титулки)
        s.different_first_page_header_footer = True
        p = s.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run()
        for tag, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
            if tag:
                e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag); r._r.append(e)
            else:
                e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
                r._r.append(e)

    def _cp(self, text, bold=False, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, after=0,
            italic=False, caps=False):
        p = self.doc.add_paragraph(); p.alignment = align
        p.paragraph_format.space_after = Pt(after); p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.first_line_indent = Cm(0)
        r = p.add_run(text); r.bold = bold; r.italic = italic; r.font.size = Pt(size)
        r.font.all_caps = caps
        return p

    def _title_page(self):
        C = WD_ALIGN_PARAGRAPH.CENTER
        self._cp("МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ", True)
        self._cp("ЛЬВІВСЬКИЙ НАЦІОНАЛЬНИЙ УНІВЕРСИТЕТ ІМЕНІ ІВАНА ФРАНКА", True)
        self._cp("Факультет Електроніки та комп'ютерних технологій")
        self._cp("Кафедра системного проектування", after=110)
        self._cp("ЗВІТ", True, 20, after=6)
        self._cp(f"про виконання лабораторної роботи № {self.n}")
        self._cp("з дисципліни \"Вибрані розділи науки про дані\"")
        self._cp(f"на тему: «{self.topic}»", True, after=120)
        R = WD_ALIGN_PARAGRAPH.LEFT
        for t, b in [("Виконав:", True), ("студент 1 курсу, групи Феі-12с", False),
                     ("Сваричевський Андрій Васильович", False), ("", False),
                     ("Перевірив:", True), ("професор Ляшкевич В. Я.", False)]:
            p = self._cp(t, b, align=R); p.paragraph_format.left_indent = Cm(9.5)
        sp = self._cp("", after=0)
        sp.paragraph_format.space_before = Pt(120)
        self._cp("Львів — 2026")
        self.page_break()

    # ---------- вміст ----------
    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def h1(self, t):
        self.doc.add_heading(t, level=1)

    def h2(self, t):
        self.doc.add_heading(t, level=2)

    def p(self, text, indent=True):
        """Абзац; **жирний** та `код` підтримуються."""
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if indent:
            para.paragraph_format.first_line_indent = Cm(1.25)
        for part in re.split(r"(\*\*.+?\*\*|`.+?`)", text):
            if part.startswith("**"):
                para.add_run(part[2:-2]).bold = True
            elif part.startswith("`"):
                r = para.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(12)
            elif part:
                para.add_run(part)
        return para

    def bullets(self, items):
        for it in items:
            para = self.p("– " + it, indent=False)
            para.paragraph_format.left_indent = Cm(1.25)

    def formula(self, t):
        para = self.doc.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = para.add_run(t); r.italic = True

    def _picture(self, path, caption, width=16.5):
        para = self.doc.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.first_line_indent = Cm(0)
        para.paragraph_format.keep_with_next = True
        para.paragraph_format.line_spacing = 1.0
        w, h = Image.open(path).size
        width = min(width, 16.5)
        if h / w * width > 22:
            width = 22 * w / h
        para.add_run().add_picture(str(path), width=Cm(width))
        self.fig_no += 1
        cp = self.doc.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(6)
        r = cp.add_run(f"Рисунок {self.n}.{self.fig_no} — {caption}"); r.font.size = Pt(12)

    def code(self, key, caption=None, lines=None):
        src, out, start = self.cells[key]
        if lines:
            ls = src.split("\n"); src = "\n".join(ls[lines[0] - 1:lines[1]]); start += lines[0] - 1
        path = self.img / f"code_{re.sub(r'[^0-9A-Za-z]+', '_', key.replace('Р', 'R'))}_{start}.png"
        code_image(src, path, f"lab{self.n:02d}.py", start)
        self._picture(path, caption or f"Код комірки [{key}] у VS Code")

    def output(self, key, caption=None, max_lines=60, select=None):
        out = self.cells[key][1]
        if select:
            out = select(out)
        ls = out.rstrip("\n").split("\n")
        if len(ls) > max_lines:
            ls = ls[:max_lines] + ["..."]
        path = self.img / f"out_{re.sub(r'[^0-9A-Za-z]+', '_', key.replace('Р', 'R'))}.png"
        output_image("\n".join(ls), path)
        self._picture(path, caption or f"Результат виконання комірки [{key}]")

    def fig(self, name, caption, width=15):
        self._picture(self.dir / "figures" / name, caption, width)

    def table(self, header, rows, caption, widths=None, size=11):
        self.tab_no += 1
        cp = self.doc.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        cp.paragraph_format.keep_with_next = True
        r = cp.add_run(f"Таблиця {self.n}.{self.tab_no} — {caption}"); r.font.size = Pt(12)
        t = self.doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]; c.text = ""
            run = c.paragraphs[0].add_run(str(h)); run.bold = True; run.font.size = Pt(size)
            sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "D9E2F3")
            c._tc.get_or_add_tcPr().append(sh)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(v)); run.font.size = Pt(size)
        for row in t.rows:
            for i, c in enumerate(row.cells):
                for para in c.paragraphs:
                    para.paragraph_format.line_spacing = 1.0
                    para.paragraph_format.first_line_indent = Cm(0)
                if widths:
                    c.width = Cm(widths[i])
        self.doc.add_paragraph().paragraph_format.space_after = Pt(0)

    def save(self):
        import os
        out_dir = Path(os.environ["REPORT_OUT_DIR"]) / str(self.n) if os.environ.get("REPORT_OUT_DIR") else self.dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"ЛР{self.n:02d}_Сваричевський_Феі-12с.docx"
        self.doc.save(out)
        print("Збережено:", out, "| рисунків:", self.fig_no, "| таблиць:", self.tab_no)
        return out
