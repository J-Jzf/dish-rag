"""保留页边界的 PDF 文本抽取。"""

from __future__ import annotations

import subprocess
from pathlib import Path


def extract_pages(pdf_path: Path) -> list[str]:
    """从 PDF 中抽取文本，每页对应一个字符串。

    优先使用 `pdfplumber`，因为它能在内存里天然保留页码。如果学习环境
    中没有该库，则回退到 Poppler 的 `pdftotext`，再按换页符切分页。
    """

    try:
        import pdfplumber

        pages: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                # `x_tolerance` 可以降低中英文混排文本被切成零散片段的概率。
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                pages.append(text)
        return pages # 返回的是一个字符串列表，pages[0] = PDF 第 1 页抽取出来的所有文字
    except Exception:
        return _extract_pages_with_pdftotext(pdf_path)


def _extract_pages_with_pdftotext(pdf_path: Path) -> list[str]:
    """使用 `pdftotext -layout` 的兜底抽取方式。"""

    output_path = pdf_path.with_suffix(".pdftotext.txt")
    command = ["pdftotext", "-layout", str(pdf_path), str(output_path)]
    subprocess.run(command, check=True)
    raw_text = output_path.read_text(encoding="utf-8")
    return raw_text.split("\f")


def write_page_markdown(pages: list[str], output_path: Path) -> None:
    """写出保留页码标记的 Markdown，方便人工检查。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# 菜谱 PDF 文本抽取稿", ""]
    for page_no, page_text in enumerate(pages, start=1):
        lines.append(f"<!-- page:{page_no} -->")
        lines.append(f"## Page {page_no}")
        lines.append("")
        lines.append(page_text.strip())
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
