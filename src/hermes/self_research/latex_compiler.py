"""Self-Researcher 组装层：把论文对象渲染成 LaTeX / Markdown，并在可用时编译 PDF。

模板使用 `<< name >>` 占位符。装有 Jinja2 时走 Jinja2 渲染（自定义定界符，
避免与 LaTeX 的花括号冲突）；没有 Jinja2 时走等价的占位符替换，
两种路径输出的 LaTeX 完全一致。
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes.self_research.paper_writer import Paper

DEFAULT_LATEX_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=1in}
\title{<< title >>}
\author{<< authors >>}
\date{<< date >>}
\begin{document}
\maketitle
\begin{abstract}
<< abstract >>
\end{abstract}

\noindent\textbf{关键词：}<< keywords >>

<< body >>

\section*{参考文献}
\begin{enumerate}
<< references >>
\end{enumerate}
\end{document}
"""

LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

PLACEHOLDER_RE = re.compile(r"<<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>>")


def latex_escape(text: str) -> str:
    """转义 LaTeX 特殊字符。"""

    return "".join(LATEX_ESCAPES.get(character, character) for character in str(text))


@dataclass
class CompileResult:
    output_path: str
    format: str
    compiled: bool = False
    engine: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_path": self.output_path,
            "format": self.format,
            "compiled": self.compiled,
            "engine": self.engine,
            "warnings": self.warnings,
        }


class LatexCompiler:
    """渲染并（可选）编译论文。"""

    def __init__(self, output_dir: str = "papers", template_path: Optional[str] = None):
        self._output_dir = output_dir
        self._template_path = template_path

    # ------------------------------------------------------------------ 渲染入口

    def render(self, paper: Paper, output_path: Optional[str] = None, fmt: str = "auto") -> CompileResult:
        resolved = self.resolve_format(output_path, fmt)

        if resolved == "markdown":
            path = output_path or os.path.join(self._output_dir, "paper.md")
            self._write(path, self.to_markdown(paper))
            return CompileResult(output_path=path, format="markdown")

        if resolved == "pdf":
            pdf_path = output_path or os.path.join(self._output_dir, "paper.pdf")
            tex_path = os.path.splitext(pdf_path)[0] + ".tex"
        else:
            tex_path = output_path or os.path.join(self._output_dir, "paper.tex")
            pdf_path = ""

        self._write(tex_path, self.to_latex(paper))
        result = CompileResult(output_path=tex_path, format="latex")

        if resolved == "pdf":
            compile_result = self.compile_pdf(tex_path)
            result.warnings.extend(compile_result.warnings)
            result.engine = compile_result.engine
            result.compiled = compile_result.compiled
            result.format = "pdf"
            result.output_path = compile_result.output_path if compile_result.compiled else tex_path

        return result

    @staticmethod
    def resolve_format(output_path: Optional[str], fmt: str) -> str:
        if fmt in ("markdown", "latex", "pdf"):
            return fmt
        if output_path:
            extension = os.path.splitext(output_path)[1].lower()
            if extension in (".md", ".markdown"):
                return "markdown"
            if extension == ".tex":
                return "latex"
            if extension == ".pdf":
                return "pdf"
        return "markdown"

    # ------------------------------------------------------------------ Markdown

    def to_markdown(self, paper: Paper) -> str:
        lines = [f"# {paper.title}", "", f"**作者**：{paper.authors}", "", "## 摘要", "", paper.abstract, ""]
        if paper.keywords:
            lines.extend([f"**关键词**：{'、'.join(paper.keywords)}", ""])

        for section in paper.sections:
            lines.extend([f"## {section.heading}", "", section.content, ""])

        if paper.figures:
            lines.extend(["## 图表", ""])
            for figure in paper.figures:
                lines.extend([f"**{figure.get('caption', figure.get('name'))}**", ""])
                if figure.get("path"):
                    lines.extend([f"（图片文件：{figure['path']}）", ""])
                art = figure.get("ascii_art") or "（无数据）"
                lines.extend(["```text", art, "```", ""])

        if paper.references:
            lines.extend(["## 参考文献", ""])
            for index, reference in enumerate(paper.references, 1):
                lines.append(
                    f"[{index}] {reference.get('authors', '')}. "
                    f"{reference.get('title', '')}. {reference.get('venue', '')}, {reference.get('year', '')}."
                )

        return "\n".join(lines).rstrip() + "\n"

    # ------------------------------------------------------------------ LaTeX

    def to_latex(self, paper: Paper) -> str:
        template = self._load_template()
        variables = {
            "title": latex_escape(paper.title),
            "authors": latex_escape(paper.authors),
            "date": latex_escape(paper.meta.get("generated_at", "")[:10]),
            "abstract": latex_escape(paper.abstract),
            "keywords": latex_escape("、".join(paper.keywords)),
            "body": self._body_to_latex(paper),
            "references": self._references_to_latex(paper),
        }
        return self._render_template(template, variables)

    def _body_to_latex(self, paper: Paper) -> str:
        blocks: List[str] = []

        for section in paper.sections:
            if section.kind == "appendix":
                blocks.append(f"\\section*{{{latex_escape(section.heading)}}}")
                blocks.append("\\addcontentsline{toc}{section}{" + latex_escape(section.heading) + "}")
            else:
                blocks.append(f"\\section{{{latex_escape(section.heading)}}}")
            blocks.append(self._content_to_latex(section.content))

        for figure in paper.figures:
            blocks.append(self._figure_to_latex(figure))

        return "\n\n".join(block for block in blocks if block.strip())

    def _content_to_latex(self, content: str) -> str:
        output: List[str] = []
        in_items = False
        in_verbatim = False

        for raw_line in content.split("\n"):
            line = raw_line.strip()

            if line.startswith("```"):
                if in_items:
                    output.append("\\end{itemize}")
                    in_items = False
                output.append("\\end{verbatim}" if in_verbatim else "\\begin{verbatim}")
                in_verbatim = not in_verbatim
                continue

            if in_verbatim:
                output.append(raw_line)
                continue

            if line.startswith("### ") or line.startswith("## "):
                if in_items:
                    output.append("\\end{itemize}")
                    in_items = False
                output.append(f"\\subsection{{{latex_escape(line.lstrip('#').strip())}}}")
            elif line.startswith("- "):
                if not in_items:
                    output.append("\\begin{itemize}")
                    in_items = True
                output.append(f"\\item {latex_escape(line[2:])}")
            elif not line:
                if in_items:
                    output.append("\\end{itemize}")
                    in_items = False
                output.append("")
            else:
                if in_items:
                    output.append("\\end{itemize}")
                    in_items = False
                output.append(latex_escape(line))

        if in_items:
            output.append("\\end{itemize}")
        if in_verbatim:
            output.append("\\end{verbatim}")

        return "\n".join(output)

    @staticmethod
    def _figure_to_latex(figure: Dict[str, Any]) -> str:
        caption = latex_escape(str(figure.get("caption") or figure.get("name") or "figure"))
        path = figure.get("path")
        raster = str(path or "").lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".eps"))
        lines = ["\\begin{figure}[htbp]", "\\centering"]
        if path and raster:
            lines.append("\\includegraphics[width=0.8\\linewidth]{" + str(path).replace("\\", "/") + "}")
        else:
            art = (figure.get("ascii_art") or "").rstrip()
            lines.extend(["\\begin{verbatim}", art, "\\end{verbatim}"])
        lines.append(f"\\caption{{{caption}}}")
        lines.append("\\end{figure}")
        return "\n".join(lines)

    @staticmethod
    def _references_to_latex(paper: Paper) -> str:
        items = []
        for reference in paper.references:
            items.append(
                "\\item "
                + latex_escape(
                    f"{reference.get('authors', '')}. {reference.get('title', '')}. "
                    f"{reference.get('venue', '')}, {reference.get('year', '')}."
                )
            )
        return "\n".join(items)

    def _load_template(self) -> str:
        if self._template_path and os.path.exists(self._template_path):
            with open(self._template_path, "r", encoding="utf-8") as handle:
                return handle.read()
        return DEFAULT_LATEX_TEMPLATE

    @staticmethod
    def _render_template(template: str, variables: Dict[str, str]) -> str:
        try:
            from jinja2 import Environment
        except Exception:
            return LatexCompiler._render_placeholders(template, variables)

        environment = Environment(
            variable_start_string="<<",
            variable_end_string=">>",
            block_start_string="<%",
            block_end_string="%>",
            comment_start_string="<#",
            comment_end_string="#>",
            autoescape=False,
            keep_trailing_newline=True,
        )
        return environment.from_string(template).render(**variables)

    @staticmethod
    def _render_placeholders(template: str, variables: Dict[str, str]) -> str:
        return PLACEHOLDER_RE.sub(lambda match: variables.get(match.group(1), ""), template)

    # ------------------------------------------------------------------ PDF

    def compile_pdf(self, tex_path: str) -> CompileResult:
        pdf_path = os.path.splitext(tex_path)[0] + ".pdf"
        result = CompileResult(output_path=pdf_path, format="pdf")

        engine = shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("tectonic")
        if not engine:
            result.warnings.append("未找到 pdflatex/xelatex/tectonic，已保留 .tex 源文件")
            return result

        result.engine = os.path.basename(engine)
        directory = os.path.dirname(os.path.abspath(tex_path)) or "."
        command = [engine, "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex_path)]

        try:
            for _ in range(2):  # 编译两次以稳定交叉引用
                process = subprocess.run(
                    command,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=180,
                )
                if process.returncode != 0:
                    result.warnings.append((process.stdout or process.stderr or "")[-800:])
                    return result
        except (OSError, subprocess.SubprocessError) as exc:
            result.warnings.append(f"{type(exc).__name__}: {exc}")
            return result

        result.compiled = os.path.exists(pdf_path)
        if not result.compiled:
            result.warnings.append("编译命令执行成功，但未生成 PDF")
        return result

    # ------------------------------------------------------------------ 工具

    @staticmethod
    def _write(path: str, content: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
