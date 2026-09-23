"""Self-Researcher 图表层。

优先用 matplotlib 输出 PNG；matplotlib 不可用时退化为纯文本 ASCII 图表，
保证在最小依赖环境（例如 CI 容器）里论文依然“有图可看”。
两条路径共享同一份数据，图片只是渲染方式不同。
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hermes.self_research.research_data_collector import ResearchDataset
from hermes.self_research.statistical_analyzer import StatisticalAnalysis

SeriesPoint = Tuple[Any, float]


def _to_index(value: float, low: float, high: float, size: int) -> int:
    """把数值映射到 `[0, size-1]` 的网格坐标。"""

    if size <= 1:
        return 0
    if high - low <= 0:
        return (size - 1) // 2
    ratio = (value - low) / (high - low)
    return max(0, min(size - 1, int(round(ratio * (size - 1)))))


def ascii_line_chart(series: Sequence[SeriesPoint], width: int = 56, height: int = 10, title: str = "") -> str:
    """用字符画一条折线。"""

    if not series:
        return "（无数据）"

    labels = [str(point[0]) for point in series]
    values = [float(point[1]) for point in series]
    low, high = min(values), max(values)

    grid = [[" "] * width for _ in range(height)]
    for index, value in enumerate(values):
        column = _to_index(float(index), 0.0, float(max(1, len(values) - 1)), width)
        row = _to_index(value, low, high, height)
        grid[height - 1 - row][column] = "*"

    lines: List[str] = []
    if title:
        lines.append(f"{title}（min={low:.3f}, max={high:.3f}, n={len(values)}）")

    for row in range(height):
        if height > 1:
            axis_value = high - (high - low) * (row / (height - 1))
        else:
            axis_value = high
        lines.append(f"{axis_value:>7.3f} |" + "".join(grid[row]))

    lines.append(" " * 8 + "+" + "-" * width)
    if len(labels) > 1:
        left, right = labels[0], labels[-1]
        padding = max(1, width - len(left) - len(right))
        lines.append(" " * 9 + left + " " * padding + right)

    return "\n".join(lines)


def ascii_bar_chart(items: Sequence[SeriesPoint], width: int = 36, title: str = "") -> str:
    """用字符画横向条形图。"""

    if not items:
        return "（无数据）"

    values = [float(item[1]) for item in items]
    scale = max(abs(value) for value in values) or 1.0
    name_width = max(len(str(item[0])) for item in items)

    lines: List[str] = []
    if title:
        lines.append(title)

    for name, raw_value in items:
        value = float(raw_value)
        filled = int(round(abs(value) / scale * width))
        bar = ("+" if value >= 0 else "-") * filled
        lines.append(f"{str(name):<{name_width}} | {bar} {value:+.3f}")

    return "\n".join(lines)


def ascii_scatter(
    points: Sequence[SeriesPoint],
    width: int = 48,
    height: int = 12,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "",
) -> str:
    """用字符画散点图。"""

    if not points:
        return "（无数据）"

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    low_x, high_x = min(xs), max(xs)
    low_y, high_y = min(ys), max(ys)

    if high_x - low_x <= 0.0 or high_y - low_y <= 0.0:
        lines = [f"{title}（n={len(points)}）"] if title else []
        for x_value, y_value in zip(xs, ys):
            lines.append(f"  {x_label}={x_value:.3f}，{y_label}={y_value:.3f}")
        lines.append("（取值范围退化为单点，无法绘制二维散点）")
        return "\n".join(lines)

    grid = [[" "] * width for _ in range(height)]
    for x_value, y_value in zip(xs, ys):
        column = _to_index(x_value, low_x, high_x, width)
        row = _to_index(y_value, low_y, high_y, height)
        grid[height - 1 - row][column] = "*"

    lines: List[str] = []
    if title:
        lines.append(f"{title}（n={len(points)}）")

    for row in range(height):
        if height > 1:
            axis_value = high_y - (high_y - low_y) * (row / (height - 1))
        else:
            axis_value = high_y
        lines.append(f"{axis_value:>7.3f} |" + "".join(grid[row]))

    lines.append(" " * 8 + "+" + "-" * width)
    left, right = f"{low_x:.2f}", f"{high_x:.2f}"
    padding = max(1, width - len(left) - len(right))
    lines.append(" " * 9 + left + " " * padding + right)
    if y_label and x_label:
        lines.append(f"（横轴：{x_label}，纵轴：{y_label}）")

    return "\n".join(lines)


@dataclass
class Figure:
    name: str
    kind: str
    caption: str
    ascii_art: str = ""
    path: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "caption": self.caption,
            "path": self.path,
            "ascii_art": self.ascii_art,
            "data": self.data,
        }


class FigureGenerator:
    """生成论文需要的三张图：成功率曲线、阶段对比、相似度-成功率散点。"""

    def __init__(self, output_dir: str = "papers/figures", prefer_png: bool = True):
        self._output_dir = output_dir
        self._prefer_png = prefer_png

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def generate_all(self, dataset: ResearchDataset, analysis: StatisticalAnalysis) -> List[Figure]:
        return [
            self._success_rate_curve(dataset),
            self._phase_comparison(dataset, analysis),
            self._similarity_scatter(dataset),
        ]

    # ------------------------------------------------------------------ 各图

    def _success_rate_curve(self, dataset: ResearchDataset) -> Figure:
        series: List[SeriesPoint] = [
            (snapshot.get("snapshot_index", index + 1), float(snapshot.get("success_rate", 0.0)))
            for index, snapshot in enumerate(dataset.snapshots)
        ]
        caption = "图 1：成功率演化曲线（横轴为快照序号）"
        figure = Figure(
            name="success_rate_curve",
            kind="line",
            caption=caption,
            ascii_art=ascii_line_chart(series, title="成功率演化"),
            data={"series": series},
        )
        self._render_png(
            figure,
            plot=self._plot_line,
            x_values=[point[0] for point in series],
            y_values=[point[1] for point in series],
            x_label="snapshot",
            y_label="success rate",
        )
        return figure

    def _phase_comparison(self, dataset: ResearchDataset, analysis: StatisticalAnalysis) -> Figure:
        phases = dataset.phases
        commit_items: List[SeriesPoint] = [
            (str(phase.get("name")), float(phase.get("total_commits", 0))) for phase in phases
        ]

        comparison = analysis.comparisons[0] if analysis.comparisons else None
        summary = ""
        if comparison is not None:
            summary = (
                f"前后期成功率对比：{comparison.group_a} 均值 {comparison.mean_a:.3f} vs "
                f"{comparison.group_b} 均值 {comparison.mean_b:.3f}"
                f"（p={comparison.p_value:.4f}，{'显著' if comparison.significant else '不显著'}）"
            )

        art = ascii_bar_chart(commit_items, title="各阶段提交数")
        if summary:
            art = f"{art}\n{summary}"

        figure = Figure(
            name="phase_comparison",
            kind="bar",
            caption="图 2：演化阶段对比（各阶段提交量与前/后期成功率对比）",
            ascii_art=art,
            data={
                "phases": [phase.get("name") for phase in phases],
                "commits": [float(phase.get("total_commits", 0)) for phase in phases],
                "comparison": comparison.to_dict() if comparison else None,
            },
        )
        self._render_png(
            figure,
            plot=self._plot_bar,
            x_values=[point[0] for point in commit_items],
            y_values=[point[1] for point in commit_items],
            x_label="phase",
            y_label="commits",
        )
        return figure

    def _similarity_scatter(self, dataset: ResearchDataset) -> Figure:
        points: List[SeriesPoint] = [
            (float(pair.get("similarity", 0.0)), float(pair.get("success_rate", 0.0)))
            for pair in dataset.similarity_pairs
        ]
        correlation = None
        art = ascii_scatter(
            points,
            x_label="相似度",
            y_label="成功率",
            title="相似度-成功率散点",
        )
        figure = Figure(
            name="similarity_success_scatter",
            kind="scatter",
            caption="图 3：知识图谱相似度与实测成功率的散点关系",
            ascii_art=art,
            data={"points": points, "correlation": correlation},
        )
        self._render_png(
            figure,
            plot=self._plot_scatter,
            x_values=[point[0] for point in points],
            y_values=[point[1] for point in points],
            x_label="similarity",
            y_label="success rate",
        )
        return figure

    # ------------------------------------------------------------------ 渲染后端

    def _render_png(self, figure: Figure, plot, x_values, y_values, x_label, y_label) -> None:
        if not self._prefer_png or not x_values:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot(plt, x_values, y_values, x_label, y_label)
            os.makedirs(self._output_dir, exist_ok=True)
            path = os.path.join(self._output_dir, f"{figure.name}.png")
            plt.tight_layout()
            plt.savefig(path, dpi=120)
            plt.close()
            figure.path = path
        except Exception:
            figure.path = None

    @staticmethod
    def _plot_line(plt, x_values, y_values, x_label, y_label) -> None:
        plt.figure(figsize=(6.4, 3.6))
        plt.plot(x_values, y_values, marker="o", linewidth=1.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(alpha=0.3)

    @staticmethod
    def _plot_bar(plt, x_values, y_values, x_label, y_label) -> None:
        plt.figure(figsize=(6.4, 3.6))
        positions = range(len(x_values))
        plt.bar(positions, y_values)
        plt.xticks(list(positions), [str(value) for value in x_values], rotation=15)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(alpha=0.3, axis="y")

    @staticmethod
    def _plot_scatter(plt, x_values, y_values, x_label, y_label) -> None:
        plt.figure(figsize=(6.4, 3.6))
        plt.scatter(x_values, y_values, marker="o")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(alpha=0.3)
