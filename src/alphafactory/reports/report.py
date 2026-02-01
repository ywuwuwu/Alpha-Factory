from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_equity_curve_plot(returns: pd.Series, out_path: Path, title: str):
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    plt.figure()
    plt.plot(eq.index, eq.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_report_md(
    out_dir: Path,
    factor_summary: pd.DataFrame,
    perf_summary: pd.DataFrame,
    plots: dict[str, str],
):
    lines = []
    lines.append("# AlphaFactory-Pro Report\n")
    lines.append("## Factor IC Summary\n")
    lines.append(factor_summary.to_markdown(index=False))
    lines.append("\n\n## Portfolio Performance Summary\n")
    lines.append(perf_summary.to_markdown(index=False))
    lines.append("\n\n## Plots\n")
    for k, rel in plots.items():
        lines.append(f"- **{k}**: ![]({rel})")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
