"""Tests for README-to-artifact consistency helpers."""

from pathlib import Path

from foresight.audit_consistency import read_readme_metric


def test_read_readme_metric_accepts_bold_markdown(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "| model | MAE |\n|---|---|\n| **XGBoost** | **0.257** |\n",
        encoding="utf-8",
    )

    assert read_readme_metric(readme, "XGBoost") == 0.257


def test_read_readme_metric_ignores_unrelated_numbers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "Version 2.0\n| Other | 9.999 |\n| XGBoost | 0.257 |\n",
        encoding="utf-8",
    )

    assert read_readme_metric(readme, "XGBoost") == 0.257
