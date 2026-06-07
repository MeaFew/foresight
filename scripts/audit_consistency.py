"""Cross-reference audit: README claims vs. actual pipeline outputs.

Run after `make all` to verify that key metrics declared in README.md
match the actual values produced by the pipeline.

Usage: python scripts/audit_consistency.py

Add project-specific checks in the `main()` function.
"""

import json
import re
import sys
from pathlib import Path


def read_readme_metric(readme_path: Path, metric_name: str) -> float | None:
    """Extract a numeric metric from README.md.

    Looks for patterns like `| **Metric Name** | **0.123** |`
    or `**Metric Name = 0.123**`.
    """
    text = readme_path.read_text(encoding="utf-8")
    pattern = rf"\*\*{re.escape(metric_name)}\*\*.*?(\d+\.\d+)"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def check(condition: bool, msg: str) -> bool:
    """Assert-like check that prints pass/fail."""
    if condition:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
    return condition


def main():
    root = Path(__file__).resolve().parents[1]
    readme = root / "README.md"
    passed = 0
    failed = 0

    # --- Project-specific checks go here ---
    # Example:
    # r2_readme = read_readme_metric(readme, "R²")
    # if r2_readme is not None:
    #     with open(root / "reports" / "model_results.json") as f:
    #         actual = json.load(f)
    #     r2_actual = actual["models"]["ridge"]["r2"]
    #     ok = check(abs(r2_readme - r2_actual) < 0.01,
    #                f"R²: README={r2_readme:.3f}, actual={r2_actual:.3f}")
    #     if ok: passed += 1 else: failed += 1

    # --- Summary ---
    total = passed + failed
    if total == 0:
        print("No checks configured. Add project-specific checks to main().")
        return

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        print("ACTION: Update README.md or pipeline to resolve mismatches.")
        sys.exit(1)


if __name__ == "__main__":
    main()
