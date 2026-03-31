from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape


def build_html_report(
    analysis: Dict[str, Any],
    templates_dir: str | Path | None = None,
) -> str:
    """Build a simple HTML report using Jinja2.

    If no templates exist, falls back to an inline template.
    """
    if templates_dir is None:
        templates_dir = Path(__file__).parent / "templates"
    templates_dir = Path(templates_dir)

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )

    inline_template = """
<!doctype html>
<html>
  <head><meta charset="utf-8"/><title>AutoAnalyst Report</title></head>
  <body>
    <h1>AutoAnalyst Report</h1>
    <h2>Overview</h2>
    <pre>{{ overview | tojson(indent=2) }}</pre>
  </body>
</html>
""".strip()

    template_name = "report.html"
    try:
        template = env.get_template(template_name)
    except Exception:
        template = env.from_string(inline_template)

    # Ensure HTML contains only JSON-serializable content.
    overview = json.loads(json.dumps(analysis, default=str))
    return template.render(overview=overview)

