from __future__ import annotations

import argparse
import html
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse

try:
    import pandas as pd
except Exception:
    pd = None


WORKDIR = Path(__file__).parent
DEFAULT_PLOTS_DIR = WORKDIR / "fixation_plots"
PROJECT_ROOT = WORKDIR.parent if (WORKDIR.parent / "experiment-ANNs").exists() else WORKDIR


def _rel_url(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    return "/" + quote(rel, safe="/-_.")


def build_manifest(plots_dir: Path, root: Path) -> dict:
    plots_dir = plots_dir.resolve()
    root = root.resolve()
    manifest: dict[str, dict] = {"plots_dir": str(plots_dir), "analyses": {}}

    if not plots_dir.exists():
        return manifest

    for analysis_dir in sorted([p for p in plots_dir.iterdir() if p.is_dir()]):
        analysis_name = analysis_dir.name
        manifest["analyses"][analysis_name] = {}

        for grouping_dir in sorted([p for p in analysis_dir.iterdir() if p.is_dir()]):
            grouping_name = grouping_dir.name
            plots_path = grouping_dir / "plots"
            matrices_path = grouping_dir / "matrices"
            items = []

            if plots_path.exists():
                plot_files = sorted(plots_path.glob("heatmap_*.json"))
                if not plot_files:
                    plot_files = sorted(plots_path.glob("heatmap_*.html"))
                for plot_file in plot_files:
                    slug = plot_file.stem.removeprefix("heatmap_")
                    matrix = matrices_path / f"matrix_{slug}.parquet"
                    if not matrix.exists():
                        matrix = matrices_path / f"matrix_{slug}.csv"
                    items.append(
                        {
                            "slug": slug,
                            "plot_url": _rel_url(plot_file, root),
                            "matrix_url": _rel_url(matrix, root) if matrix.exists() else None,
                        }
                    )

            combined = matrices_path / "all_matrices.parquet"
            if not combined.exists():
                combined = matrices_path / "all_matrices.csv"
            manifest["analyses"][analysis_name][grouping_name] = {
                "combined_matrix_url": (
                    _rel_url(combined, root) if combined.exists() else None
                ),
                "items": items,
            }

    return manifest


def _resolve_existing_file(
    directory: Path, stems: tuple[str, ...], suffixes: tuple[str, ...]
) -> Path | None:
    for stem in stems:
        for suffix in suffixes:
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def build_correlation_manifest(plots_dir: Path, root: Path) -> dict:
    plots_dir = plots_dir.resolve()
    root = root.resolve()
    correlations_dir = plots_dir / "correlations"
    manifest: dict[str, dict] = {
        "correlations_dir": str(correlations_dir),
        "analyses": {},
    }

    if not correlations_dir.exists():
        return manifest

    for analysis_dir in sorted([p for p in correlations_dir.iterdir() if p.is_dir()]):
        analysis_name = analysis_dir.name
        manifest["analyses"][analysis_name] = {}

        direct_plot = _resolve_existing_file(
            analysis_dir,
            stems=("correlation_heatmap", "heatmap"),
            suffixes=(".json", ".html"),
        )
        if direct_plot is not None:
            matrix_file = _resolve_existing_file(
                analysis_dir,
                stems=("correlation_matrix",),
                suffixes=(".parquet", ".csv"),
            )
            pairwise_file = _resolve_existing_file(
                analysis_dir,
                stems=("pairwise_correlations",),
                suffixes=(".parquet", ".csv"),
            )
            manifest["analyses"][analysis_name]["summary"] = {
                "items": [
                    {
                        "slug": analysis_name,
                        "plot_url": _rel_url(direct_plot, root),
                        "matrix_url": _rel_url(matrix_file, root)
                        if matrix_file is not None
                        else None,
                        "pairwise_url": _rel_url(pairwise_file, root)
                        if pairwise_file is not None
                        else None,
                    }
                ]
            }

        for group_dir in sorted([p for p in analysis_dir.iterdir() if p.is_dir()]):
            group_name = group_dir.name
            items = []

            group_plot = _resolve_existing_file(
                group_dir,
                stems=("correlation_heatmap", "heatmap"),
                suffixes=(".json", ".html"),
            )
            if group_plot is not None:
                matrix_file = _resolve_existing_file(
                    group_dir,
                    stems=("correlation_matrix",),
                    suffixes=(".parquet", ".csv"),
                )
                pairwise_file = _resolve_existing_file(
                    group_dir,
                    stems=("pairwise_correlations",),
                    suffixes=(".parquet", ".csv"),
                )
                items.append(
                    {
                        "slug": group_name,
                        "plot_url": _rel_url(group_plot, root),
                        "matrix_url": _rel_url(matrix_file, root)
                        if matrix_file is not None
                        else None,
                        "pairwise_url": _rel_url(pairwise_file, root)
                        if pairwise_file is not None
                        else None,
                    }
                )

            for item_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()]):
                item_plot = _resolve_existing_file(
                    item_dir,
                    stems=("correlation_heatmap", "heatmap"),
                    suffixes=(".json", ".html"),
                )
                if item_plot is None:
                    continue

                matrix_file = _resolve_existing_file(
                    item_dir,
                    stems=("correlation_matrix",),
                    suffixes=(".parquet", ".csv"),
                )
                pairwise_file = _resolve_existing_file(
                    item_dir,
                    stems=("pairwise_correlations",),
                    suffixes=(".parquet", ".csv"),
                )
                items.append(
                    {
                        "slug": item_dir.name,
                        "plot_url": _rel_url(item_plot, root),
                        "matrix_url": _rel_url(matrix_file, root)
                        if matrix_file is not None
                        else None,
                        "pairwise_url": _rel_url(pairwise_file, root)
                        if pairwise_file is not None
                        else None,
                    }
                )

            if items:
                manifest["analyses"][analysis_name][group_name] = {"items": items}

    return manifest


def _heatmap_count(plots_dir: Path) -> int:
    if not plots_dir.exists():
        return -1
    try:
        json_count = sum(1 for _ in plots_dir.rglob("heatmap_*.json"))
        if json_count > 0:
            return json_count
        return sum(1 for _ in plots_dir.rglob("heatmap_*.html"))
    except Exception:
        return -1


def _candidate_plots_dirs(user_dir: Path | None, root_dir: Path) -> list[Path]:
    candidates = []
    if user_dir is not None:
        candidates.append(user_dir)
    candidates.append(DEFAULT_PLOTS_DIR)
    candidates.append(Path.cwd() / "fixation_plots")
    candidates.append(root_dir / "fixation_plots")
    candidates.append(root_dir / "analysis/fixation_plots")

    resolved = []
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        resolved.append(rp)
    return resolved


def resolve_plots_dir(user_dir: Path | None, root_dir: Path) -> tuple[Path, list[tuple[Path, int]]]:
    candidates = _candidate_plots_dirs(user_dir, root_dir)
    scored = [(p, _heatmap_count(p)) for p in candidates]
    best = max(scored, key=lambda x: x[1])[0]
    return best, scored


def _matrix_preview_table(path: Path | None, max_rows: int = 300) -> str:
    if path is None or not path.exists():
        return "<p class='muted'>No matrix file found for this item.</p>"
    try:
        if path.suffix == ".parquet":
            if pd is None:
                return "<p class='muted'>Parquet preview requires pandas. Open the matrix file link instead.</p>"
            df = pd.read_parquet(path)
        else:
            if pd is None:
                return "<p class='muted'>CSV preview requires pandas. Open the matrix file link instead.</p>"
            df = pd.read_csv(path)
    except Exception as exc:
        return f"<p class='muted'>Failed to read matrix file: {html.escape(str(exc))}</p>"
    if df.empty and len(df.columns) == 0:
        return "<p class='muted'>Matrix file is empty.</p>"

    df = df.head(max_rows)
    rows = [df.columns.tolist()] + df.astype(str).values.tolist()

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in rows[0])
    body_rows = []
    for row in rows[1:]:
        body_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        )
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"


def _option_html(options: list[str], selected: str | None) -> str:
    if not options:
        return "<option value=''> (none) </option>"
    out = []
    for opt in options:
        sel = " selected" if opt == selected else ""
        out.append(
            f"<option value='{html.escape(opt, quote=True)}'{sel}>{html.escape(opt)}</option>"
        )
    return "".join(out)


def _render_page(manifest: dict, root_dir: Path, query: dict[str, list[str]]) -> str:
    analyses = sorted(manifest.get("analyses", {}).keys())
    analysis = query.get("analysis", [None])[0]
    if analysis not in analyses:
        analysis = analyses[0] if analyses else None

    groups = (
        sorted(manifest["analyses"].get(analysis, {}).keys())
        if analysis is not None
        else []
    )
    group = query.get("group", [None])[0]
    if group not in groups:
        group = groups[0] if groups else None

    items = (
        manifest["analyses"][analysis][group].get("items", [])
        if analysis is not None and group is not None
        else []
    )
    slugs = [item["slug"] for item in items]
    item_slug = query.get("item", [None])[0]
    if item_slug not in slugs:
        item_slug = slugs[0] if slugs else None

    selected_item = next((item for item in items if item["slug"] == item_slug), None)
    combined_url = (
        manifest["analyses"][analysis][group].get("combined_matrix_url")
        if analysis is not None and group is not None
        else None
    )
    item_plot_url = selected_item.get("plot_url") if selected_item else None
    plot_embed_url = (
        f"/plot?{urlencode({'url': item_plot_url})}" if item_plot_url else "about:blank"
    )
    item_matrix_url = selected_item.get("matrix_url") if selected_item else None

    matrix_path = (
        root_dir / item_matrix_url.lstrip("/") if item_matrix_url else None
    )
    matrix_table = _matrix_preview_table(matrix_path)

    status_text = f"Analyses discovered: {len(analyses)}"
    if analysis and group:
        status_text += f" | Items in selection: {len(items)}"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fixation Plots Viewer</title>
  <style>
    :root {{
      --bg: #f4f5f7;
      --panel: #ffffff;
      --text: #122030;
      --muted: #59697b;
      --line: #d6dde6;
      --accent: #0b6a89;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #eef2f6 0%, #f7f9fb 60%, #f4f5f7 100%);
    }}
    .shell {{
      width: min(1400px, calc(100vw - 24px));
      margin: 12px auto;
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: 0 4px 16px rgba(10, 20, 30, 0.05);
    }}
    .controls {{ padding: 14px; position: sticky; top: 12px; height: fit-content; min-width: 0; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    .muted {{ margin: 0 0 10px; color: var(--muted); font-size: 13px; }}
    .nav {{
      margin: 8px 0 12px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .nav a {{
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #eef4f7;
      text-decoration: none;
      color: var(--text);
      font-size: 12px;
    }}
    .nav a.active {{
      background: #d8ebf3;
      border-color: #bdd9e6;
      font-weight: 600;
    }}
    label {{
      display: block; margin: 10px 0 6px; font-size: 12px; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.06em;
    }}
    select {{
      width: 100%; padding: 8px; border: 1px solid var(--line); border-radius: 7px;
      background: #fff; color: var(--text); font-size: 14px;
    }}
    button {{
      margin-top: 10px; width: 100%; padding: 8px; border: 1px solid var(--line);
      border-radius: 7px; background: #edf3f7; cursor: pointer;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .right {{ display: grid; gap: 14px; min-width: 0; }}
    iframe {{
      display: block; width: 100%; min-width: 0; min-height: 600px; border: 1px solid var(--line);
      border-radius: 8px; background: #fff;
    }}
    .pane {{ padding: 12px; min-width: 0; }}
    .table-wrap {{
      max-height: 420px; overflow: auto; width: 100%; min-width: 0;
      border: 1px solid var(--line); border-radius: 8px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; background: #fff; }}
    th, td {{
      border: 1px solid var(--line); padding: 6px 8px; text-align: center;
      white-space: nowrap;
    }}
    thead th {{ background: #eef4f7; position: sticky; top: 0; z-index: 1; }}
    @media (max-width: 1100px) {{
      .shell {{ grid-template-columns: 1fr; }}
      .controls {{ position: static; }}
      iframe {{ min-height: 480px; }}
    }}
    @media (max-width: 640px) {{
      iframe {{ min-height: 360px; }}
      .card {{ border-radius: 8px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="card controls">
      <h1>Fixation Plots Viewer</h1>
      <div class="nav">
        <a href="/" class="active">Heatmaps</a>
        <a href="/correlations">Correlations</a>
      </div>
      <p class="muted">Data folder: {html.escape(manifest.get("plots_dir", ""))}</p>
      <p class="muted">{html.escape(status_text)}</p>
      <form method="get">
        <label for="analysis">Analysis</label>
        <select id="analysis" name="analysis" onchange="this.form.submit()">
          {_option_html(analyses, analysis)}
        </select>
        <label for="group">Grouping</label>
        <select id="group" name="group" onchange="this.form.submit()">
          {_option_html(groups, group)}
        </select>
        <label for="item">Item</label>
        <select id="item" name="item" onchange="this.form.submit()">
          {_option_html(slugs, item_slug)}
        </select>
        <button type="submit">Apply</button>
      </form>
      <p class="muted"><a href="{html.escape(combined_url or '#')}" target="_blank" rel="noreferrer">Open combined matrices file</a></p>
      <p class="muted"><a href="{html.escape(item_matrix_url or '#')}" target="_blank" rel="noreferrer">Open selected matrix file</a></p>
    </div>
    <div class="right">
      <div class="card pane">
        <iframe src="{html.escape(plot_embed_url)}" title="Heatmap"></iframe>
      </div>
      <div class="card pane">
        <div class="table-wrap">{matrix_table}</div>
      </div>
    </div>
  </div>
</body>
</html>"""


def _render_correlations_page(
    manifest: dict, root_dir: Path, query: dict[str, list[str]]
) -> str:
    analyses = sorted(manifest.get("analyses", {}).keys())
    analysis = query.get("analysis", [None])[0]
    if analysis not in analyses:
        analysis = analyses[0] if analyses else None

    groups = (
        sorted(manifest["analyses"].get(analysis, {}).keys())
        if analysis is not None
        else []
    )
    group = query.get("group", [None])[0]
    if group not in groups:
        group = groups[0] if groups else None

    items = (
        manifest["analyses"][analysis][group].get("items", [])
        if analysis is not None and group is not None
        else []
    )
    slugs = [item["slug"] for item in items]
    item_slug = query.get("item", [None])[0]
    if item_slug not in slugs:
        item_slug = slugs[0] if slugs else None

    selected_item = next((item for item in items if item["slug"] == item_slug), None)
    item_plot_url = selected_item.get("plot_url") if selected_item else None
    plot_embed_url = (
        f"/plot?{urlencode({'url': item_plot_url})}" if item_plot_url else "about:blank"
    )
    matrix_url = selected_item.get("matrix_url") if selected_item else None
    pairwise_url = selected_item.get("pairwise_url") if selected_item else None

    matrix_path = root_dir / matrix_url.lstrip("/") if matrix_url else None
    pairwise_path = root_dir / pairwise_url.lstrip("/") if pairwise_url else None
    matrix_table = _matrix_preview_table(matrix_path)
    pairwise_table = _matrix_preview_table(pairwise_path)

    status_text = f"Correlation analyses discovered: {len(analyses)}"
    if analysis and group:
        status_text += f" | Items in selection: {len(items)}"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fixation Correlations Viewer</title>
  <style>
    :root {{
      --bg: #f4f5f7;
      --panel: #ffffff;
      --text: #122030;
      --muted: #59697b;
      --line: #d6dde6;
      --accent: #0b6a89;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #eef2f6 0%, #f7f9fb 60%, #f4f5f7 100%);
    }}
    .shell {{
      width: min(1400px, calc(100vw - 24px));
      margin: 12px auto;
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: 0 4px 16px rgba(10, 20, 30, 0.05);
    }}
    .controls {{ padding: 14px; position: sticky; top: 12px; height: fit-content; min-width: 0; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    .muted {{ margin: 0 0 10px; color: var(--muted); font-size: 13px; }}
    .nav {{
      margin: 8px 0 12px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .nav a {{
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #eef4f7;
      text-decoration: none;
      color: var(--text);
      font-size: 12px;
    }}
    .nav a.active {{
      background: #d8ebf3;
      border-color: #bdd9e6;
      font-weight: 600;
    }}
    label {{
      display: block; margin: 10px 0 6px; font-size: 12px; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.06em;
    }}
    select {{
      width: 100%; padding: 8px; border: 1px solid var(--line); border-radius: 7px;
      background: #fff; color: var(--text); font-size: 14px;
    }}
    button {{
      margin-top: 10px; width: 100%; padding: 8px; border: 1px solid var(--line);
      border-radius: 7px; background: #edf3f7; cursor: pointer;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .right {{ display: grid; gap: 14px; min-width: 0; }}
    iframe {{
      display: block; width: 100%; min-width: 0; min-height: 600px; border: 1px solid var(--line);
      border-radius: 8px; background: #fff;
    }}
    .pane {{ padding: 12px; min-width: 0; }}
    .table-wrap {{
      max-height: 420px; overflow: auto; width: 100%; min-width: 0;
      border: 1px solid var(--line); border-radius: 8px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; background: #fff; }}
    th, td {{
      border: 1px solid var(--line); padding: 6px 8px; text-align: center;
      white-space: nowrap;
    }}
    thead th {{ background: #eef4f7; position: sticky; top: 0; z-index: 1; }}
    @media (max-width: 1100px) {{
      .shell {{ grid-template-columns: 1fr; }}
      .controls {{ position: static; }}
      iframe {{ min-height: 480px; }}
    }}
    @media (max-width: 640px) {{
      iframe {{ min-height: 360px; }}
      .card {{ border-radius: 8px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="card controls">
      <h1>Fixation Correlations Viewer</h1>
      <div class="nav">
        <a href="/">Heatmaps</a>
        <a href="/correlations" class="active">Correlations</a>
      </div>
      <p class="muted">Data folder: {html.escape(manifest.get("correlations_dir", ""))}</p>
      <p class="muted">{html.escape(status_text)}</p>
      <form method="get" action="/correlations">
        <label for="analysis">Analysis</label>
        <select id="analysis" name="analysis" onchange="this.form.submit()">
          {_option_html(analyses, analysis)}
        </select>
        <label for="group">Correlation Group</label>
        <select id="group" name="group" onchange="this.form.submit()">
          {_option_html(groups, group)}
        </select>
        <label for="item">Item</label>
        <select id="item" name="item" onchange="this.form.submit()">
          {_option_html(slugs, item_slug)}
        </select>
        <button type="submit">Apply</button>
      </form>
      <p class="muted"><a href="{html.escape(matrix_url or '#')}" target="_blank" rel="noreferrer">Open correlation matrix file</a></p>
      <p class="muted"><a href="{html.escape(pairwise_url or '#')}" target="_blank" rel="noreferrer">Open pairwise correlations file</a></p>
    </div>
    <div class="right">
      <div class="card pane">
        <iframe src="{html.escape(plot_embed_url)}" title="Correlation Heatmap"></iframe>
      </div>
      <div class="card pane">
        <h3>Correlation Matrix Preview</h3>
        <div class="table-wrap">{matrix_table}</div>
      </div>
      <div class="card pane">
        <h3>Pairwise Correlations Preview</h3>
        <div class="table-wrap">{pairwise_table}</div>
      </div>
    </div>
  </div>
</body>
</html>"""


class ViewerHandler(SimpleHTTPRequestHandler):
    root_dir: Path = WORKDIR
    plots_dir: Path = DEFAULT_PLOTS_DIR

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            manifest = build_manifest(self.plots_dir, self.root_dir)
            query = parse_qs(parsed.query, keep_blank_values=True)
            payload = _render_page(manifest, self.root_dir, query).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/correlations" or parsed.path == "/correlations.html":
            manifest = build_correlation_manifest(self.plots_dir, self.root_dir)
            query = parse_qs(parsed.query, keep_blank_values=True)
            payload = _render_correlations_page(
                manifest, self.root_dir, query
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/plot":
            query = parse_qs(parsed.query, keep_blank_values=True)
            rel_url = query.get("url", [""])[0]
            if not rel_url.startswith("/"):
                self.send_error(400, "Invalid plot URL")
                return

            plot_path = (self.root_dir / rel_url.lstrip("/")).resolve()
            if not plot_path.exists() or not str(plot_path).startswith(str(self.root_dir)):
                self.send_error(404, "Plot file not found")
                return

            if plot_path.suffix == ".json":
                try:
                    figure_json = plot_path.read_text(encoding="utf-8")
                except Exception:
                    self.send_error(500, "Failed to read plot JSON")
                    return

                payload = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body, #plot {{
      margin: 0; width: 100%; height: 100%;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: #fff;
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const fig = {figure_json};
    const plotDiv = document.getElementById("plot");
    Plotly.newPlot(
      plotDiv,
      fig.data || [],
      fig.layout || {{}},
      Object.assign({{responsive: true, displaylogo: false}}, fig.config || {{}})
    );
    window.addEventListener("resize", () => Plotly.Plots.resize(plotDiv));
  </script>
</body>
</html>""".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if plot_path.suffix == ".html":
                self.send_response(302)
                self.send_header("Location", rel_url)
                self.end_headers()
                return

            self.send_error(400, "Unsupported plot format")
            return
        if parsed.path == "/manifest.json":
            manifest = build_manifest(self.plots_dir, self.root_dir)
            payload = json.dumps(manifest).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/correlations_manifest.json":
            manifest = build_correlation_manifest(self.plots_dir, self.root_dir)
            payload = json.dumps(manifest).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        return super().do_GET()


def main():
    parser = argparse.ArgumentParser(description="Serve fixation plot outputs in a web UI.")
    parser.add_argument("--dir", type=Path, default=None, help="Plots output directory")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    args = parser.parse_args()

    root_dir = PROJECT_ROOT.resolve()
    plots_dir, scored = resolve_plots_dir(args.dir, root_dir)
    manifest = build_manifest(plots_dir, root_dir)
    analyses = manifest.get("analyses", {})

    def _handler_init(self, *a, **kw):
        super(BoundViewerHandler, self).__init__(*a, directory=str(root_dir), **kw)

    BoundViewerHandler = type(
        "BoundViewerHandler",
        (ViewerHandler,),
        {"root_dir": root_dir, "plots_dir": plots_dir, "__init__": _handler_init},
    )

    server = ThreadingHTTPServer((args.host, args.port), BoundViewerHandler)
    print(f"Serving viewer at http://{args.host}:{args.port}")
    print(f"Using data folder: {plots_dir}")
    print("Candidate folders:")
    for cand, score in scored:
        print(f"  - {cand} (heatmaps: {score})")
    print(f"Discovered analyses: {', '.join(sorted(analyses.keys())) or 'none'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
