"""The docs overlay: one obvious button, and a panel that renders the project's own markdown in-page.

The published site is the first thing a reader meets and is otherwise a bare map. This puts the write-ups behind a single control -- `widget/docs/about.md`, written for the site, then the four repo documents as reference.

READ FROM THE ORIGINALS. The four reference docs are read from their repo-root paths at layout-build time, so editing README.md and re-exporting ships the new text; there is no copy to fall out of date. They ride to the browser inside a dcc.Store, which the snapshot inlines into index.html -- about 50 KB of markdown, ~20 KB gzipped, against a site that already ships a 2 MB MathJax chunk.

Deliberately NOT a build_bundle group: that path's staleness check is existence-based, so an edited doc would never rebuild and the export would ship the previous text silently.

Every callback is clientside (widget/assets/clientside/docs.js). A server callback on a static build renders a live-looking control that does nothing -- see widget/static/export.py's audit.
"""

import functools
from pathlib import Path

from dash import ClientsideFunction, Input, Output, State, dcc, html

_ROOT = Path(__file__).resolve().parents[2]  # widget/components -> widget -> repo root

# (slug, label, subtitle, path relative to the repo root, needs mathjax, youtube id or None)
#
# An entry with a video id renders the embed above its markdown, which is why that markdown reads as
# a caption. dcc.Markdown strips raw HTML, so an <iframe> written into a .md file would silently
# vanish; the embed is built as a real component in docs.js instead.
DOCS = [
    ("about", "About this project", "Start here -- short version", "widget/docs/about.md", False, None),
    ("stakeholders", "Applications", "For potential stakeholders", "widget/docs/stakeholders.md", False, None),
    ("summary", "Executive summary", "Full account, one page", "executive_summary.md", False, None),
    ("kpis", "Results & metrics", "Objectives and metrics for evaluation", "kpis.md", True, None),
    # ("readme", "Github README", "How to run it yourself", "README.md", True, None),
    ("data", "Data inventory", "All our data, documented", "data_inventory.md", False, None),
    ("video", "Presentation", "Five-minute walkthrough", "widget/docs/presentation.md", False, "O_ZCylQCXe8"),
    ("links", "Links", "Repo, video, data sources", "widget/docs/links.md", False, None),
]

# A document carrying this marker is still being written and is withheld entirely -- no tab, no text
# in the payload, so nothing half-finished reaches the published site.
#
# The switch lives in the DOCUMENT rather than in a flag here, so publishing happens in the same
# place the writing does and cannot be left half-done in two files. Deleting the marker line both
# publishes the tab and is how you preview it locally.
_DRAFT_MARKER = "<!-- DRAFT"


def _is_draft(rel: str) -> bool:
    """Whether a listed document is still a stub. A missing file counts as one, so a half-added tab degrades to absent rather than crashing the app."""
    path = _ROOT / rel
    if not path.exists():
        return True
    with path.open() as fh:
        return _DRAFT_MARKER in fh.read(500)


# What actually ships. DOCS stays the full declaration; everything downstream -- the nav, the
# callbacks, the payload -- is built from this, so the three cannot disagree about which tabs exist.
PUBLISHED = [d for d in DOCS if not _is_draft(d[3])]
_WITHHELD = [d[0] for d in DOCS if d not in PUBLISHED]

# Relative href -> slug, for the in-viewer link switch. Keyed on the FULL relative path: "src/README.md" must not resolve to the root README, and basename matching is exactly how it would.
_LINKS = {rel: slug for slug, _, _, rel, _, _ in PUBLISHED if "/" not in rel}
_REPO_BLOB = "https://github.com/Erdos-Projects/summer26-sustainability-of-agriculture/blob/main/"


def _read(rel: str) -> str:
    """One document's markdown, with the two failures that would ship a broken site made loud.

    A missing file is a build error, not an empty tab. `</script` would terminate the inlined layout JSON -- dash2html writes it into a <script> block without escaping `<`, so one such string anywhere in these docs blanks the whole published page.
    """
    path = _ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"docs panel: {path} is missing; it is listed in docs_panel.DOCS")
    text = path.read_text()
    if "</script" in text.lower():
        raise ValueError(f"docs panel: {rel} contains '</script', which would terminate the inlined layout JSON")
    return text


@functools.lru_cache(maxsize=1)
def payload() -> dict:
    """Everything the browser needs: text by slug, plus the link map the click handler resolves against."""
    return {
        "order": [slug for slug, *_ in PUBLISHED],
        "text": {slug: _read(rel) for slug, _, _, rel, _, _ in PUBLISHED},
        "mathjax": {slug: mj for slug, _, _, _, mj, _ in PUBLISHED},
        # Privacy-enhanced host: youtube-nocookie does not set tracking cookies until the viewer
        # actually presses play.
        "embed": {
            slug: (f"https://www.youtube-nocookie.com/embed/{vid}" if vid else None)
            for slug, _, _, _, _, vid in PUBLISHED
        },
        "links": _LINKS,
        "repo_blob": _REPO_BLOB,
    }


def _nav_button(slug: str, label: str, subtitle: str, first: bool) -> html.Button:
    return html.Button(
        id=f"docs-nav-{slug}",
        n_clicks=0,
        className="docs-nav-item" + (" is-active" if first else ""),
        children=[html.Span(label, className="docs-nav-label"), html.Span(subtitle, className="docs-nav-sub")],
    )


def layout():
    # Said out loud, because a tab that is declared in DOCS and simply absent from the panel is
    # otherwise indistinguishable from a bug.
    if _WITHHELD:
        print(f"[docs] withheld while still marked DRAFT: {', '.join(_WITHHELD)}")
    return html.Div(
        [
            dcc.Store(id="docs-payload", data=payload()),
            dcc.Store(id="docs-active", data=None),  # None until first open, so no markdown chunk loads at boot
            html.Button(
                id="docs-open-btn",
                n_clicks=0,
                className="docs-open-btn",
                title="About this project",
                children=[html.Span("☰", className="docs-open-icon"), html.Span("About this project")],
            ),
            html.Div(
                id="docs-overlay",
                className="docs-overlay",
                children=[
                    # Sibling of the panel, not its parent: nested, every click inside the panel would
                    # bubble here and n_clicks could not tell "clicked outside" from "clicked the text".
                    html.Div(id="docs-backdrop", className="docs-backdrop", n_clicks=0),
                    html.Div(
                        className="docs-panel",
                        children=[
                            html.Div(
                                className="docs-header",
                                children=[
                                    html.Span("Virtual nitrate sensors for Iowa", className="docs-title"),
                                    html.Button(
                                        "×",
                                        id="docs-close-btn",
                                        n_clicks=0,
                                        className="docs-close",
                                        title="Close (Esc)",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="docs-main",
                                children=[
                                    html.Nav(
                                        className="docs-nav",
                                        children=[
                                            _nav_button(s, lab, sub, i == 0)
                                            for i, (s, lab, sub, _, _, _) in enumerate(PUBLISHED)
                                        ],
                                    ),
                                    html.Div(id="docs-body", className="docs-body"),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )


def register_callbacks(app):
    """MUST also be called from widget/static/export.py::_build_app, which builds its own app.

    The callback audit reports callbacks that ARE server-side; it cannot report ones that are missing. Register here only and the panel works in `python widget/app.py` and is inert on the published site, silently.
    """
    # Open / close. Which one it was is read off the trigger id, the same idiom as ui.helpPopup.
    app.clientside_callback(
        ClientsideFunction(namespace="docs", function_name="toggle"),
        Output("docs-overlay", "className"),
        Input("docs-open-btn", "n_clicks"),
        Input("docs-close-btn", "n_clicks"),
        Input("docs-backdrop", "n_clicks"),
        prevent_initial_call=True,
    )

    # Which document. Opening with nothing selected lands on the custom copy.
    app.clientside_callback(
        ClientsideFunction(namespace="docs", function_name="activeDoc"),
        Output("docs-active", "data"),
        Input("docs-open-btn", "n_clicks"),
        *[Input(f"docs-nav-{slug}", "n_clicks") for slug, *_ in PUBLISHED],
        State("docs-active", "data"),
        prevent_initial_call=True,
    )

    # Render, plus the nav highlight. The dcc.Markdown is built in JS so its async chunk -- and
    # MathJax's 2 MB one -- load on first open rather than at page boot.
    app.clientside_callback(
        ClientsideFunction(namespace="docs", function_name="render"),
        Output("docs-body", "children"),
        *[Output(f"docs-nav-{slug}", "className") for slug, *_ in PUBLISHED],
        Input("docs-active", "data"),
        State("docs-payload", "data"),
        prevent_initial_call=True,
    )
