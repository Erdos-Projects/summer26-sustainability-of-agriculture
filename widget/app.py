"""Run the widget locally.

The panels read the precomputed bundle in widget/assets/data/ rather than src/data/ directly -- the same code path the static export ships, so what you see here is what the published site does. The cost is that the app no longer shows edits to src/data/ the moment they land; `--refresh` rebuilds the stale groups first and restores that.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dash import Dash

from layout import build_layout
from components import map_panel, info_panel, forecast_panel

# suppress_callback_exceptions: several callbacks target components that only exist once a
# selection has been made (the per-row table buttons carry pattern-matching ids).
app = Dash(__name__, suppress_callback_exceptions=True)
app.layout = build_layout()

map_panel.register_callbacks(app)
info_panel.register_callbacks(app)
forecast_panel.register_callbacks(app)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--refresh",
        nargs="*",
        metavar="GROUP",
        help="rebuild the asset bundle before serving; names one or more groups, or all of them if given no names",
    )
    p.add_argument("--force", action="store_true", help="with --refresh, rebuild even if the manifest says current")
    p.add_argument("--port", type=int, default=8050)
    a = p.parse_args()

    if a.refresh is not None:
        from widget.static import build_bundle

        build_bundle.main(only=a.refresh or None, force=a.force)
        # bundle.manifest() is lru_cached and the layout above already read it, so any coverage the rebuild just changed (years, intervals, site count) would not reach the controls.
        import bundle

        bundle.manifest.cache_clear()
        app.layout = build_layout()

    app.run(debug=True, port=a.port)
