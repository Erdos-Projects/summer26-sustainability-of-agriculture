"""Read side of the precomputed asset bundle (widget/assets/data/).

widget/static/build_bundle.py writes the bundle and its manifest; this is the only thing the app should use to find out what is in it. Two jobs:

1. Expose the bundle's COVERAGE (which years, which intervals, how many sites) so the layout never has to glob the data directory to populate a control. That glob is exactly what made the app un-snapshottable: importing it touched src/data/interim.

2. Build asset URLs. These MUST be relative ("assets/data/..."), never rooted ("/assets/..."): the static site is published as a GitHub *project* page under user.github.io/<repo>/, and dash2html's index.html patch does not rewrite URLs embedded in the _dash-layout JSON. A leading slash there 404s on the published site while working fine locally, so it would not be caught until deploy.
"""

import json
from functools import lru_cache
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / "assets"
DATA_DIR = _ASSETS / "data"
MANIFEST = DATA_DIR / "manifest.json"

# Used when the manifest is absent (a fresh clone that has not run build_bundle yet). Keeps the app importable and the layout renderable; the panels themselves will show empty until the bundle exists.
_FALLBACK_COVERAGE = {
    "covariate_years": [2000, 2017],
    "forecast_years": [2013, 2017],
    "series_intervals": [{"interval": "1D", "nitrate_agg": "max", "precip_agg": "mean"}],
    "crop_classes": [],
    "basin_types": [0, 1, 2, 3],
    "n_sites": 0,
}


@lru_cache(maxsize=1)
def manifest() -> dict:
    if not MANIFEST.exists():
        return {"artifacts": {}, "groups": {}, "coverage": dict(_FALLBACK_COVERAGE)}
    return json.loads(MANIFEST.read_text())


def coverage() -> dict:
    return manifest().get("coverage", _FALLBACK_COVERAGE)


def url(rel: str) -> str:
    """Relative URL for a bundle artifact. See the module docstring on why it must stay relative."""
    return f"assets/data/{rel.lstrip('/')}"


def asset_url(rel: str) -> str:
    """Relative URL for a plain (non-bundle) file in widget/assets/."""
    return f"assets/{rel.lstrip('/')}"


def covariate_years() -> list:
    lo, hi = coverage().get("covariate_years", _FALLBACK_COVERAGE["covariate_years"])
    return list(range(lo, hi + 1))


def forecast_years() -> list:
    lo, hi = coverage().get("forecast_years", _FALLBACK_COVERAGE["forecast_years"])
    return list(range(lo, hi + 1))


def series_intervals() -> list:
    """[(interval, nitrate_agg, precip_agg)] the bundle actually contains."""
    return [(s["interval"], s["nitrate_agg"], s["precip_agg"]) for s in coverage()["series_intervals"]]


def has(rel: str) -> bool:
    return rel in manifest().get("artifacts", {})
