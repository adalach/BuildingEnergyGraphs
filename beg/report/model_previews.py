"""3D building previews for the graph explorer.

``build_pyvis_html_multi`` shows a building next to its graphs when it is given
``model_preview_urls``. :func:`write_model_previews` writes one interactive page
per building and returns that mapping. A building is read from its HBJSON when
one exists (BEM4AI ships them as ``models/hbjson.zip``) and rebuilt from its
epJSON otherwise, so the preview works with the epJSON files alone.

Plotly's library is written once, as ``plotly.min.js`` beside the pages, so the
previews open offline.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from beg.settings import data_dir


@dataclass(frozen=True)
class PreviewResult:
    #: model id -> page path relative to the explorer page, for ``model_preview_urls``
    urls: dict[str, str]
    #: model id -> "hbjson" | "epjson"
    sources: dict[str, str]
    #: model id -> reason, for buildings with no preview
    failed: dict[str, str]


def _find(root: Path | None, model_id: str, suffixes: tuple[str, ...]) -> Path | None:
    if root is None or not root.exists():
        return None
    for suffix in suffixes:
        direct = root / f"{model_id}{suffix}"
        if direct.exists():
            return direct
    lowered = {f"{model_id}{s}".lower() for s in suffixes}
    return next((p for p in root.iterdir() if p.name.lower() in lowered), None)


def load_model(model_id: str, *, hbjson_dir: Path | None = None, epjson_dir: Path | None = None):
    """Honeybee model for ``model_id`` and where it came from."""
    hbjson_dir = data_dir() / "hbjson" if hbjson_dir is None else Path(hbjson_dir)
    epjson_dir = data_dir() / "epjson" if epjson_dir is None else Path(epjson_dir)

    hbjson = _find(hbjson_dir, model_id, (".hbjson",))
    if hbjson is not None:
        from honeybee.model import Model

        return Model.from_hbjson(str(hbjson)), "hbjson"

    epjson = _find(epjson_dir, model_id, (".epJSON", ".epjson"))
    if epjson is not None:
        from beg.report.epjson_model import model_from_epjson

        model, _report = model_from_epjson(epjson, identifier=model_id)
        return model, "epjson"

    raise FileNotFoundError(f"no HBJSON or epJSON for {model_id}")


def write_model_previews(
    model_ids: Iterable[str],
    out_dir: Path | str,
    *,
    relative_to: Path | str | None = None,
    hbjson_dir: Path | None = None,
    epjson_dir: Path | None = None,
) -> PreviewResult:
    """Write ``<out_dir>/<model_id>.html`` for each building.

    ``relative_to`` is the directory of the page that embeds the previews (the
    explorer); the returned URLs are relative to it, so the pages keep working
    when the output folder is moved and no absolute path ends up in the HTML.
    """
    import os

    # Checked up front: the loop below records failures per building, where a
    # missing package would show as every preview being unavailable.
    missing = [name for name in ("plotly", "honeybee", "ladybug_geometry")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            f"3D previews need {', '.join(missing)}, which this environment does not have. "
            'Install the visualisation extras with  pip install -e ".[viz]"  and restart the kernel.'
        )

    from beg.report.honeybee_plot import plot_honeybee_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(relative_to) if relative_to is not None else out_dir.parent

    urls, sources, failed = {}, {}, {}
    for model_id in dict.fromkeys(str(m) for m in model_ids):
        try:
            model, source = load_model(model_id, hbjson_dir=hbjson_dir, epjson_dir=epjson_dir)
            fig = plot_honeybee_model(model, title=None, show=False, show_wireframe=True)
            # Fill the iframe.
            fig.update_layout(autosize=True, width=None, height=None,
                              margin=dict(l=0, r=0, t=0, b=0))
            page = out_dir / f"{model_id}.html"
            fig.write_html(
                str(page),
                include_plotlyjs="directory",
                full_html=True,
                default_width="100%",
                default_height="100%",
                config={"displaylogo": False, "scrollZoom": True, "responsive": True},
            )
        except Exception as exc:  # one broken building must not cost the others
            failed[model_id] = f"{type(exc).__name__}: {exc}"
            continue
        urls[model_id] = Path(os.path.relpath(page, base)).as_posix()
        sources[model_id] = source
    return PreviewResult(urls=urls, sources=sources, failed=failed)
