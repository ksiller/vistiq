"""Save and load hierarchical analysis artifacts for offline visualization.

Use ``save_analysis`` after post-hoc spatial analysis to write a
``{stem}-analysis/`` directory that ``load_analysis`` can reload
without rerunning the pipeline.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from vistiq.graph.graph import subtree_origin_key

logger = logging.getLogger(__name__)


def _scope_snapshot(scope: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if scope.match is not None:
        payload["match"] = scope.match
    if scope.exclude is not None:
        payload["exclude"] = scope.exclude
    if scope.auto_root:
        payload["auto_root"] = scope.auto_root
    return payload


def _knn_cfg_snapshot(cfg: Any) -> dict[str, Any]:
    return {
        "k": cfg.k,
        "mode": cfg.mode,
        "scope": _scope_snapshot(cfg.scope),
    }


def _rnn_cfg_snapshot(cfg: Any) -> dict[str, Any]:
    return {
        "radius": cfg.radius,
        "mode": cfg.mode,
        "scope": _scope_snapshot(cfg.scope),
    }


def _load_scope(payload: dict[str, Any]) -> Any:
    from vistiq.analysis import SpatialScopeConfig

    allowed = {
        key: payload[key]
        for key in ("match", "exclude", "auto_root")
        if key in payload
    }
    return SpatialScopeConfig.model_validate(allowed)


def _load_knn_cfg(payload: dict[str, Any]) -> Any:
    from vistiq.analysis import KnnAnalysisConfig

    scope = _load_scope(payload.get("scope", {}))
    try:
        return KnnAnalysisConfig.model_validate(
            {
                "k": payload.get("k", 5),
                "mode": payload.get("mode", "homotypic"),
                "scope": scope,
            }
        )
    except Exception:
        return KnnAnalysisConfig(
            k=payload.get("k", 5),
            mode=payload.get("mode", "homotypic"),
            scope=scope,
        )


def _load_rnn_cfg(payload: dict[str, Any]) -> Any:
    from vistiq.analysis import RnnAnalysisConfig

    scope = _load_scope(payload.get("scope", {}))
    try:
        return RnnAnalysisConfig.model_validate(
            {
                "radius": payload.get("radius", 15.0),
                "mode": payload.get("mode", "homotypic"),
                "scope": scope,
            }
        )
    except Exception:
        return RnnAnalysisConfig(
            radius=payload.get("radius", 15.0),
            mode=payload.get("mode", "homotypic"),
            scope=scope,
        )


def save_analysis(
    outdir: Path | str,
    *,
    stem: str,
    measurements: dict[str, Any],
    features: pd.DataFrame,
    spatial_results: dict[str, Any],
    spatial_origins: list[Any],
    knn_cfg: Any = None,
    rnn_cfg: Any = None,
) -> Path:
    """Write analysis tables, spatial matrices, and graphs to ``{stem}-analysis/``.

    Returns the bundle directory path.
    """
    bundle_dir = Path(outdir) / f"{stem}-analysis"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    features.to_parquet(bundle_dir / "features.parquet")
    region_all = measurements.get("region_analyzer_all")
    if isinstance(region_all, pd.DataFrame):
        region_all.to_parquet(bundle_dir / "region_analyzer_all.parquet")
    spatial_summary = measurements.get("spatial_summary")
    if isinstance(spatial_summary, pd.DataFrame):
        spatial_summary.to_parquet(bundle_dir / "spatial_summary.parquet")

    with (bundle_dir / "containment_graph.pkl").open("wb") as handle:
        pickle.dump(measurements["containment_graph"], handle, protocol=pickle.HIGHEST_PROTOCOL)

    spatial_dir = bundle_dir / "spatial"
    spatial_dir.mkdir(exist_ok=True)
    for key, result in spatial_results.items():
        result.distance_matrix.to_parquet(spatial_dir / f"{key}.distance.parquet")
        result.matrix.to_parquet(spatial_dir / f"{key}.matrix.parquet")
        with (spatial_dir / f"{key}.graph.pkl").open("wb") as handle:
            pickle.dump(result.graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "spatial_origins": [subtree_origin_key(origin) for origin in spatial_origins],
        "spatial_result_keys": list(spatial_results.keys()),
    }
    if knn_cfg is not None:
        meta["knn_analysis"] = knn_cfg.model_dump(mode="json")
    if rnn_cfg is not None:
        meta["rnn_analysis"] = rnn_cfg.model_dump(mode="json")
    (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return bundle_dir


def load_analysis(
    bundle_dir: Path | str,
) -> dict[str, Any]:
    """Load a bundle written by ``save_analysis``.

    Returns a dict with ``containment_graph``, ``features``,
    ``region_analyzer_all``, ``spatial_summary``, ``spatial_results``,
    ``spatial_origins``, ``knn_cfg``, ``rnn_cfg``, ``meta``, and
    ``bundle_dir``.
    """
    from vistiq.analysis.spatial import SpatialGraphResult

    root = Path(bundle_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    meta = json.loads((root / "meta.json").read_text())
    with (root / "containment_graph.pkl").open("rb") as handle:
        containment_graph = pickle.load(handle)

    features = pd.read_parquet(root / "features.parquet")
    region_path = root / "region_analyzer_all.parquet"
    region_analyzer_all = (
        pd.read_parquet(region_path) if region_path.exists() else None
    )
    summary_path = root / "spatial_summary.parquet"
    spatial_summary = (
        pd.read_parquet(summary_path) if summary_path.exists() else None
    )

    spatial_results: dict[str, SpatialGraphResult] = {}
    spatial_dir = root / "spatial"
    for key in meta.get("spatial_result_keys", []):
        with (spatial_dir / f"{key}.graph.pkl").open("rb") as handle:
            graph = pickle.load(handle)
        spatial_results[key] = SpatialGraphResult(
            distance_matrix=pd.read_parquet(spatial_dir / f"{key}.distance.parquet"),
            matrix=pd.read_parquet(spatial_dir / f"{key}.matrix.parquet"),
            graph=graph,
        )

    knn_cfg = rnn_cfg = None
    if "knn_analysis" in meta:
        try:
            knn_cfg = _load_knn_cfg(meta["knn_analysis"])
        except Exception as exc:
            logger.warning(
                "Skipping knn_analysis config in %s: %s",
                root / "meta.json",
                exc,
            )
    if "rnn_analysis" in meta:
        try:
            rnn_cfg = _load_rnn_cfg(meta["rnn_analysis"])
        except Exception as exc:
            logger.warning(
                "Skipping rnn_analysis config in %s: %s",
                root / "meta.json",
                exc,
            )

    return {
        "containment_graph": containment_graph,
        "features": features,
        "region_analyzer_all": region_analyzer_all,
        "spatial_summary": spatial_summary,
        "spatial_results": spatial_results,
        "spatial_origins": meta.get("spatial_origins", []),
        "knn_cfg": knn_cfg,
        "rnn_cfg": rnn_cfg,
        "meta": meta,
        "bundle_dir": root,
    }


def default_spatial_result_key(
    spatial_results: dict[str, Any],
    analysis: str,
    mode: str,
    *,
    origin_key: Optional[str] = None,
) -> str:
    """Return the ``spatial_results`` dict key for one analysis/mode pair.

    Keys have the form ``{analysis}_{mode}@{origin_key}``. When multiple
    subtree roots match, pass *origin_key* explicitly.
    """
    prefix = f"{analysis}_{mode}@"
    matches = [key for key in spatial_results if key.startswith(prefix)]
    if origin_key is not None:
        target = f"{prefix}{origin_key}"
        if target not in spatial_results:
            raise KeyError(f"spatial result {target!r} not found")
        return target
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"no spatial result for {analysis!r} mode={mode!r}")
    raise ValueError(
        f"multiple spatial results for {analysis!r} mode={mode!r}: {matches}; "
        "pass origin_key explicitly"
    )
