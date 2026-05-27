"""Shared helpers for modular batch CLI commands (preprocess, segment, enrich, coincidence)."""

from __future__ import annotations

import itertools
import logging
import re
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from vistiq.analysis.coincidence import CoincidenceDetector, CoincidenceDetectorConfig
from vistiq.analysis.enrichment import (
    RegionDataFrameEnricher,
    RegionDataFrameEnrichmentConfig,
)
from vistiq.core import ArrayIteratorConfig, Configurable
from vistiq.io import DataFrameWriter, ImageWriter
from vistiq.segment.validation import validate_label_feature_alignment
from vistiq.utils import load_image

logger = logging.getLogger(__name__)


def sanitize_channel_name(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name.strip()) or "Ch0"


def channel_name_from_stem(stem: str, prefix: str) -> Optional[str]:
    token = f"{prefix}_"
    if stem.startswith(token):
        return sanitize_channel_name(stem[len(token) :])
    return None


def iter_channels(
    img: np.ndarray, metadata: dict[str, Any]
) -> Iterator[Tuple[str, np.ndarray]]:
    ch_axis = metadata.get("channel_axis")
    ch_names = list(metadata.get("channel_names") or [])
    if ch_axis is not None and img.ndim >= 4:
        for i, raw_name in enumerate(ch_names):
            yield sanitize_channel_name(raw_name), np.take(img, i, axis=ch_axis)
        return
    if img.ndim == 3:
        name = sanitize_channel_name(ch_names[0] if ch_names else "Ch0")
        yield name, img
        return
    raise ValueError(f"Cannot split channels for image shape {img.shape}")


def regions_to_dataframe(regions: Any) -> pd.DataFrame:
    if isinstance(regions, pd.DataFrame):
        df = regions.copy()
        if "label" not in df.columns and getattr(df.index, "name", None) == "label":
            df = df.reset_index()
        return df
    if isinstance(regions, list):
        rows: list[dict[str, Any]] = []
        for r in regions:
            row: dict[str, Any] = {"label": r.label}
            for prop in (
                "area",
                "volume",
                "aspect_ratio",
                "sphericity",
                "solidity",
            ):
                if hasattr(r, prop):
                    row[prop] = getattr(r, prop)
            if hasattr(r, "centroid") and r.centroid is not None:
                for i, v in enumerate(r.centroid):
                    row[f"centroid-{i}"] = v
            if hasattr(r, "bbox") and r.bbox is not None:
                for i, v in enumerate(r.bbox):
                    row[f"bbox-{i}"] = v
            rows.append(row)
        return pd.DataFrame(rows)
    raise TypeError(f"Unsupported regions type: {type(regions)!r}")


def run_component_chain(
    img: np.ndarray,
    metadata: dict[str, Any],
    components: List[Configurable],
    workers: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any, dict[str, Any]]:
    """Run a chain of components on one channel image.

    Returns:
        (labels, preprocessed_image, regions, metadata) where preprocessed_image is the
        last 2-tuple image output if a preprocessor ran, else the input image.
    """
    work = img
    meta = dict(metadata)
    labels: Optional[np.ndarray] = None
    regions: Any = None
    preprocessed: Optional[np.ndarray] = None

    for component in components:
        out = component.run(work, workers=workers, metadata=meta)
        if isinstance(out, tuple) and len(out) == 3:
            _mask, labels, regions = out
        elif isinstance(out, tuple) and len(out) == 2:
            work, meta = out
            preprocessed = work
        else:
            work = out
            preprocessed = work
    return labels, preprocessed, regions, meta


def save_channel_outputs(
    output_dir: Path,
    channel: str,
    labels: np.ndarray,
    regions: Any,
    imgwriter: ImageWriter,
    dfwriter: DataFrameWriter,
    metadata: dict[str, Any],
    *,
    preprocessed: Optional[np.ndarray] = None,
    context: str = "segment",
) -> None:
    if labels is None or regions is None:
        raise ValueError(f"{context}: missing labels or regions for channel {channel}")

    df = regions_to_dataframe(regions)
    validate_label_feature_alignment(labels, df, context=f"{context}:{channel}")

    ch_meta = {
        **metadata,
        "channel_names": [channel],
        "dim_order": "ZYX" if labels.ndim == 3 else "YX",
    }
    if preprocessed is not None:
        imgwriter.run(
            preprocessed,
            output_dir / f"Preprocessed_{channel}.tif",
            metadata=ch_meta,
        )
    imgwriter.run(
        labels.astype(np.uint16, copy=False),
        output_dir / f"Labels_{channel}.tif",
        metadata=ch_meta,
    )
    dfwriter.run(df, output_dir / f"Features_{channel}.csv")


def collect_label_volumes(search_root: Path) -> dict[Path, dict[str, Path]]:
    """Map output directories to channel -> Labels_<channel>.tif paths."""
    grouped: dict[Path, dict[str, Path]] = {}
    for path in sorted(search_root.rglob("Labels_*.tif")):
        channel = channel_name_from_stem(path.stem, "Labels")
        if channel is None:
            continue
        grouped.setdefault(path.parent, {})[channel] = path
    return grouped


def run_coincidence_on_directory(
    work_dir: Path,
    *,
    threshold: float,
    method: str,
    mode: str,
) -> None:
    label_map = collect_label_volumes(work_dir).get(work_dir, {})
    if len(label_map) < 2:
        logger.warning(
            "Skipping coincidence for %s: need >=2 Labels_*.tif, found %d",
            work_dir,
            len(label_map),
        )
        return

    volume_it = ArrayIteratorConfig(slice_def=(-3, -2, -1))
    detector = CoincidenceDetector(
        CoincidenceDetectorConfig(
            method=method,
            mode=mode,
            iterator_config=volume_it,
            threshold=threshold,
        )
    )
    channels = sorted(label_map.keys())
    arrays = {}
    for ch in channels:
        data, _meta = load_image(label_map[ch], squeeze=True)
        arrays[ch] = data

    for ch_a, ch_b in itertools.combinations(channels, 2):
        _, dfs = detector.run(arrays[ch_a], arrays[ch_b], stack_names=(ch_a, ch_b))
        for key, frame in dfs.items():
            out_csv = work_dir / f"Coincidence_{key}_{ch_a}_vs_{ch_b}.csv"
            frame.to_csv(out_csv, index=True)
            logger.info("Wrote %s", out_csv)
