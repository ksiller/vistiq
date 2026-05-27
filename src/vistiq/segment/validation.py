"""Validate that label masks and region feature tables stay in sync."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Set, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LabelFeatureAlignmentError(ValueError):
    """Raised when label IDs in a mask do not match region feature label IDs."""


def label_ids_from_mask(labels: np.ndarray) -> Set[int]:
    """Return positive integer label IDs present in a label image."""
    unique = np.unique(labels)
    return {int(x) for x in unique if x != 0}


def label_ids_from_regions(
    regions: Union[List[Any], pd.DataFrame, None],
) -> Set[int]:
    """Return unique label IDs referenced by a region feature table or list."""
    if regions is None:
        return set()
    if isinstance(regions, pd.DataFrame):
        if regions.empty:
            return set()
        if "label" in regions.columns:
            return {int(x) for x in regions["label"].unique() if pd.notna(x)}
        if getattr(regions.index, "name", None) == "label":
            return {int(x) for x in regions.index.unique() if pd.notna(x)}
        raise LabelFeatureAlignmentError(
            "Region DataFrame has no 'label' column and index is not named 'label'."
        )
    if isinstance(regions, list):
        if len(regions) == 0:
            return set()
        return {int(r.label) for r in regions}
    raise LabelFeatureAlignmentError(
        f"Unsupported regions type for alignment check: {type(regions)!r}"
    )


def validate_label_feature_alignment(
    labels: np.ndarray,
    regions: Union[List[Any], pd.DataFrame, None],
    *,
    context: str = "",
    warn_duplicate_feature_rows: bool = True,
) -> None:
    """Ensure every label in the mask has exactly one feature row (by ID), and vice versa.

    Args:
        labels: Labeled image (2D or 3D).
        regions: Region properties as list or DataFrame.
        context: Optional caller name for error messages.
        warn_duplicate_feature_rows: Log when the feature table has repeated label IDs.

    Raises:
        LabelFeatureAlignmentError: If label ID sets differ or regions type is unsupported.
    """
    prefix = f"{context}: " if context else ""
    try:
        mask_ids = label_ids_from_mask(labels)
        feature_ids = label_ids_from_regions(regions)
    except LabelFeatureAlignmentError:
        raise
    except Exception as exc:
        raise LabelFeatureAlignmentError(
            f"{prefix}failed while extracting label IDs for alignment check"
        ) from exc

    if warn_duplicate_feature_rows and isinstance(regions, pd.DataFrame):
        if "label" in regions.columns and len(regions) != regions["label"].nunique():
            logger.warning(
                "%sfeature table has %d rows but only %d unique label IDs "
                "(per-slice 2D analysis on a 3D stack can cause this)",
                prefix,
                len(regions),
                regions["label"].nunique(),
            )

    if mask_ids == feature_ids:
        return

    only_in_mask = sorted(mask_ids - feature_ids)
    only_in_features = sorted(feature_ids - mask_ids)
    raise LabelFeatureAlignmentError(
        f"{prefix}label mask and region features are out of sync: "
        f"{len(mask_ids)} label(s) in mask, {len(feature_ids)} in features; "
        f"{len(only_in_mask)} only in mask {only_in_mask[:20]!r}, "
        f"{len(only_in_features)} only in features {only_in_features[:20]!r}"
    )
