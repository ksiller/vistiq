"""Debug helpers for segmentation (kept small to avoid import cycles)."""

import logging
import numpy as np


logger = logging.getLogger(__name__)


def debug_mask_labels(name, labels) -> None:
    unique_labels = sorted(set(np.unique(labels)) - {0})
    logger.debug(f"{name} mask labels (first 20):", unique_labels[:20])
    logger.debug(f"{name} mask label count:", len(unique_labels))
