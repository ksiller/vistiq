import numpy as np
import pandas as pd
import pytest

from vistiq.segment.validation import (
    LabelFeatureAlignmentError,
    label_ids_from_mask,
    label_ids_from_regions,
    validate_label_feature_alignment,
)


def test_label_ids_from_mask_ignores_background():
    labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
    assert label_ids_from_mask(labels) == {1, 2}


def test_label_ids_from_dataframe():
    df = pd.DataFrame({"label": [1, 2], "area": [10, 20]})
    assert label_ids_from_regions(df) == {1, 2}


def test_validate_passes_when_aligned():
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)
    regions = pd.DataFrame({"label": [1, 2], "area": [5, 6]})
    validate_label_feature_alignment(labels, regions)


def test_validate_raises_on_mismatch():
    labels = np.array([[0, 1, 3], [0, 0, 0]], dtype=np.int32)
    regions = pd.DataFrame({"label": [1, 2], "area": [5, 6]})
    with pytest.raises(LabelFeatureAlignmentError, match="out of sync"):
        validate_label_feature_alignment(labels, regions, context="test")
