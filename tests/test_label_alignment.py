import numpy as np
import pandas as pd

from vistiq.seg import (
    RegionFilter,
    RegionFilterConfig,
    RangeFilter,
    RangeFilterConfig,
    LabelRemover,
    LabelRemoverConfig,
)
from vistiq.utils import ArrayIteratorConfig


def main():
    # Fake label image with non-consecutive labels on purpose
    labels = np.array([
        [0, 5, 5, 0, 6, 6],
        [0, 5, 5, 0, 6, 6],
        [0, 0, 0, 0, 8, 8],
        [10, 10, 0, 0, 8, 8],
    ], dtype=np.int32)

    # Fake features table
    features = pd.DataFrame({
        "label": [5, 6, 8, 10],
        "area": [4, 4, 4, 2],
    })

    # Filter keeps area >= 3, so label 10 should be removed
    region_filter = RegionFilter(
        RegionFilterConfig(
            filters=[
                RangeFilter(
                    RangeFilterConfig(attribute="area", range=(3, 100))
                )
            ]
        )
    )

    accepted_regions, removed_labels = region_filter.run(features)

    label_remover = LabelRemover(
        LabelRemoverConfig(
            iterator_config=ArrayIteratorConfig(slice_def=()),
            remap=False,
            output_type="stack",
            squeeze=False,
        )
    )

    filtered_labels = label_remover.run(labels, removed_labels)

    csv_labels = set(accepted_regions["label"].astype(int))
    mask_labels = set(np.unique(filtered_labels)) - {0}

    print("Original feature labels:", set(features["label"]))
    print("Removed labels:", set(np.asarray(removed_labels).tolist()))
    print("Remaining CSV labels:", csv_labels)
    print("Remaining mask labels:", mask_labels)
    print("Match:", csv_labels == mask_labels)
    print("\nFiltered label image:\n", filtered_labels)

    assert set(np.asarray(removed_labels).tolist()) == {10}, "Expected label 10 to be removed"
    assert csv_labels == {5, 6, 8}, "Expected remaining CSV labels to be {5, 6, 8}"
    assert mask_labels == {5, 6, 8}, "Expected remaining mask labels to be {5, 6, 8}"
    assert csv_labels == mask_labels, "Mask labels and CSV labels do not match"

    print("\nTest passed.")


if __name__ == "__main__":
    main()