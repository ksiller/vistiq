import numpy as np
from vistiq.seg import LabelRemoverConfig, LabelRemover

def main():
    labels = np.array(
        [
            [0, 1, 1, 0, 2, 2],
            [0, 1, 0, 0, 2, 2],
            [0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0, 0],
            [0, 3, 3, 3, 0, 0],
        ],
        dtype=np.int32,
    )

    # pretend feature table still has labels 1,2,3
    feature_labels_before = {1, 2, 3}

    remover = LabelRemover(LabelRemoverConfig())
    filtered_labels = remover.run(labels, [1])

    mask_labels_after = set(np.unique(filtered_labels)) - {0}
    feature_labels_after = feature_labels_before - {1}

    print("mask labels after:", mask_labels_after)
    print("feature labels after:", feature_labels_after)

    assert mask_labels_after == feature_labels_after

if __name__ == "__main__":
    main()