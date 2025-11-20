import numpy as np
import pandas as pd
from typing import Optional, Union
from napari.utils.colormaps.colormap_utils import Colormap
import matplotlib.pyplot as plt

def color_labels_by_feature(labels_array: np.ndarray, features_table: pd.DataFrame, feature: str, colors: Optional[Union[str,Colormap, dict[int, tuple[float, float, float, float]]]] = None) -> ColorMap:
    """Color labels by a feature in a features table. Colormap can be a dict keyed by label and values representing RGBA values. If colormap is None, a default colormap will be used."""
    if colormap is None:
        colormap = plt.get_cmap("tab10")
        # turn colormap into array of colors
        colors = colormap(np.linspace(0, 1, colormap.N))
    if feature in features_table.columns:
        if features_table[feature].dtype == bool:
            pass
        else:
            new_colormap = {
                'colors': colors,
                'name': 'gradient',
                'interpolation': 'linear'
            }
    return colormap(labels_array)