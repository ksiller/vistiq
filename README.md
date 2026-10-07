# Vistiq

[![License BSD-3](https://img.shields.io/github/license/ksiller/vistiq?label=license&style=flat)](https://github.com/ksiller/vistiq/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/vistiq.svg?color=green)](https://pypi.org/project/vistiq)
[![Python Version](https://img.shields.io/pypi/pyversions/vistiq.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/ksiller/vistiq/branch/main/graph/badge.svg)](https://codecov.io/gh/ksiller/vistiq)

**Vistiq** turns complex imaging data into actionable, quantitative insight with modular, multi-step analysis. Vistiq runs on Fluxon to orchestrate scalable and reproducible analysis pipelines.

## Capabilities

Vistiq offers a comprehensive suite of image analysis tools for image processing and analysis:

- **Preprocessing**: Multiple methods for image denoising and feature enhancement
- **Segmentation**: Multiple segmentation methods including local thresholding, iterative thresholding, and watershed-based techniques to identify atom positions in image stacks
- **Object Analysis & Filtering**: Quantitative analysis of individual object properties (including area, perimeter, circularity, solidity, aspect ratio, eccentricity, sphericity)
- **Spatio-Temporal Statistics**: Quantitative measurements of inter-object properties (object density, nearest-neighbor distances, coincidence detection, etc.) in space and time.
- **Visualization**: Voronoi tessellation, spatial density probability maps for objects, export of animations and creation of napari-compatible visualization layers


The toolkit leverages scikit-image for image processing, scipy for spatial operations, MicroSAM for segmentation, and joblib for parallel processing across multiple planes in image stacks. It supports both 2D images and 3D/4D image stacks with efficient parallel processing.

## Installation 

You can install `vistiq` via [pip]:

    pip install vistiq


To install latest development version:

    pip install git+https://github.com/ksiller/vistiq.git

### Development (editable) install with uv

To work on the source code, clone the repository and create a virtual environment with [uv](https://docs.astral.sh/uv/):

    git clone https://github.com/ksiller/vistiq.git
    cd vistiq
    uv venv --python 3.12
    source .venv/bin/activate

Install `vistiq` in editable mode, so that changes to the source code take effect without reinstalling:

    uv pip install -e .

Optional dependency groups (`microsam`, `notebook`, `napari`, `test`, or `all`) can be added as extras:

    uv pip install -e ".[all,test]"

Verify the installation and run the test suite:

    vistiq --help
    pytest

Alternatively, let uv manage the environment for you. `uv sync` creates `.venv` and installs `vistiq` in editable mode, and `uv run` executes commands inside that environment without activating it:

    uv sync --all-extras
    uv run vistiq --help

## Image file formats

Vistiq is reading image files using the [Bioio](https://github.com/bioio-devs/bioio) package. A variety of plugins exist to support common image file formats, including .tiff, .ome-tiff, .zarr, .nd2, .czi, .lif, etc.. By installing these additional bioio plugins you can easily expand Vistiq's ability to process a large variety of image formats without the need to touch the source code.  

## Using Vistiq in Python/Jupyter

Vistiq can also be used programmatically in Python scripts or Jupyter notebooks:

```python
from vistiq.segment import Segmenter, SegmenterConfig
from vistiq.preprocess import Preprocessor, PreprocessorConfig

# Logging is automatically configured when importing vistiq modules
# You can customize the logging level if needed:
from vistiq import configure_logger
configure_logger("DEBUG", force=True)

# Use the segmentation and preprocessing classes
config = SegmenterConfig(...)
segmenter = Segmenter(config)
results = segmenter.run(image_stack)
```

**Note:** Logging is automatically configured when you import vistiq modules, so you don't need to call `configure_logger()` unless you want to customize the logging level. This makes vistiq work seamlessly in Jupyter notebooks and interactive Python sessions.

## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [BSD-3] license, "vistiq" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[BSD-3]: http://opensource.org/licenses/BSD-3-Clause

[file an issue]: https://github.com/ksiller/vistiq/issues

[pip]: https://pypi.org/project/pip/

[PyPI]: https://pypi.org/



