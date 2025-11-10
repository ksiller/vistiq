# Vistiq

[![License BSD-3](https://img.shields.io/github/license/ksiller/vistiq?label=license&style=flat)](https://github.com/ksiller/vistiq/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/vistiq.svg?color=green)](https://pypi.org/project/vistiq)
[![Python Version](https://img.shields.io/pypi/pyversions/vistiq.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/ksiller/vistiq/branch/main/graph/badge.svg)](https://codecov.io/gh/ksiller/vistiq)

**Vistiq**: Vistiq: Turn complex imaging data into actionable, quantitative insight with modular, multi-step analysis. Vistiq runs on Fluxon to orchestrate scalable and reproducible analysis pipelines.

## Capabilities

Vistiq offers a comprehensive suite of image analysis tools for atomic-scale microscopy:

- **Segmentation**: Multiple segmentation methods including local thresholding, iterative thresholding, and watershed-based techniques to identify atom positions in image stacks
- **Object Statistics**: Quantitative analysis of detected objects including area, perimeter, circularity, solidity, aspect ratio, eccentricity, and nearest-neighbor distances
- **Filtering**: Flexible filtering based on object properties (area, circularity, solidity, aspect ratio, etc.) to refine detections
- **Lattice Analysis**: Identification of lattice holes and analysis of atomic coordination patterns
- **Voronoi Tessellation**: Construction of Voronoi diagrams to analyze local coordination environments
- **Spatial Probability Maps**: Generation of 3D probability distributions for atom positions based on detected centroids
- **Temporal Analysis**: Weighted temporal summation for analyzing time-series data with decay-based contributions
- **Visualization**: Export animations and create napari-compatible visualization layers

## Approach

Vistiq uses a multi-stage analysis pipeline:

1. **Segmentation**: Detects atom positions using adaptive thresholding and morphological operations
2. **Filtering**: Refines detections based on statistical properties of the detected objects
3. **Statistics**: Computes comprehensive object statistics including spatial distribution metrics
4. **Analysis**: Performs Voronoi tessellation and coordination analysis to characterize local atomic environments
5. **Visualization**: Generates probability maps and visualization layers for interactive exploration

The toolkit leverages scikit-image for image processing, scipy for spatial operations, and joblib for parallel processing across multiple planes in image stacks. It supports both 2D images and 3D/4D image stacks with efficient parallel processing.

## Installation 

You can install `vistiq` via [pip]:

    pip install vistiq


To install latest development version:

    pip install git+https://github.com/ksiller/vistiq.git

## Image file formats

Vistiq is reading image files using the [Bioio](https://github.com/bioio-devs/bioio) package. A variety of plugins exist to support common image file formats, including .tiff, .ome-tiff, .zarr, .nd2, .czi, .lif, etc.. By installing these additional bioio plugins you can easily expand Vistiq's ability to process a large variety of image formats without the need to touch the source code.  

## Running the Vistiq application

In a command line shell, run the following command:
```
vistiq -i imagestack.nd2 -o my_outputdir -g -f 1-100
```

**Command line arguments:**

```
  -h, --help            show this help message and exit
  -l LOGLEVEL, --loglevel LOGLEVEL
                        logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  -i INPUT_PATH, --input INPUT_PATH
                        single image file or directory with image files to be processed
  -o OUTPUT_PATH, --output OUTPUT_PATH
                        output file or directory
  -g, --grayscale       convert loaded images/videos to grayscale before processing
  -f FRAMES, --frames FRAMES
                        frames to process; examples: '10' or '2-40' (inclusive). Default: all frames
```                  

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



