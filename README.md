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

## Image file formats

Vistiq is reading image files using the [Bioio](https://github.com/bioio-devs/bioio) package. A variety of plugins exist to support common image file formats, including .tiff, .ome-tiff, .zarr, .nd2, .czi, .lif, etc.. By installing these additional bioio plugins you can easily expand Vistiq's ability to process a large variety of image formats without the need to touch the source code.  

## Running the Vistiq application

Vistiq provides several subcommands for image analysis. The following commands are available in the new Typer-based CLI:

- **preprocess** - Preprocess images with a chain of preprocessing steps
- **segment** - Segment and label images with a chain of processing steps  
- **train** - Train models using paired image and label datasets

**Note:** The `analyze`, `coincidence`, `full`, and `workflow` subcommands are currently available through the legacy argparse-based CLI. They will be migrated to the new Typer-based CLI in a future release.

All commands use a flexible, JSON-based configuration system that allows you to specify paths, patterns, and component configurations.

### Preprocess

Preprocess images with a chain of preprocessing steps (e.g., denoising, resizing):

```bash
# Simple usage with default DoG configuration
vistiq preprocess -i input.tif -o output -s DoG

# Multiple steps with JSON configuration
vistiq preprocess -i input.tif -o output -s '{"classname":"Resize", "width":256, "height":256}' -s '{"classname": "DoG", "sigma_low": 10, "sigma_high": 20}'

# Step name with default config
vistiq preprocess -i input.tif -o output -s Resize -s DoG

# With file list configuration
vistiq preprocess -i '{"paths": "~/test/", "include": "*.tif", "exclude": ["*.tmp"]}' -o ~/output -s DoG
```

**Available preprocessing steps:**
- `DoG` - Difference of Gaussians filtering
- `Resize` - Image resizing
- `Noise2Stack` - Temporal denoising

**Preprocess options:**
- `-i, --input`: Input file or directory configuration (can be a path string or JSON config)
- `-o, --output`: Output file or directory configuration (can be a path string or JSON config)
- `-s, --step`: Processing step/component (can be specified multiple times). Use step name or JSON config
- `--loader`: Optional image loader configuration (defaults to ImageLoaderConfig)

### Segment

Segment and label images with a chain of processing steps (thresholding, segmentation, labelling):

```bash
# Simple usage with default Otsu thresholding
vistiq segment -i input.tif -o output -s OtsuThreshold

# Multiple steps
vistiq segment -i '{"paths": "~/test/", "include": "*.tif"}' -o ~/output -s OtsuThreshold -s Watershed

# Step with custom configuration via JSON
vistiq segment -i input.tif -o output -s '{"classname": "OtsuThreshold", "threshold": 0.5}'

# Using MicroSAMSegmenter for instance segmentation
vistiq segment -i input.tif -o output -s '{"classname": "MicroSAMSegmenter", "model_type": "vit_l_lm"}'

# MicroSAMSegmenter with custom model type
vistiq segment -i input.tif -o output -s '{"classname": "MicroSAMSegmenter", "model_type": "vit_b_lm"}'
```

**Available segmentation steps:**
- `OtsuThreshold` - Otsu thresholding
- `LocalThreshold` - Local thresholding
- `Watershed` - Watershed segmentation
- `Labeller` - Connected components labelling
- `MicroSAMSegmenter` - MicroSAM-based instance segmentation

**Segment options:**
- `-i, --input`: Input file or directory configuration (can be a path string or JSON config)
- `-o, --output`: Output file or directory configuration (can be a path string or JSON config)
- `-s, --step`: Processing step/component (can be specified multiple times). Use step name or JSON config
- `--loader`: Optional image loader configuration (defaults to ImageLoaderConfig)

**Note:** The `analyze`, `coincidence`, `full`, and `workflow` subcommands are currently available through the legacy argparse-based CLI in `vistiq.app`. They will be migrated to the new Typer-based CLI in a future release.

### Common Arguments

All subcommands support these common arguments:

#### `-i, --input INPUT` (required for most commands)
Specifies the input file or directory configuration. Can be provided as:

- **Simple path string**: `--input path/to/file.tif` or `--input ~/images/`
- **JSON configuration**: `--input '{"paths": "path/to/dir", "include": "*.tif", "exclude": ["*.tmp"]}'`

**JSON configuration options:**
- `paths`: String, Path, or list of paths (files or directories) to search
- `include`: String or list of file patterns to include (e.g., `"*.tif"` or `["*.tif", "*.tiff"]`)
- `exclude`: String or list of file patterns to exclude (e.g., `"*training*"` or `["*.tmp", "*.bak"]`)
- `files`: Boolean - whether to search for files (default: `true`)
- `directories`: Boolean - whether to search for directories (default: `false`)
- `recursive`: Boolean - whether to search recursively (default: `true`)

**Examples:**
```bash
# Simple path
vistiq segment -i input.tif -o output/

# Directory with pattern matching
vistiq preprocess -i '{"paths": "~/test/", "include": "*.tif", "exclude": "*training*"}' -o output/

# Multiple paths
vistiq segment -i '{"paths": ["path1/", "path2/"], "include": "*.tif"}' -o output/
```

#### `-o, --output OUTPUT` (optional)
Specifies the output file or directory configuration. Can be provided as:

- **Simple path string**: `--output path/to/output` or `--output ~/results/`
- **JSON configuration**: `--output '{"path": "path/to/output", "format": "tif", "overwrite": false}'`

**Default**: Current working directory (`.`)

**Examples:**
```bash
# Simple path
vistiq segment -i input.tif -o ~/output/

# JSON configuration
vistiq preprocess -i input.tif -o '{"path": "~/output/", "format": "tif", "overwrite": true}'
```

#### `-s, --step STEP` (optional, can be specified multiple times)
Specifies a processing step/component to include in the pipeline. Can be provided as:

- **Component name**: `--step DoG` or `--step OtsuThreshold` (uses default configuration)
- **JSON configuration**: `--step '{"classname": "DoG", "sigma_low": 1.0, "sigma_high": 5.0}'`

Steps are executed in the order they are specified. Each step can be configured with step-specific arguments using the `--step{i}-*` prefix pattern (e.g., `--step0-sigma-low`, `--step1-threshold`).

**Examples:**
```bash
# Single step with default config
vistiq preprocess -i input.tif -o output -s DoG

# Multiple steps
vistiq segment -i input.tif -o output -s OtsuThreshold -s Watershed

# Step with JSON configuration
vistiq preprocess -i input.tif -o output -s '{"classname": "Resize", "width": 256, "height": 256}'
```

#### `--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}` (optional)
Sets the verbosity level for logging output.

- **DEBUG**: Most verbose, shows detailed diagnostic information
- **INFO**: Default level, shows general informational messages
- **WARNING**: Shows only warnings and errors
- **ERROR**: Shows only errors
- **CRITICAL**: Shows only critical errors

**Default**: `INFO`

**Example:**
```bash
vistiq segment -i data/image.tiff -o output/ --loglevel DEBUG
```

#### `--device {auto,cuda,mps,cpu}` (optional)
Specifies the device to use for processing.

- **auto**: Automatically select the best available device (default)
- **cuda**: Use CUDA GPU if available
- **mps**: Use Apple Metal Performance Shaders (MPS) if available
- **cpu**: Use CPU only

**Default**: `auto`

**Example:**
```bash
vistiq segment -i data/image.tiff -o output/ --device cuda
```

#### `--processes PROCESSES` (optional)
Specifies the number of parallel processes to use for processing.

- Use a positive integer to specify the exact number of processes (e.g., `4` for 4 processes)
- **Default**: `1` (single-threaded processing)

**Example:**
```bash
# Use 4 parallel processes
vistiq segment -i data/image.tiff -o output/ --processes 4
```

#### `-h, --help`
Displays help information for the command or subcommand.

### Examples

**Basic preprocessing with DoG:**
```bash
vistiq preprocess -i data/images.tiff -o output/preprocessed/ -s DoG
```

**Preprocessing with multiple steps:**
```bash
vistiq preprocess -i data/images.tiff -o output/preprocessed/ \
  -s '{"classname": "Resize", "width": 512, "height": 512}' \
  -s '{"classname": "DoG", "sigma_low": 1.0, "sigma_high": 5.0}'
```

**Preprocessing with file list configuration:**
```bash
vistiq preprocess -i '{"paths": "~/data/", "include": "*.tif", "exclude": ["*.tmp", "*training*"]}' \
  -o ~/output/ -s DoG
```

**Basic segmentation:**
```bash
vistiq segment -i data/images.tiff -o output/segmented/ -s OtsuThreshold
```

**Segmentation with multiple steps:**
```bash
vistiq segment -i data/images.tiff -o output/segmented/ \
  -s OtsuThreshold \
  -s Watershed \
  -s Labeller
```

**Segmentation with custom configuration:**
```bash
vistiq segment -i data/images.tiff -o output/segmented/ \
  -s '{"classname": "OtsuThreshold", "threshold": 0.5}'
```

**Segmentation with MicroSAM:**
```bash
# Use MicroSAMSegmenter with default model (vit_l_lm)
vistiq segment -i data/images.tiff -o output/segmented/ -s MicroSAMSegmenter

# Use MicroSAMSegmenter with custom model type
vistiq segment -i data/images.tiff -o output/segmented/ \
  -s '{"classname": "MicroSAMSegmenter", "model_type": "vit_b_lm"}'
```

**Training with image and label pairs:**
```bash
vistiq train \
  --input '{"paths": "~/images/", "include": "*Preprocessed_Red.tif", "exclude": "*training*"}' \
  --labels '{"paths": "~/labels/", "include": "*Labelled_Red.tif", "exclude": "*training*"}' \
  --output ~/trained \
  --step '{"classname": "MicroSAMTrainer"}'
```

## Using Vistiq in Python/Jupyter

Vistiq can also be used programmatically in Python scripts or Jupyter notebooks:

```python
from vistiq.seg import Segmenter, SegmenterConfig
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



