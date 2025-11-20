# Vistiq

[![License BSD-3](https://img.shields.io/github/license/ksiller/vistiq?label=license&style=flat)](https://github.com/ksiller/vistiq/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/vistiq.svg?color=green)](https://pypi.org/project/vistiq)
[![Python Version](https://img.shields.io/pypi/pyversions/vistiq.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/ksiller/vistiq/branch/main/graph/badge.svg)](https://codecov.io/gh/ksiller/vistiq)

**Vistiq**: Vistiq: Turn complex imaging data into actionable, quantitative insight with modular, multi-step analysis. Vistiq runs on Fluxon to orchestrate scalable and reproducible analysis pipelines.

## Capabilities

Vistiq offers a comprehensive suite of image analysis tools for image processing and analysis:

- **Preprocessing**: Multiple methods for image denoising and feature enhancement
- **Segmentation**: Multiple segmentation methods including local thresholding, iterative thresholding, and watershed-based techniques to identify atom positions in image stacks
- **Object Statistics**: Quantitative analysis of detected objects including area, perimeter, circularity, solidity, aspect ratio, eccentricity, and nearest-neighbor distances
- **Filtering**: Flexible filtering based on object properties (area, circularity, solidity, aspect ratio, etc.) to refine detections
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

Vistiq provides several subcommands for image analysis:

### Preprocess

Preprocess images with denoising and filtering operations:

```bash
vistiq preprocess -i imagestack.nd2 -o my_outputdir -g --substack 1-100
```

**Preprocess-specific arguments:**
```
  --preprocess-method {dog,noise2stack,none}
                        preprocessing method to use (default: dog)
  --sigma-low SIGMA_LOW
                        sigma for lower Gaussian blur in DoG (default: 1.0)
  --sigma-high SIGMA_HIGH
                        sigma for higher Gaussian blur in DoG (default: 5.0)
  --mode {reflect,constant,nearest,mirror,wrap}
                        border handling mode for Gaussian filtering (default: reflect)
  --window WINDOW       temporal window size for Noise2Stack denoising (odd recommended, default: 5)
  --exclude-center      exclude center frame from Noise2Stack average (default: True)
  --no-exclude-center   include center frame in Noise2Stack average
  --normalize           normalize output to [0, 1] range (default: True)
  --no-normalize        do not normalize output
```

### Segment

Segment images to identify objects using thresholding and labeling:

```bash
vistiq segment -i imagestack.nd2 -o my_outputdir -g --substack 1-100
```

**Segment-specific arguments:**
```
  --threshold-method {otsu,local,niblack,sauvola}
                        thresholding method to use (default: otsu)
  --block-size BLOCK_SIZE
                        block size for local thresholding (must be odd, default: 51)
  --connectivity {1,2}  connectivity for labeling (1 or 2, default: 1)
  --min-area MIN_AREA   minimum object area for filtering
  --max-area MAX_AREA   maximum object area for filtering
```

### Analyze

Analyze segmented images to extract object properties and statistics:

```bash
vistiq analyze -i segmented_images/ -o analysis_results/ -g
```

**Analyze-specific arguments:**
```
  --include-stats       include object statistics in analysis (default: True)
  --no-stats            exclude object statistics from analysis
  --include-coords      include coordinate extraction (default: True)
  --no-coords           exclude coordinate extraction
```

### Full Pipeline

Run the complete pipeline (segment + analyze) in a single command:

```bash
vistiq full -i imagestack.nd2 -o results/ -g --substack 1-100 --threshold-method otsu
```

**Full pipeline arguments:**
Includes all arguments from both `segment` and `analyze` subcommands.

### Coincidence Detection

Run a complete workflow for coincidence detection with DoG preprocessing, MicroSAM segmentation, and overlap analysis:

```bash
vistiq coincidence -i imagestack.nd2 -o output/ -g --substack T:4-10,Z:2-20
```

**Coincidence-specific arguments:**
```
  --sigma-low SIGMA_LOW
                        sigma for lower Gaussian blur in DoG (default: 1.0)
  --sigma-high SIGMA_HIGH
                        sigma for higher Gaussian blur in DoG (default: 12.0)
  --normalize           normalize DoG output to [0, 1] range (default: True)
  --no-normalize        disable normalization of DoG output
  --area-lower AREA_LOWER
                        lower bound for area filter (default: 100)
  --area-upper AREA_UPPER
                        upper bound for area filter (default: 10000)
  --volume-lower VOLUME_LOWER
                        lower bound for volume filter (default: 100)
  --volume-upper VOLUME_UPPER
                        upper bound for volume filter (default: 10000)
  --model-type {vit_l_lm,vit_b_lm,vit_t_lm,vit_h_lm}
                        MicroSAM model type (default: vit_l_lm)
  --threshold THRESHOLD
                        threshold for coincidence detection (default: 0.1)
  --method {iou,dice}   coincidence detection method (default: dice)
  --mode {bounding_box,outline}
                        coincidence detection mode (default: outline)
```

**Note:** The coincidence command does not require an output path. Results are saved to a directory based on the input file path (with extension stripped).

### Workflow

Build and run modular workflows from CLI-specified components:

```bash
vistiq workflow -i imagestack.nd2 -o output/ --component DoG --component OtsuThreshold
```

**Workflow-specific arguments:**
```
  --component COMPONENT  Component to include in workflow (can be specified multiple times)
                        Available components are auto-discovered from registered modules
```

The workflow subcommand allows you to dynamically build analysis pipelines by specifying components and their configurations via command-line arguments. All registered `Configurable` classes from `vistiq.preprocess`, `vistiq.seg`, `vistiq.analysis`, and `vistiq.core` modules are available.

### Common Arguments

All subcommands support these common arguments:

#### `-i, --input INPUT_PATH` (required)
Specifies the input image file or directory containing image files to be processed. 

- **Single file**: Provide the path to a single image file (e.g., `data/image.nd2`, `data/stack.tiff`)
- **Directory**: Provide the path to a directory containing multiple image files (e.g., `data/images/`)
- The input path can be absolute or relative to the current working directory
- Supported formats depend on installed bioio plugins (common formats: `.tiff`, `.ome-tiff`, `.nd2`, `.czi`, `.lif`, `.zarr`, etc.)

**Example:**
```bash
vistiq segment -i /path/to/images.nd2 -o output/
vistiq analyze -i ./segmented_images/ -o results/
```

#### `-o, --output OUTPUT_PATH` (optional)
Specifies where processed results should be saved.

- **File**: For single-file outputs, provide the full path including filename (e.g., `output/result.tiff`)
- **Directory**: For multi-file outputs, provide a directory path (e.g., `output/segmented/`)
- The output directory will be created if it doesn't exist
- **Default**: If not specified, results are saved to the current working directory

**Example:**
```bash
# Specify output directory
vistiq preprocess -i data/image.tiff -o output/preprocessed/
vistiq segment -i data/image.tiff -o output/segmented/

# Use default (current directory)
vistiq preprocess -i data/image.tiff
vistiq coincidence -i data/image.tiff
```

#### `-f, --substack SUBSTACK` (optional)
Selects a subset of the image data to process, allowing you to work with specific frames, slices, or channels without processing the entire dataset.

**Two formats are supported:**

1. **Legacy format** (applied to first axis):
   - Single frame: `'10'` - processes only frame 10 (1-based indexing)
   - Frame range: `'2-40'` - processes frames 2 through 40, inclusive (1-based indexing)
   - The first axis may be time (T), Z-stack, or another dimension depending on the image

2. **New format** (explicit dimension names):
   - Multiple dimensions: `'T:4-10,Z:2-20'` - processes time points 4-10 and Z-slices 2-20
   - Single dimension: `'C:0'` - processes only channel 0
   - Dimension names: `T` (time), `Z` (depth), `C` (channel), `Y` (height), `X` (width)
   - Ranges are 1-based and inclusive (e.g., `T:4-10` includes both frame 4 and frame 10)
   - Multiple dimensions are comma-separated

**Default behavior**: If `--substack` is not specified, all frames/slices/channels are processed.

**Examples:**
```bash
# Process single frame (legacy format)
vistiq segment -i data/image.tiff -o output/ --substack 10

# Process frame range (legacy format)
vistiq segment -i data/image.tiff -o output/ --substack 2-40

# Process specific time and Z ranges (new format)
vistiq segment -i data/image.tiff -o output/ --substack T:4-10,Z:2-20

# Process specific channel and Z range
vistiq segment -i data/image.tiff -o output/ --substack C:0,Z:5-50

# Process specific time points and channels
vistiq analyze -i data/image.tiff -o output/ --substack T:1-5,C:0-2
```

#### `-g, --grayscale` (optional)
Converts loaded images to grayscale before processing.

- Useful when working with color/multi-channel images but only intensity information is needed
- Reduces memory usage and processing time for color images
- If not specified, images are processed in their original format

**Example:**
```bash
vistiq segment -i data/color_image.tiff -o output/ -g
```

#### `-l, --loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}` (optional)
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

#### `-h, --help`
Displays help information for the command or subcommand.

### Examples

**Basic preprocessing with DoG:**
```bash
vistiq preprocess -i data/images.tiff -o output/preprocessed/
```

**Preprocessing with Noise2Stack denoising:**
```bash
vistiq preprocess -i data/images.tiff -o output/preprocessed/ \
  --preprocess-method noise2stack --window 7 --exclude-center
```

**Custom DoG filtering:**
```bash
vistiq preprocess -i data/images.tiff -o output/preprocessed/ \
  --preprocess-method dog --sigma-low 0.5 --sigma-high 3.0 --mode reflect
```

**Basic segmentation:**
```bash
vistiq segment -i data/images.tiff -o output/segmented/
```

**Segmentation with custom thresholding:**
```bash
vistiq segment -i data/images.tiff -o output/segmented/ \
  --threshold-method local --block-size 51 --min-area 10 --max-area 1000
```

**Analysis of segmented images:**
```bash
vistiq analyze -i output/segmented/ -o output/analysis/ --include-stats --include-coords
```

**Complete pipeline:**
```bash
vistiq full -i data/images.tiff -o output/results/ \
  --threshold-method otsu --connectivity 2 --min-area 5
```

**Coincidence detection with custom parameters:**
```bash
vistiq coincidence -i data/images.tiff -g \
  --substack T:0-10,Z:5-50 \
  --sigma-low 0.5 --sigma-high 10.0 \
  --volume-lower 50 --volume-upper 5000 \
  --threshold 0.2 --method iou --mode outline
```

**Substack examples:**
```bash
# Process single frame (legacy format, first axis)
vistiq segment -i data/images.tiff -o output/ --substack 10

# Process frame range (legacy format, first axis)
vistiq segment -i data/images.tiff -o output/ --substack 2-40

# Process specific time and Z ranges (new format)
vistiq segment -i data/images.tiff -o output/ --substack T:4-10,Z:2-20

# Process specific channel and Z range
vistiq segment -i data/images.tiff -o output/ --substack C:0,Z:5-50
```

## Using Vistiq in Python/Jupyter

Vistiq can also be used programmatically in Python scripts or Jupyter notebooks:

```python
from vistiq.seg import Segmenter, SegmenterConfig
from vistiq.preprocess import Preprocessor, PreprocessConfig

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



