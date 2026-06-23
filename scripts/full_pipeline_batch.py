#!/usr/bin/env python3
"""Single-file full pipeline for SLURM job arrays (notebook: hierarchical-analysis-gwas.ipynb).

Same workflow as :mod:`full_pipeline` (tissue + cells + hierarchical analysis) but
accepts one input file and an output directory, for use with
``scripts/batch_process.sbatch``::

    sbatch scripts/batch_process.sbatch scripts/full_pipeline_batch.py filelist.txt /path/to/output

Or directly::

    python scripts/full_pipeline_batch.py -i "$input_file" -o "$output_dir"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import vistiq
from vistiq.utils import check_device

from full_pipeline import (
    DEFAULT_EMBEDDING_PATH,
    parse_channel_list,
    run_full_pipeline,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run tissue + cell segmentation and hierarchical analysis on one "
            "microscopy volume (.lif, .tif, etc.)."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input image path (.lif, .tif, etc.)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output directory for label TIFFs and measurement CSVs",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Scene index for multi-scene containers (default: 0)",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path(DEFAULT_EMBEDDING_PATH),
        help="Directory with MicroSAM embeddings "
        f"(default: VISTIQ_EMBEDDING_PATH or {DEFAULT_EMBEDDING_PATH!r})",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated channel names applied after loading (e.g. Dpn,InR,EdU)",
    )
    parser.add_argument(
        "--channels-from-filename",
        action="store_true",
        help="Parse channel names from the input basename: split on '-', take the "
        "last N segments (N = number of channels). Ignored when --channels is set.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=-1,
        help="Worker count for PreprocessFlow (default: -1, all cores)",
    )
    parser.add_argument(
        "--segment-workers",
        type=int,
        default=2,
        help="Worker count for TiledSegmentationFlow (default: 2)",
    )
    parser.add_argument(
        "--tissue-labels",
        choices=("2d", "3d"),
        default="2d",
        help=(
            "Tissue segmentation mode: 2d Z-projects before segmentation and expands "
            "lobe labels to 3D; 3d segments the full volume (default: 2d)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logging.getLogger(vistiq.__name__).setLevel(getattr(logging, args.log_level))
    logger.info("Available Torch accelerators: %s", check_device())

    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_path.is_file():
        logger.error("Input file not found: %s", input_path)
        return 2

    channel_names = None
    if args.channels is not None:
        try:
            channel_names = parse_channel_list(args.channels)
        except ValueError as exc:
            logger.error("%s", exc)
            return 2

    embedding_path = args.embedding_path.expanduser().resolve()
    if not embedding_path.is_dir():
        logger.error("Embedding directory not found: %s", embedding_path)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing %s -> %s", input_path, output_dir)
    try:
        outputs = run_full_pipeline(
            input_path,
            outdir=output_dir,
            scene_index=args.scene_index,
            embedding_path=embedding_path,
            preprocess_workers=args.preprocess_workers,
            segment_workers=args.segment_workers,
            tissue_labels_mode=args.tissue_labels,
            channel_names=channel_names,
            channels_from_filename=args.channels_from_filename,
        )
    except Exception:
        logger.exception("Full pipeline failed for %s", input_path)
        return 1

    for name, path in outputs.items():
        logger.info("%s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
