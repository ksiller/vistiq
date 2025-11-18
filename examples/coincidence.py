from vistiq.core import ArrayIteratorConfig
from vistiq.preprocess import DoG, DoGConfig
from vistiq.seg import MicroSAMSegmenter, MicroSAMSegmenterConfig, RegionFilter, RegionFilterConfig, RangeFilter, RangeFilterConfig
from vistiq.analysis import CoincidenceDetector, CoincidenceDetectorConfig
from vistiq.utils import load_image, get_scenes, to_tif
import numpy as np
import pandas as pd
import os
import itertools
import argparse
from pathlib import Path


def parse_arguments():
    """Parse command-line arguments for coincidence detection workflow."""
    parser = argparse.ArgumentParser(
        description="Run coincidence detection workflow with DoG preprocessing, MicroSAM segmentation, and coincidence detection"
    )
    
    parser.add_argument(
        "--input",
        dest="input_path",
        type=str,
        default="input.tif",
        help="Input image file path (default: input.tif)"
    )
    
    parser.add_argument(
        "--sigma-low",
        dest="sigma_low",
        type=float,
        default=1.0,
        help="Sigma for lower Gaussian blur in DoG (default: 1.0)"
    )
    
    parser.add_argument(
        "--sigma-high",
        dest="sigma_high",
        type=float,
        default=12.0,
        help="Sigma for higher Gaussian blur in DoG (default: 12.0)"
    )
    
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="Normalize DoG output to [0, 1] range (default: True)"
    )
    
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable normalization of DoG output"
    )
    
    parser.add_argument(
        "--area-lower",
        dest="area_lower",
        type=float,
        default=100,
        help="Lower bound for area filter (default: 100)"
    )
    
    parser.add_argument(
        "--area-upper",
        dest="area_upper",
        type=float,
        default=10000,
        help="Upper bound for area filter (default: 10000)"
    )
    
    parser.add_argument(
        "--model-type",
        dest="model_type",
        type=str,
        default="vit_l_lm",
        choices=["vit_l_lm", "vit_b_lm", "vit_t_lm", "vit_h_lm"],
        help="MicroSAM model type (default: vit_l_lm)"
    )
    
    parser.add_argument(
        "--threshold",
        dest="threshold",
        type=float,
        default=0.1,
        help="Threshold for coincidence detection (default: 0.1)"
    )
    
    parser.add_argument(
        "--method",
        dest="method",
        type=str,
        default="dice",
        choices=["iou", "dice"],
        help="Coincidence detection method: 'iou' or 'dice' (default: dice)"
    )
    
    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="outline",
        choices=["bounding_box", "outline"],
        help="Coincidence detection mode: 'bounding_box' or 'outline' (default: outline)"
    )
    
    return parser.parse_args()


def coincidence(input_path, sigma_low, sigma_high, normalize, area_lower, area_upper, model_type, threshold, method, mode):
    """Main function to run the coincidence detection workflow.
    
    Args:
        input_path: Path to input image file
        sigma_low: Sigma for lower Gaussian blur in DoG
        sigma_high: Sigma for higher Gaussian blur in DoG
        normalize: Whether to normalize DoG output to [0, 1] range
        area_lower: Lower bound for area filter
        area_upper: Upper bound for area filter
        model_type: MicroSAM model type
        threshold: Threshold for coincidence detection
        method: Coincidence detection method ('iou' or 'dice')
        mode: Coincidence detection mode ('bounding_box' or 'outline')
    """

    # Set up configurations
    dog_config = DoGConfig(sigma_low=sigma_low, sigma_high=sigma_high, normalize=normalize)
    region_filter_config = RegionFilterConfig(
        filters=[RangeFilter(RangeFilterConfig(attribute="area", range=(area_lower, area_upper)))]
    )
    volume_it_cfg = ArrayIteratorConfig(slice_def=(-3, -2, -1))
    coincidence_config = CoincidenceDetectorConfig(
        method=method,
        mode=mode,
        iterator_config=volume_it_cfg,
        threshold=threshold
    )

    # Get absolute path of input and strip extension for output directory
    input_path_obj = Path(input_path).resolve()

    scenes = get_scenes(input_path)
    for idx, sc in enumerate(scenes):
        print (f"scene: {sc}")
        # Load image
        img, scale = load_image(input_path, scene_index=idx, squeeze=True)
        img = img[:,30:40,]

        output_dir = f"{input_path_obj.with_suffix('')}-{sc}-dog-{sigma_low}-{sigma_high}-threshold-{threshold}"
        os.makedirs(output_dir, exist_ok=True)
        print (f"output_dir: {output_dir}")
    
        img_ch = np.unstack(img, axis=0)
        print (f"img: {img.shape}, scale: {scale}")
        for im in img_ch:
            print (f"im.shape: {im.shape}")

        masks_ch = []
        labels_ch = []
        regions_ch = []

        for i, im in enumerate(img_ch):
            # 1. DoG preprocessing
            dog = DoG(dog_config)
            preprocessed = dog.run(im)
            output_path = f"{output_dir}/Preprocessed_Ch{i}.tif"
            to_tif(output_path, preprocessed)

            # 2. MicroSAMSegmenter with RegionFilter
            region_filter = RegionFilter(region_filter_config)
            # need to set up new microsam config for each channel to deal with different embeddings
            microsam_config = MicroSAMSegmenterConfig(
                model_type=model_type,
                region_filter=region_filter,
                do_labels=True,
                do_regions=True
            )

            microsam = MicroSAMSegmenter(microsam_config)
            masks, labels, regions = microsam.run(preprocessed)
            output_path = f"{output_dir}/Labels_Ch{i}.tif"
            to_tif(output_path, labels)
            masks_ch.append(masks)
            labels_ch.append(labels)
            regions_ch.append(regions)


        # 3. CoincidenceDetector
        # Create pairwise combinations of labels
        label_index_combinations = list(itertools.combinations(range(len(labels_ch)), 2))
        print(label_index_combinations)
        feature_dfs = {}
        for idx_combination in label_index_combinations:
            coincidence = CoincidenceDetector(coincidence_config)
            _, dfs = coincidence.run(
                labels_ch[idx_combination[0]], 
                labels_ch[idx_combination[1]], 
                stack_names=(f"Ch{idx_combination[0]}", f"Ch{idx_combination[1]}")
            )
            for key,df in dfs.items():
                if key not in feature_dfs:
                    feature_dfs[key] = [df]
                else:
                    feature_dfs[key].append(df)
                # print (f"key: {key}, df.columns: {df.columns}, df.describe(): {df.describe()}")
                # # Ch (key) vs Ch (df.columns[1])
                # output_path = f"{output_dir}/Coincidence_{key}_vs_{df.columns[1]}.csv"
                # df.to_csv(output_path, index=False)
            for key,dfs in feature_dfs.items():
                output_path = f"{output_dir}/Coincidence_{key}.csv"
                print (f"Saving to: {output_path}")
                dfs[0].join(dfs[1:]).to_csv(output_path, index=True)


if __name__ == "__main__":
    args = parse_arguments()
    coincidence(args.input_path, args.sigma_low, args.sigma_high, args.normalize, args.area_lower, args.area_upper, args.model_type, args.threshold, args.method, args.mode)