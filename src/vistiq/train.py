import os
import numpy as np
import logging
import torch
import torch_em
from torch_em.transform.label import PerObjectDistanceTransform
from torchvision.transforms import v2
import micro_sam.training as sam_training
from pathlib import Path
from typing import Literal, Tuple, List, Optional, Union, Callable
from micro_sam.util import export_custom_sam_model
from bioio import BioImage
from vistiq.core import Configuration, Configurable
from prefect import task
from vistiq.utils import find_matching_file_pairs, collect_all_files
from pydantic import Field, PositiveInt, field_validator, ValidationInfo, model_validator
from bioio_ome_tiff.writers import OmeTiffWriter

logger = logging.getLogger(__name__)

class DatasetCreatorConfig(Configuration):
    """Configuration for dataset.
    
    Attributes:
        out_path: Path to save the dataset files (default: "./training_data").
        patterns: Tuple of image and label patterns (default: ("*_img.tif", "*_label.tif")).
        rematch: Whether to rematch the image and label pairs (default: False).
        strategy: Tuple of image and label strategy (default: (0,1)).
        remove_empty_labels: Whether to remove empty labels (default: True).
        random_samples: Number of random slices to take from each image stack and label stack pair. Ignored for 2D datasets (default: None, all slices are used).
    """
    out_path: Path = Field(description="Path to save the dataset files", default="./training_data")
    patterns: Tuple[str,str] = Field(description="Tuple of image and label patterns", default=("*_img.tif", "*_label.tif"))
    rematch: bool = False
    strategy: Tuple[int,int] = (0,1)
    remove_empty_labels: bool = True
    random_samples: Optional[PositiveInt] = None
    
    @field_validator('out_path', mode='after')
    @classmethod
    def normalize_out_path(cls, v: Optional[Path]) -> Path:
        """Normalize out_path to a Path object.
        
        Ensures out_path is always a Path object. Does NOT expand '~/' to preserve
        portability across different users. Expansion happens at runtime.
        
        Args:
            v: Path object or None (Pydantic already converts strings to Path).
            
        Returns:
            Path object (defaults to "./training_data" if None).
        """
        if v is None:
            return Path("./training_data")
        return v

class DatasetCreator(Configurable[DatasetCreatorConfig]):
    """Create a dataset.
    
    Attributes:
        config: Dataset configuration.
    """
    def __init__(self, config: DatasetCreatorConfig):
        """Initialize the dataset.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: DatasetCreatorConfig) -> "DatasetCreator":
        """Create a DatasetCreator instance from a configuration.
        """
        return cls(config)

    @task(name="DatasetCreator.run")
    def run(self, image_path: List[Union[Path, str]], label_path: List[Union[Path, str]]) -> List[Tuple[Path, Path]]:
        """Run the dataset creation.
        
        Args:
            image_path: List of image file or directory paths.
            label_path: List of label file or directory paths.
            
        Returns:
            List of (image_path, label_path) tuples for training.
        """
        logger.info(f"Running dataset creation with config: {self.config}")
        # Normalize to Path objects
        image_paths = [Path(p) if isinstance(p, str) else p for p in image_path]
        label_paths = [Path(p) if isinstance(p, str) else p for p in label_path]
        
        logger.info(f"Image paths ({len(image_paths)}): {[str(p.resolve()) for p in image_paths]}")
        logger.info(f"Label paths ({len(label_paths)}): {[str(p.resolve()) for p in label_paths]}")
        logger.info(f"Patterns: {self.config.patterns}")
        logger.info(f"Strategy: {self.config.strategy}")
        
        # If both lists are the same length and all are files, assume they're already matched
        if (not self.config.rematch and len(image_paths) == len(label_paths) and 
            all(p.is_file() for p in image_paths) and 
            all(p.is_file() for p in label_paths)):
            matches = list(zip(image_paths, label_paths))
            logger.info(f"Using pre-matched pairs: {len(matches)} pairs")
        else:
            # Collect files from paths
            pattern_a = self.config.patterns[0] if self.config.patterns else None
            pattern_b = self.config.patterns[1] if self.config.patterns else None
            
            files_a_dict = collect_all_files(image_paths, pattern_a)
            files_b_dict = collect_all_files(label_paths, pattern_b)
            
            # Convert to lists for matching
            files_a_list = list(files_a_dict.values())
            files_b_list = list(files_b_dict.values())
            
            logger.info(f"Collected {len(files_a_list)} image files and {len(files_b_list)} label files")
            
            # Match files using the specified strategy
            matches = find_matching_file_pairs(
                files_a_list, 
                files_b_list, 
                strategy=self.config.strategy,
                patterns=self.config.patterns
            )
        
        logger.info(f"Found {len(matches)} matching pairs")
        # check if self.config.out_path is relative path; if so use image_path dir and label_path dir as root and append out_path to it
        # Expand ~/ at runtime for portability (but don't resolve yet - need to check if absolute first)
        out_path = self.config.out_path.expanduser()
        
        # If absolute, resolve it once before the loop
        if out_path.is_absolute():
            out_path = out_path.resolve()
        
        training_pairs = []
        total_2d_pairs = 0
        for image_path, label_path in matches:
            if not out_path.is_absolute():
                # Relative path: append to parent directories of the matched files
                out_path_img = image_path.parent / out_path
                out_path_label = label_path.parent / out_path
            else:
                # Absolute path: use as-is
                out_path_img = out_path
                out_path_label = out_path
            os.makedirs(out_path_img, exist_ok=True)
            os.makedirs(out_path_label, exist_ok=True)
            img_out = out_path_img / image_path.name
            label_out = out_path_label / label_path.name

            img = BioImage(image_path)
            label = BioImage(label_path)
            img_data = img.get_image_data()
            label_data = label.get_image_data()
            if self.config.remove_empty_labels:
                keep_mask = np.sum(label_data, axis=(-2,-1)) > 0
                img_data = img_data[keep_mask]
                label_data = label_data[keep_mask]
            if self.config.random_samples is not None and len(img_data) > self.config.random_samples and len(label_data) > self.config.random_samples:
                rnd_indices = np.random.choice(img_data.shape[0], size=self.config.random_samples, replace=False)
                img_data = img_data[rnd_indices]
                label_data = label_data[rnd_indices]
            OmeTiffWriter.save(img_data, img_out, dim_order=img.dims.order[-img_data.ndim:], physical_pixel_sizes=img.physical_pixel_sizes, channel_names=img.channel_names)
            OmeTiffWriter.save(label_data, label_out, dim_order=(label.dims.order[-label_data.ndim:]), physical_pixel_sizes=label.physical_pixel_sizes, channel_names=label.channel_names)
            logger.info(f"Created dataset pair with image {img_data.shape} and label {label_data.shape}")
            training_pairs.append((img_out, label_out))
            total_2d_pairs += len(img_data)
        logger.info(f"Total 2D pairs: {total_2d_pairs}")
        return training_pairs

class TrainerConfig(Configuration):
    """Configuration for trainer.
    
    Attributes:
        model_type: Model type (default: "vit_l_lm").
        checkpoint_name: Name for the checkpoint (default: "sam_synthetic").
        export_path: Path to export the trained model (default: "./finetuned_synthetic_model.pth").
        batch_size: Batch size for training (default: 1).
        patch_shape: Shape of patches for training (default: (1, 512, 512)).
        roi_def: Region of interest definition as a slice (default: all slices).
        device: Device to use for training, "cuda" or "cpu" (default: "cuda").
        split_ratio: Ratio for train/validation split (default: 0.8).
    """
    model_type: str = "vit_l_lm"
    export_path: Path = "./"
    export_model_name: str | None = None # field validator will ensure that a default will be generated if not provided
    save_root: Path = "./"
    batch_size: int = 1
    patch_shape: Tuple[int, ...] = (1, 256, 256)
    roi_def: tuple[Union[int, slice], ...] = np.s_[:, :, :] # default all slices in volume
    device: str = "cuda"
    split_ratio: float = 0.8
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    @field_validator('export_path', 'save_root', mode='after')
    @classmethod
    def normalize_path_fields(cls, v: Union[str, Path, None]) -> Path:
        """Normalize path fields to Path objects.
        
        Ensures path fields are always Path objects. Does NOT expand '~/' to preserve
        portability across different users. Expansion happens at runtime.
        
        Args:
            v: Path object, string, or None. Handles both cases in case validation is bypassed.
            
        Returns:
            Path object (defaults to "./" if None).
        """
        if v is None:
            return Path("./")
        if isinstance(v, str):
            return Path(v)
        return v

    @model_validator(mode='after')
    def set_export_model_name(self) -> 'TrainerConfig':
        """Set default export_model_name if not provided.
        
        If export_model_name is empty, generates a default name based on model_type.
        This runs after all field validators, so we can access model_type.
        """
        # Only update if value is empty
        if not self.export_model_name or self.export_model_name == "":
            new_value = f"finetuned_{self.model_type}_model.pth"
        else:
            new_value = self.export_model_name
        
        # Ensure .pth suffix
        new_value = str(Path(new_value).with_suffix(".pth"))
        
        # Return new instance if value changed, otherwise return self
        if new_value != self.export_model_name:
            return self.model_copy(update={"export_model_name": new_value})
        return self

class MicroSAMTrainerConfig(TrainerConfig):
    """Configuration for MicroSAM trainer.
    
    Attributes:
        split_ratio: Ratio for train/validation split (default: 0.8).
        raw_key: Key for raw image data in the data file (default: None).
        label_key: Key for label data in the data file (default: None).
        learning_rate: Learning rate for training (default: 1e-5).
        log_image_interval: Interval for logging images during training (default: 10).
        n_objects_per_batch: Number of objects per batch (default: 10).
        n_iterations: Number of training iterations (default: 10000).
        n_sub_iteration: Number of sub-iterations (default: 8).
        mixed_precision: Whether to use mixed precision training (default: True).
        compile_model: Whether to compile the model (default: False).
        verbose: Whether to print verbose output (default: True).
    """
    raw_key: Optional[str] = None
    label_key: Optional[str] = None
    epochs: int = 100
    learning_rate: float = 1e-5
    log_image_interval: int = 10
    n_objects_per_batch: int = 10
    n_iterations: int = 10000
    n_sub_iteration: int = 8
    mixed_precision: bool = True
    compile_model: bool = False
    instance_segmentation: bool = True
    verbose: bool = True

class Trainer(Configurable[TrainerConfig]):
    """Base trainer class for model training.
    
    This is an abstract base class that defines the interface for trainers.
    Subclasses must implement the `run` method.
    
    Attributes:
        config: Trainer configuration.
    """
    def __init__(self, config: TrainerConfig):
        """Initialize the trainer.
        
        Args:
            config: Trainer configuration.
        """
        super().__init__(config)

    @classmethod
    def from_config(cls, config: TrainerConfig) -> "Trainer":
        """Create a Trainer instance from a configuration.

        Args:
            config: Trainer configuration.

        Returns:
            A new Trainer instance.
        """
        return cls(config)

    @task(name="Trainer.run")
    def run(self, image_paths: List[str | Path], label_paths: List[str | Path]) -> None:
        """Run the trainer.
        
        Args:
            image_paths: List of image file paths for training.
            label_paths: List of label file paths corresponding to image_paths.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

class MicroSAMTrainer(Trainer):
    """MicroSAM trainer for fine-tuning Segment Anything models.
    
    This trainer handles the fine-tuning of MicroSAM models using training and validation
    data loaders. It supports mixed precision training, learning rate scheduling, and
    model export.
    
    Attributes:
        config: MicroSAM trainer configuration.
        train_loader: Training data loader.
        val_loader: Validation data loader.
    """
    CHECKPOINTS_FOLDER = "checkpoints"
    BEST_MODEL_NAME = "best.pt"

    def __init__(self, config: MicroSAMTrainerConfig):
        """Initialize the MicroSAM trainer.
        
        Args:
            config: MicroSAM trainer configuration.
        """
        super().__init__(config)
        self.checkpoint_name = str(Path(self.config.export_model_name).stem)
        # Note: These lines reference attributes that may not exist in the config
        # They are commented out or would need to be fixed based on actual config structure
        # self.train_loader = self._get_dataloader(image_paths=self.config.train_image_paths, label_paths=self.config.train_label_paths, split="train", workers=self.config.workers, split_ratio=self.config.split_ratio, raw_key=self.config.raw_key, label_key=self.config.label_key)
        # self.val_loader = self._get_dataloader(image_paths=self.config.val_image_paths, label_paths=self.config.val_label_paths, split="val", workers=self.config.workers, split_ratio=self.config.split_ratio, raw_key=self.config.raw_key, label_key=self.config.label_key)

    def _get_dataloader(
        self, 
        image_paths: List[str | Path], 
        label_paths: List[str | Path], 
        mode: Literal["train", "val"], 
        workers: int = 8
    ) -> torch.utils.data.DataLoader:
        """Return train or validation data loader for fine-tuning SAM.
        
        The data loader must be a torch data loader that returns `x, y` tensors,
        where `x` is the image data and `y` are the labels.
        The labels have to be in a label mask instance segmentation format.
        I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
        Important: the ID 0 is reserved for background, and the IDs must be consecutive.
        Here, we use `torch_em.default_segmentation_loader` for creating a suitable data loader.
        
        Args:
            image_paths: List of image file paths.
            label_paths: List of label file paths corresponding to image_paths.
            mode: Data loader mode, either "train" or "val".
            workers: Number of worker processes for data loading (default: 8).
            
        Returns:
            PyTorch DataLoader for training or validation.
        """
        # We need to define a training and validation split. An easy option is to use some volumes for training
        # and some for validation. Here, we have 4 volumes
        # and use the first complete 3 + lower half of the 4th for training.
        # The upper half of the 4th volume is used for validation.
        train_size = int(self.config.split_ratio * len(image_paths))
        val_size = len(image_paths) - train_size
        if mode == "train":
            # np.s_[:, :, :] means that the full volume is used,
            # np.s_[:35, :, :] that only the lower 35 slices (in this case half of the volume) are used.
            rois = [self.config.roi_def] * train_size
            image_paths = image_paths[:train_size]
            label_paths = label_paths[:train_size]
        else:
            rois = [self.config.roi_def] * val_size
            image_paths = image_paths[train_size:]
            label_paths = label_paths[train_size:]

        raw_transform = v2.Compose([sam_training.identity]) # v2.Compose([self._reshape_array, v2.ToDtype(torch.uint8, scale=True)])
        if self.config.instance_segmentation:
            # Computes the distance transform for objects to perform end-to-end automatic instance segmentation.
            label_transform = v2.Compose([
                PerObjectDistanceTransform(
                distances=True, boundary_distances=True, directed_distances=False,
                foreground=True, instances=True, min_size=25
            )])
        else:
            label_transform = v2.Compose([sam_training.identity])

        loader = torch_em.default_segmentation_loader(
            raw_paths=image_paths,
            raw_key=self.config.raw_key,
            raw_transform=raw_transform,
            label_paths=label_paths, 
            label_key=self.config.label_key,
            label_transform=label_transform,
            patch_shape=self.config.patch_shape, 
            batch_size=self.config.batch_size,
            ndim=2, 
            is_seg_dataset=True, 
            rois=rois,
            shuffle=True, 
            num_workers=workers,
        )
        return loader

    def _reshape_array(self, arr: np.ndarray) -> np.ndarray:
        """Reshape the array to the desired shape.
        
        Args:
            arr: Array to reshape.
        
        Returns:
            Reshaped array.
        """
        return arr.reshape(arr.shape)

    def _contiguous(self, t: torch.Tensor) -> torch.Tensor:
        """Make the tensor contiguous.
        
        Args:
            t: Tensor to make contiguous.
        
        Returns:
            Contiguous tensor.
        """
        return t.contiguous()

    def _finetune_new(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> None:
        """Fine-tune the SAM model.
        
        This method sets up the model, optimizer, scheduler, and trainer, then runs
        the training loop for the specified number of iterations.
        """
        # Run training.
        # Ensure save_root is a Path object
        save_root_path = self.config.save_root if isinstance(self.config.save_root, Path) else Path(self.config.save_root)
        sam_training.train_sam(
            save_root=str(save_root_path.expanduser().resolve()),
            name=self.checkpoint_name,
            model_type=self.config.model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=self.config.epochs,
            n_objects_per_batch=self.config.n_objects_per_batch,
            with_segmentation_decoder=self.config.instance_segmentation,
            device=self.config.device,
            verify_n_labels_in_loader=None
        )

    def _finetune(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader
    ) -> None:
        """Fine-tune the SAM model.
        
        This method sets up the model, optimizer, scheduler, and trainer, then runs
        the training loop for the specified number of iterations.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
        """
        device = torch.device(self.config.device)
        logger.info(f"Using device: {device}")

        # Get the segment anything model, the optimizer and the LR scheduler
        model = sam_training.get_trainable_sam_model(model_type=self.config.model_type, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=self.config.verbose)

        # This class creates all the training data for a batch (inputs, prompts and labels).
        convert_inputs = sam_training.ConvertToSamInputs(transform=self.config.transform, box_distortion_factor=None)

        # the trainer which performs training and validation (implemented using "torch_em")
        # checkpoints will be saved as {self.config.save_root}/checkpoints/{self.checkpoint_name}/*.pt files
        # logs will be saved as {save_root}/logs/{self.checkpoint_name}/event* files
        trainer = sam_training.SamTrainer(
            name=self.checkpoint_name,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            # currently we compute loss batch-wise, else we pass channelwise True
            #loss=torch_em.loss.DiceLoss(channelwise=False),
            #metric=torch_em.loss.DiceLoss(),
            device=device,
            lr_scheduler=scheduler,
            logger=sam_training.SamLogger,
            log_image_interval=self.config.log_image_interval,
            mixed_precision=self.config.mixed_precision,
            convert_inputs=convert_inputs,
            n_objects_per_batch=self.config.n_objects_per_batch,
            n_sub_iteration=self.config.n_sub_iteration,
            compile_model=self.config.compile_model
        )
        trainer.fit(self.config.n_iterations)


    def _export_model(self) -> None:
        """Export the trained model.
        
        Exports the fine-tuned model after training so that it can be used by the rest
        of the micro_sam library. The model is exported from the checkpoint to the
        specified export path.
        """
        # export the model after training so that it can be used by the rest of the micro_sam library
        # Expand ~/ at runtime for portability
        # Ensure paths are Path objects
        save_root_path = self.config.save_root if isinstance(self.config.save_root, Path) else Path(self.config.save_root)
        export_path_path = self.config.export_path if isinstance(self.config.export_path, Path) else Path(self.config.export_path)
        save_root = str(save_root_path.expanduser().resolve())
        export_path = str(export_path_path.expanduser().resolve())
        
        export_custom_sam_model(
            checkpoint_path=os.path.join(save_root, self.CHECKPOINTS_FOLDER, self.checkpoint_name, self.BEST_MODEL_NAME),
            model_type=self.config.model_type,
            save_path=os.path.join(export_path, self.config.export_model_name),
        )

    def validate_image_label_pairs(self, image_paths: List[str | Path], label_paths: List[str | Path]) -> None:
        """Validate the image and label pairs.
        
        Checks that the number of image and label paths match, and that each
        corresponding image and label have the same shape.
        
        Args:
            image_paths: List of image file paths.
            label_paths: List of label file paths corresponding to image_paths.
            
        Raises:
            AssertionError: If the number of paths don't match or if shapes don't match.
        """
        assert len(image_paths) == len(label_paths)
        for image_path, label_path in zip(image_paths, label_paths):
            image = BioImage(image_path)
            label = BioImage(label_path)
            assert image.shape == label.shape
        logger.info(f"Validation of {len(image_paths)} image and label pairs: Passed")
    
    @task(name="MicroSAMTrainer.run")
    def run(self, image_paths: List[str | Path], label_paths: List[str | Path]) -> None:
        """Fine-tune a Segment Anything model.
        
        This method orchestrates the complete fine-tuning process:
        1. Validates image and label pairs
        2. Creates training and validation data loaders
        3. Fine-tunes the model
        4. Exports the trained model
        
        Args:
            image_paths: List of image file paths for training.
            label_paths: List of label file paths corresponding to image_paths.
        """
        logger.info(f"Finetuning MicroSAM model with config: {self.config}")

        # Validate the image and label pairs.
        self.validate_image_label_pairs(image_paths, label_paths)

        # Get the dataloaders.
        train_loader = self._get_dataloader(image_paths, label_paths, mode="train")
        val_loader = self._get_dataloader(image_paths, label_paths, mode="val")

        #t, _ = next(iter(train_loader))
        #v, _ = next(iter(val_loader))
        #logger.debug(f" t, _ = next(iter(train_loader)):{t.shape}, {t.dtype}, min: {t.min()}, max: {t.max()}, t.is_contiguous: {t.is_contiguous()}")
        #logger.debug(f" v, _ = next(iter(val_loader)): {v.shape}, {v.dtype}, min: {v.min()}, max: {v.max()}, v.is_contiguous: {v.is_contiguous()}")

        # Train and validate a finetuned model.
        self._finetune_new(train_loader, val_loader)

        # Export the finetuned model.
        self._export_model()