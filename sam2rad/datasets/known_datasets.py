# Description: this file registers commonly used datasets by their names.
import warnings
from pathlib import Path

import kornia as K
import torch
from sklearn.model_selection import train_test_split
from torchvision import io

from sam2rad.datasets import BaseDataset
from sam2rad.datasets.main import group_files_by_patient_id
from sam2rad.datasets.registry import register_dataset

warnings.filterwarnings("ignore")


@register_dataset("default_segmentation")
class SegmentationDataset(BaseDataset):
    def __init__(self, img_files, gt_files, mode, config):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = None


@register_dataset("shoulder")
class SegmentationDataset(BaseDataset):
    def __init__(self, img_files, gt_files, mode, config):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.2),
            K.augmentation.RandomVerticalFlip(p=0.2),
            K.augmentation.RandomAffine(degrees=90, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("sonance")
class WristDataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Define custom augmentations here
        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.4),
            K.augmentation.RandomVerticalFlip(p=0.4),
            K.augmentation.RandomAffine(degrees=30, translate=(0.0, 0.0), p=0.4),
            data_keys=["input", "mask"],
        )

    @staticmethod
    def split_train_test(file_names, test_size, seed=None):
        """
        Group files by patient id and split them into train and validation sets.
        """
        patient_id_2_filename = group_files_by_patient_id(file_names)
        patient_ids = list(patient_id_2_filename.keys())

        train_ids, val_ids = train_test_split(
            patient_ids, test_size=test_size, random_state=seed
        )
        train_files, val_files = [], []
        for _id in train_ids:
            train_files.extend(patient_id_2_filename[_id])

        for _id in val_ids:
            val_files.extend(patient_id_2_filename[_id])

        return train_files, val_files

    @classmethod
    def from_path(cls, config, mode="Train"):
        """
        Dataset object from a directory containing images and masks.
        """

        path = Path(config.root)
        mask_file_names = [f for f in path.glob(f"{mode}/gts/*")]
        mask_file_names.sort()

        trn_mask_file_names, val_mask_file_names = cls.split_train_test(
            mask_file_names, 1 - config.split, seed=config.seed
        )

        trn_img_file_names = [
            cls.get_corresponding_image_name(str(f)) for f in trn_mask_file_names
        ]

        val_img_file_names = [
            cls.get_corresponding_image_name(str(f)) for f in val_mask_file_names
        ]

        return (
            cls(trn_img_file_names, trn_mask_file_names, config, mode="Train"),
            cls(val_img_file_names, val_mask_file_names, config, mode="Val"),
        )


@register_dataset("wrist")
class WristScans(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Define custom augmentations here
        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomAffine(degrees=90, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
         
        )

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("hip")
class Hip(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("3dus_chop")
class Hip3DChopDataset(BaseDataset):
    # Map RGB values to class IDs
    label_to_class_id = {
        (255, 0, 0): 1,  # Red -> Class 0
        (0, 255, 0): 2,  # Green -> Class 1
        (0, 0, 255): 3,  # Blue -> Class 2
    }

    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )

    @staticmethod
    def remap_labels(gt: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Remap RGB labels to class IDs.
        Assumes `gt` is a tensor of shape (3, H, W) or (
        H, W, 3) where each pixel is an RGB tuple.
        """
        # If the input is in shape (3, H, W), we need to permute it to (H, W, 3)
        if gt.shape[0] == 3:
            gt = gt.permute(1, 2, 0)  # Change from (3, H, W) to (H, W, 3)

        # Now proceed with the original logic
        class_map = torch.zeros(gt.shape[:2], dtype=torch.long)
        for color, class_id in Hip3DChopDataset.label_to_class_id.items():
            match = (gt == torch.tensor(color, dtype=gt.dtype)).all(dim=-1)
            class_map[match] = class_id
        return class_map


@register_dataset("3dus_chop_test")
class ChopTestDataset(Hip3DChopDataset):
    """
    Returns images without any augmentation.
    """

    def __getitem__(self, index):
        img = io.read_image(self.img_files[index]).float() / 255.0
        gt = io.read_image(self.gt_files[index])
        gt = self.remap_labels(gt).long()
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]

        return img, gt, input_size, self.img_files[index]
