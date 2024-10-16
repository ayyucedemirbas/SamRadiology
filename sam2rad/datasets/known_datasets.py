# Description: this file registers commonly used datasets by their names.
import warnings

import kornia as K
import torch
import torch.nn.functional as F

from . import BaseDataset
from .registry import register_dataset

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
        img_orig = self.read_image(self.img_files[index])
        img = img_orig.float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        gt = self.remap_labels(gt).long()
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]
        # convert to one-hot
        gt = F.one_hot(gt.long(), num_classes=self.num_classes + 1).permute(
            2, 0, 1
        )  # (C+1, H, W)
        # remove background class
        gt = gt[1:]  # (C, H, W)

        boxes = self.masks_to_boxes(gt)

        if gt.ndim == 3:
            gt = gt.squeeze(0)

        return (
            img_orig.permute(1, 2, 0),
            img,
            gt,
            input_size,
            boxes,
            self.img_files[index],
        )
