import argparse
import logging
import math
import os
import random
import sys
from functools import lru_cache
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import termcolor
import torch.utils
import torchvision.ops as ops
import yaml
from torchmetrics import Dice
from torchmetrics.classification import MulticlassJaccardIndex as IoU

from sam2rad.blob.main.misc import AverageMeter, DotDict

# fmt: off
sys.path.insert(0, os.path.abspath('sam2rad/models'))
print(sys.path)
# fmt: on
from functools import partial
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from sam2rad.datasets import DATASETS, get_dataloaders
from sam2rad.losses import CompositeLoss, dice_loss, focal_loss
from sam2rad.models.sam2rad.build_model import build_model as build_sam2rad
from sam2rad.models.samrad.build_model import build_model as build_samrad

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])


# Pytorch verbose error messages
# torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser(description="Train a segmentation model")
parser.add_argument("--model_config", type=str, help="Path to model config file")
parser.add_argument("--dataset_config", type=str, help="Path to dataset config file")
parser.add_argument("--trainer_config", type=str, help="Path to trainer config file")


class SavePredictionsCallback(pl.Callback):
    """
    A PyTorch Lightning callback to save and visualize predictions during training and validation.
    """

    def __init__(self):
        self.val_outputs = []
        self.train_outputs = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 10:
            self.train_outputs.append(outputs)

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 10:
            self.val_outputs.append(outputs)

        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def get_random_color(cls: int):
        return tuple(random.randint(0, 255) for _ in range(3))

    def overlay_contours(self, img: np.array, mask: np.array):
        unique_classes = np.unique(mask)

        for cls in unique_classes:
            if cls == 0:  # Assuming 0 is the background class
                continue

            binary_mask = (mask == cls).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            color = self.get_random_color(cls)
            cv2.drawContours(img, contours, -1, color, 2)

        return img

    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = self.val_outputs
        pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
        gt = torch.cat([o["target_masks"] for o in outputs], dim=0)
        images = torch.cat([o["images"] for o in outputs], dim=0)
        fig, axes = plt.subplots(pred.size(0), 2, figsize=(2 * 4, pred.size(0) * 4))
        for i, (p, g, img) in enumerate(zip(pred, gt, images)):
            img_gt = self.overlay_contours(
                img.cpu().numpy().astype(np.uint8), g.cpu().numpy().astype(np.uint8)
            )
            img_pred = self.overlay_contours(
                img.cpu().numpy().astype(np.uint8), p.cpu().numpy().astype(np.uint8)
            )
            axes[i, 0].imshow(img_gt)
            axes[i, 0].imshow(g.cpu(), alpha=0.2)
            axes[i, 1].imshow(img_pred)
            axes[i, 1].imshow(p.cpu(), alpha=0.2)

        plt.savefig("debug_val_progress.png", bbox_inches="tight")
        plt.close()

        self.val_outputs.clear()

        return super().on_validation_epoch_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = self.train_outputs
        if len(outputs) > 0:
            pred = torch.cat([o["pred_masks"] for o in outputs], dim=0)
            gt = torch.cat([o["target_masks"] for o in outputs], dim=0)
            images = torch.cat([o["images"] for o in outputs], dim=0)

            fig, axes = plt.subplots(pred.size(0), 2, figsize=(2 * 4, pred.size(0) * 4))

            for i, (p, g, img) in enumerate(zip(pred, gt, images)):
                img_gt = self.overlay_contours(
                    img.cpu().numpy().astype(np.uint8), g.cpu().numpy().astype(np.uint8)
                )
                img_pred = self.overlay_contours(
                    img.cpu().numpy().astype(np.uint8), p.cpu().numpy().astype(np.uint8)
                )
                axes[i, 0].imshow(img_gt)
                axes[i, 0].imshow(g.cpu(), alpha=0.2)
                axes[i, 1].imshow(img_pred)
                axes[i, 1].imshow(p.cpu(), alpha=0.2)

                # Contours

            plt.savefig("debug_train_progress.png", bbox_inches="tight")
            plt.close()

            self.train_outputs.clear()

        return super().on_train_epoch_end(trainer, pl_module)


class ConcatDataloaders:
    """
    Fetches batch from multiple dataloaders in an alternating fashion.
    """

    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def get_next_batch(self, dataloaders, every_iters=5):
        """
        Fetches the next batch from one of the dataloaders.
        """
        _dataloader_iters = [iter(dl) for dl in dataloaders]
        total_iters = sum(len(dl) for dl in self.dataloaders)

        dl_index = 0
        for step in range(total_iters):
            # Select datasets in an alternating fashion, such as 0, 1, 2, 0, 1, 2, 0, ...
            if (step + 1) % every_iters == 0:
                dl_index = (dl_index + 1) % len(dataloaders)  # 0, 1 % len = 2

            try:
                batch = next(_dataloader_iters[dl_index])
            except StopIteration:
                _dataloader_iters[dl_index] = iter(
                    self.dataloaders[dl_index]
                )  # Reset the iterator for the current dataloader
                batch = next(_dataloader_iters[dl_index])

            yield {"data": batch, "dataloader_idx": dl_index}

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    def __iter__(self):
        for batch in self.get_next_batch(self.dataloaders):
            yield batch


def build_model(config):
    if "sam2" in config.image_encoder:
        return build_sam2rad(config)

    return build_samrad(config)


class SegmentationModule(torch.nn.Module):
    """
    Combines segment anything with learnable prompts.
    """

    def __init__(
        self,
        config,
        prompts: Dict[str, torch.nn.Parameter],
    ):
        super(SegmentationModule, self).__init__()
        self.model = build_model(config)
        # Sometimes use manual prompts only (box, mask, etc.) so that the predicted prompts align with manual prompts.
        self.model.prompt_sampler.p[0] = 0.9  # Learned prompts
        # If box or mask prompt is used during training, the model can be prompted to correct a prediction by providing a box or mask prompt (human-in-the-loop)
        self.model.prompt_sampler.p[2] = 0.5  # Box
        self.model.prompt_sampler.p[3] = 0.1  # Mask

        self.dataset_names = list(prompts.keys())
        self.learnable_prompts = torch.nn.ParameterDict(prompts)

        # TODO: fix this. It breaks if there are multiple datasets because the number of classes is different for each dataset.
        self.num_classes = self.learnable_prompts[self.dataset_names[0]].size(0)

    def forward(self, batch, dataset_index, inference=False):
        """Get the learnable prompts for the dataset and make predictions"""
        imgs = batch["images"]
        prompts = self.learnable_prompts[self.dataset_names[dataset_index]].to(
            imgs.device
        )  # (num_classes, num_tokens, 256)

        outputs = self.model(batch, prompts, inference=inference)
        return outputs


class Learner(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        lr: List[float],
        pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.pixel_mean = torch.tensor(pixel_mean).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(1, 3, 1, 1)

        self.label_smoothing = 0.1
        self.image_size = (
            self.model.model.prompt_sampler.prompt_encoder.input_image_size[0]
        )

        self.train_dice_metric = AverageMeter()
        self.train_iou_metric = AverageMeter()
        self.val_dice_metric = AverageMeter()
        self.val_iou_metric = AverageMeter()

        self.iou = IoU(
            num_classes=self.model.num_classes + 1,
            ignore_index=0,  # ignore background
            average="micro",
        )

        self.dice = Dice(
            num_classes=self.model.num_classes + 1,
            ignore_index=0,  # ignore background
            average="micro",
        )

    def on_validation_epoch_end(self) -> None:
        self.val_dice_metric.reset()
        self.val_iou_metric.reset()
        return super().on_validation_epoch_end()

    def on_train_epoch_end(self) -> None:
        self.train_dice_metric.reset()
        self.train_iou_metric.reset()
        return super().on_train_epoch_end()

    @staticmethod
    def generalized_box_iou_loss(pred_boxes, target_boxes, ignore_boxes=None):
        """
        Generalized box iou loss.
        pred_boxes: (B, 4) x1, y1, x2, y2
        target_boxes: (B, 4) x1, y1, x2, y2
        """
        if ignore_boxes is None:
            ignore_boxes = torch.zeros_like(pred_boxes).bool()
        loss = ops.generalized_box_iou_loss(pred_boxes, target_boxes, reduction="none")
        loss = (loss * (1 - ignore_boxes)).sum() / (1 - ignore_boxes).sum()

        return loss

    @staticmethod
    def reshape_inputs(batch):
        batch["boxes"] = batch["boxes"].reshape(-1, 4)
        batch["boxes_normalized"] = batch["boxes_normalized"].reshape(-1, 4)
        batch["ignore"] = batch["ignore"].reshape(-1)
        lr_masks = batch["low_res_masks"]
        batch["low_res_masks"] = lr_masks.reshape(
            -1, 1, lr_masks.size(2), lr_masks.size(3)
        )
        masks = batch["masks"]
        batch["masks"] = masks.reshape(-1, 1, masks.size(2), masks.size(3))

        return batch

    def training_step(self, batch, batch_idx):
        dataloader_idx = batch["dataloader_idx"]
        b, c, h, w = batch["data"]["masks"].shape
        batch = self.reshape_inputs(batch["data"])
        gt = batch["masks"].float()  # (B*C, 1, H, W)
        outputs = self.model(batch, dataloader_idx)
        pred = outputs["pred"]
        loss_seg = self.loss_fn(pred, gt)  # (B,)
        # Make prediction for non-empty masks only
        is_non_empty = (gt.sum(dim=(1, 2, 3)) > 10).float()
        loss_seg = (loss_seg * is_non_empty).sum() / is_non_empty.sum()
        # Bounding box regression loss
        loss_box = 0.0
        if outputs["pred_boxes"] is not None:
            pred_boxes = outputs["pred_boxes"]  # x1, y1, x2, y2
            target_boxes = batch["boxes_normalized"]  # x1, y1, x2, y2
            ignore_boxes = batch["ignore"].float()
            loss_box = self.generalized_box_iou_loss(
                pred_boxes, target_boxes, ignore_boxes
            )

        # Object prediction head
        object_score_logits = torch.clip(
            outputs["object_score_logits"].view(-1), -10, 10
        )
        if self.label_smoothing > 0:
            target = (
                is_non_empty * (1 - self.label_smoothing) + self.label_smoothing / 2
            )

        else:
            target = is_non_empty

        loss_object = F.binary_cross_entropy_with_logits(object_score_logits, target)

        interim_mask_loss = 0.0
        if outputs["interim_mask_output"] is not None:
            interim_mask_loss = ops.sigmoid_focal_loss(
                outputs["interim_mask_output"], gt, reduction="none", alpha=0.6, gamma=3
            )

            interim_mask_loss = interim_mask_loss.mean(dim=(1, 2, 3))
            interim_mask_loss = (
                interim_mask_loss * is_non_empty
            ).sum() / is_non_empty.sum()

        train_loss = loss_seg + loss_object + loss_box + 100 * interim_mask_loss

        # Compute metrics
        _pred = pred.clone().detach()
        _pred[object_score_logits < 0] = -1
        pred_semantic = self.to_semantic(_pred.detach().view(b, c, h, w))
        gt_semantic = self.to_semantic(gt.view(b, c, h, w))

        self.train_dice_metric.update(self.dice(pred_semantic, gt_semantic), b)
        self.train_iou_metric.update(self.iou(pred_semantic, gt_semantic), b)

        self.log_dict(
            {
                "train_loss_seg": loss_seg,
                "interim_mask_loss": interim_mask_loss,
                "train_loss_box": loss_box,
                "train_loss_object": loss_object,
                "train_iou": self.train_iou_metric.get_avg(),
                "train_dice": self.train_dice_metric.get_avg(),
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "loss": train_loss,
            "iou": self.train_iou_metric.get_avg(),
            "dice": self.train_dice_metric.get_avg(),
            "confidence": object_score_logits,
            "images": self.denormalize(batch["images"]),
            "target_masks": gt_semantic,
            "pred_masks": pred_semantic,
            # DEBUG
            "pred_boxes": outputs["pred_boxes"],
            "interim_mask_output": outputs["interim_mask_output"],
            "gt_boxes": batch["boxes_normalized"],
        }

    def denormalize(self, img):
        img = img * self.pixel_std.to(img.device) + self.pixel_mean.to(img.device)
        return (img * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu()

    @staticmethod
    def to_semantic(mask: torch.Tensor):
        """
        Convert a multi-channel mask to a semantic segmentation mask.

        Args:
        mask: (B, C, H, W) - A PyTorch tensor where B is batch size, C is number of classes,
                            H is height, and W is width.

        Returns:
        sem_mask: (B, H, W) - A PyTorch tensor representing the semantic segmentation mask.
        """

        # Get the class with the highest probability for each pixel (adding 1 to account for ignored background)
        sem_mask = mask.argmax(dim=1) + 1

        # Create a foreground mask
        fg = (mask > 0).any(dim=1).float()

        # Apply the foreground mask to sem_mask
        sem_mask = sem_mask * fg

        return sem_mask.long()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if "data" in batch:
            batch = batch["data"]

        b, c, h, w = batch["masks"].shape
        batch = self.reshape_inputs(batch)
        gt = batch["masks"].float()  # (B, C, H, W)
        outputs = self.model(batch, dataloader_idx, inference=True)
        pred = outputs["pred"]
        loss_seg = self.loss_fn(pred, gt)  # (B,)
        # train on non-empty masks only
        is_non_empty = (gt.sum(dim=(1, 2, 3)) > 1).float()
        # loss_seg = (loss_seg * is_non_empty).sum() / is_non_empty.sum()
        loss_seg = loss_seg.mean()
        object_score_logits = outputs["object_score_logits"].view(-1)

        loss_object = F.binary_cross_entropy_with_logits(
            object_score_logits, is_non_empty
        )

        pred[object_score_logits < 0] = -1
        pred_semantic = self.to_semantic(pred.detach().view(b, c, h, w))
        gt_semantic = self.to_semantic(gt.view(b, c, h, w))

        self.val_dice_metric.update(self.dice(pred_semantic, gt_semantic), b)
        self.val_iou_metric.update(self.iou(pred_semantic, gt_semantic), b)

        # log the loss and metrics
        self.log_dict(
            {
                "val_loss_seg": loss_seg,
                "val_loss_object": loss_object,
                "val_iou": self.val_iou_metric.get_avg(),
                "val_dice": self.val_dice_metric.get_avg(),
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "images": self.denormalize(batch["images"]),
            "target_masks": gt_semantic,
            "pred_masks": pred_semantic,
            "confidence": object_score_logits,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=trainer_config.max_epochs, eta_min=1e-5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class DataModule(pl.LightningDataModule):
    """
    This will automatically handle multi-GPU training and distributed data loading.
    """

    def __init__(self, train_loader, *val_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.val_loader

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Training data
        if "data" in batch:
            for key, value in batch["data"].items():
                if isinstance(value, torch.Tensor):
                    batch["data"][key] = value.to(device)

        else:
            # Validation data
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

        return batch


if __name__ == "__main__":
    args = parser.parse_args()
    # Get dataset object
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
        dataset_config = DotDict.from_dict(dataset_config)

    # build model
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
        model_config = DotDict.from_dict(model_config)

    # Hyperparameter configs
    with open(args.trainer_config) as f:
        trainer_config = yaml.safe_load(f)
        trainer_config = DotDict.from_dict(trainer_config)

    # Register a custom dataset or use a default one, e.g., dataset_obj = DATASETS["default_segmentation"]
    default_dataset = DATASETS["default_segmentation"]

    # Get dataloaders
    learnable_prompts = {}
    trn_dls, val_dls = [], []
    for dataset in trainer_config["datasets"]:
        # Register a custom dataset or use a default one, e.g., dataset_obj = DATASETS["default_segmentation"]
        dataset_obj = (
            DATASETS[dataset] if dataset in DATASETS else default_dataset
        )  # if the dataset is not registered, use the default dataset
        trn_ds, val_ds = dataset_obj.from_path(dataset_config.get(dataset))
        val_ds = torch.utils.data.Subset(val_ds, range(0, 100))

        # Get dataloaders
        trn_dl = get_dataloaders(dataset_config[dataset], trn_ds)
        val_dl = get_dataloaders(dataset_config[dataset], val_ds)

        # Initialize learnable prompts for each dataset
        class_tokens = torch.nn.Parameter(
            torch.randn(
                dataset_config[dataset]["num_classes"],
                dataset_config[dataset]["num_tokens"],
                256,
            )
            / math.sqrt(256)
        )

        learnable_prompts[dataset] = class_tokens

        trn_dls.append(trn_dl)
        val_dls.append(val_dl)

    # Concatenate dataloaders
    concat_dl = ConcatDataloaders(*trn_dls)

    print(f"Train dataset size: {len(trn_dl.dataset)}")
    print(f"Validation dataset size: {len(val_dl.dataset)}")
    datamodule = DataModule(concat_dl, *val_dls)
    # loss functions
    loss_fn = CompositeLoss(
        [
            partial(dice_loss, reduction="none"),
            partial(focal_loss, reduction="none", alpha=0.7, gamma=3),
        ],
        weights=torch.tensor([1.0, 10.0]),
    )

    model = SegmentationModule(model_config, learnable_prompts)
    print(model)
    termcolor.colored("Trainable parameters:", "red")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(termcolor.colored(f"{name} | {param.size()}", "red"))

    learner = Learner(model, loss_fn=loss_fn, lr=trainer_config["lr"])

    # Log model if validation accuracy increases
    wandb_logger = WandbLogger(log_model=False, project="Sam2Med", config=vars(args))

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",  # TODO: the validation dice should be averaged across all datasets. https://github.com/Lightning-AI/pytorch-lightning/discussions/5793
        dirpath="checkpoints"
        if trainer_config.get("save_path") is None
        else trainer_config.get("save_path"),
        save_last=True,
        filename="model_{epoch:02d}-{val_dice:.2f}",
        save_top_k=3,
        mode="max",
    )

    # Watch gradients
    # wandb_logger.watch(model, log="gradients", log_freq=100, log_graph=False)

    trainer = pl.Trainer(
        max_epochs=trainer_config["max_epochs"],
        enable_progress_bar=True,
        gradient_clip_val=10,
        check_val_every_n_epoch=10,
        log_every_n_steps=10,
        # strategy="ddp_find_unused_parameters_true",  #
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            SavePredictionsCallback(),
        ],
        # profiler="simple",
        accelerator="gpu",  # run on all available GPUs
    )

    trainer.fit(
        learner,
        datamodule=datamodule,
        ckpt_path=trainer_config.get("resume"),
    )
