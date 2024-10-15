import torch


def dice_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    smooth: float = 1e-5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Computes the Dice loss for binary segmentation tasks.

    Parameters:
        y_pred (torch.Tensor): Predicted probabilities, expected to be logits.
        y_true (torch.Tensor): Ground truth binary masks.
        smooth (float): Smoothing factor to avoid division by zero.
        reduction (str): Specifies the reduction to apply to the output: 'mean' or 'none'.

    Returns:
        torch.Tensor: The calculated Dice loss. Scalar if reduction is 'mean', otherwise a tensor.
    """
    y_pred = torch.sigmoid(y_pred)
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    y_sum = torch.sum(y_true**2, dim=(1, 2, 3))
    z_sum = torch.sum(y_pred**2, dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (y_sum + z_sum + smooth)
    # CAUTION: If the foreground is empty and the prediction is also empty, the loss should be 0 not 1.
    empty = (y_true.sum(dim=(1, 2, 3)) < 10) & ((y_pred > 0.5).sum(dim=(1, 2, 3)) < 10)
    dice[empty] = 1 - dice[empty]  # Overlap: 0% -> 100%

    return 1 - torch.mean(dice) if reduction == "mean" else 1 - dice
