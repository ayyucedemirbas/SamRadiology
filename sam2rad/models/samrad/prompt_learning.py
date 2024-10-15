import math
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.transformer import (
    TwoWayAttentionBlock,
    TwoWayTransformer,
)
from sam2.modeling.sam2_utils import MLP

PROMPT_PREDICTORS = {}


class BoxRegressionHead(torch.nn.Module):
    """
    Given box embeddings, predict the coordinates of the bounding box.
    """

    def __init__(self, in_features, hidden_dim, num_layers):
        super().__init__()
        self.mlp = MLP(in_features, hidden_dim, 4, num_layers)

    def forward(self, x):
        """
        x.shape: (B, 2, 256)
        """
        x = x.flatten(1)
        xywh = self.mlp(x)
        xywh = torch.sigmoid(xywh)
        x1y1 = xywh[:, :2]
        wh = xywh[:, 2:]
        xyxy = torch.cat([x1y1, x1y1 + wh], dim=1)
        return xyxy


class MaskClassifier(torch.nn.Module):
    """
    Given mask embeddings, predict the binary mask.
    """

    def __init__(self, in_features, hidden_dim, num_layers):
        super().__init__()
        # 4x transposed convolutions
        self.conv = torch.nn.Conv2d(in_features, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: (B, 256, 64, 64)
        """
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear")
        return x


# Source: https://github.com/wzlxjtu/PositionalEncoding2D/blob/d1714f29938970a4999689255f2c230a0b63ebda/positionalembedding2d.py#L24
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe


def register_prompt_predictor(name):
    """
    Register prompt predictor.
    """

    def register_model_cls(cls):
        if name in PROMPT_PREDICTORS:
            raise ValueError(f"Cannot register duplicate model ({name})")
        PROMPT_PREDICTORS[name] = cls
        return cls

    return register_model_cls


class PromptPredictor(nn.Module):
    """
    A prompt predictor to predict prompts from image embeddings.
    """


@register_prompt_predictor("identity")
class IdentityTransformer(PromptPredictor):
    def __init__(self):
        super().__init__()

    def forward(self, point_pe: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return point_pe


@register_prompt_predictor("linear")
class TwoWayCrossAttention(TwoWayAttentionBlock):
    def __init__(
        self,
        prompt_encoder: nn.Module,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__(
            embedding_dim,
            num_heads,
            mlp_dim,
            activation,
            attention_downsample_rate,
            skip_first_layer_pe,
        )
        self.pos_encoding = PositionEmbeddingSine(embedding_dim)

        self.box_regression_head = BoxRegressionHead(
            in_features=256 * 2, hidden_dim=256, num_layers=2
        )

        self.classifier = MaskClassifier(256, 256, 2)
        self.prompt_encoder = prompt_encoder

        self.freeze_pretrained_weights()

    def freeze_pretrained_weights(self):
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        image_embedding: torch.Tensor,
        point_embedding: torch.Tensor,
    ):
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_pe = self.pos_encoding(image_embedding[:1])
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        sparse_embeddings, dense_embeddings = super().forward(
            queries=point_embedding,
            keys=image_embedding,
            query_pe=point_embedding,
            key_pe=image_pe,
        )
        dense_embeddings = dense_embeddings.view(bs, h, w, -1).permute(0, 3, 1, 2)

        # Predict bounding box and mask prompts
        interim_mask_output = self.classifier(dense_embeddings)
        dense_embeddings = self.prompt_encoder._embed_masks(
            interim_mask_output
        )  # (B, 256, H//16, W//16)

        # Encode learned box coordinates
        pred_boxes = self.box_regression_head(sparse_embeddings[:, :2])
        learned_box_embeddings = self.prompt_encoder._embed_boxes(
            boxes=pred_boxes * self.prompt_encoder.input_image_size[0]
        )

        sparse_embeddings = torch.cat(
            [learned_box_embeddings, sparse_embeddings[:, 2:]], dim=1
        )

        return (
            sparse_embeddings,
            dense_embeddings,
            interim_mask_output,
            pred_boxes,
        )


@register_prompt_predictor("transformer")
class TransformerPredictor(TwoWayTransformer):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(
            depth,
            embedding_dim,
            num_heads,
            mlp_dim,
            activation,
            attention_downsample_rate,
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ):
        queries, _ = super().forward(image_embedding, image_pe, point_embedding)
        return queries

    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

        # Extract the mask decoder state dict from model checkpoint
        state_dict = {
            k.replace("sam_mask_decoder.", ""): v
            for k, v in state_dict.items()
            if "sam_mask_decoder" in k
        }

        # Get transformer state dict
        state_dict = {
            k.replace("transformer.", ""): v
            for k, v in state_dict.items()
            if k.startswith("transformer.")
        }
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)

        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")

        if len(missing_keys) > 0:
            print("Missing keys: ", missing_keys)
            raise RuntimeError()

        print(f"Loaded checkpoint from {checkpoint_path}")
