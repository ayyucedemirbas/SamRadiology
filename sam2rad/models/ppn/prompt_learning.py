from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.transformer import (
    TwoWayAttentionBlock,
)
from sam2.modeling.sam2_utils import MLP
from .registry import register_prompt_predictor


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionEmbeddingSine1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionEmbeddingSine1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


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
        self.pos_encoding_1d = PositionEmbeddingSine1D(embedding_dim)

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
        query_pe = self.pos_encoding_1d(point_embedding)  # B x N_query_tokens x C
        image_embedding = image_embedding.flatten(2).permute(
            0, 2, 1
        )  # B x N_image_tokens x C
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # B x N_image_tokens x C

        sparse_embeddings, dense_embeddings = super().forward(
            queries=point_embedding,
            keys=image_embedding,
            query_pe=query_pe,
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
