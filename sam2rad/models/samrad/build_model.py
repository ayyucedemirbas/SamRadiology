import logging
import warnings

import torch

from sam2rad.decoders.build_decoder import *
from sam2rad.decoders.registry import MASK_DECODER_REGISTRY
from sam2rad.encoders.build_encoder import *
from sam2rad.encoders.registry import IMAGE_ENCODER_REGISTRY
from sam2rad.models.sam.modeling import PromptEncoder

from .model import Model
from .prompt_learning import TwoWayCrossAttention as LightweightAttention
from .prompt_sampler import PromptSampler


def build_model(args):
    """
    Build model by choosing from a range of image encoder and decoders.
    """

    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = args.image_size // vit_patch_size
    args.prompt_embed_dim = prompt_embed_dim
    args.patch_size = vit_patch_size
    args.embed_dim = prompt_embed_dim
    args.r = 32  # If LoRA is used

    # Build encoder
    image_encoder = IMAGE_ENCODER_REGISTRY[args.image_encoder]().build(args)

    try:
        image_encoder.load_checkpoint(args.sam_checkpoint)
        logging.info(f"Loaded SAM image encoder from {args.sam_checkpoint}.")
    except AttributeError:
        # We only need to load pre-trained SAM checkpoint during training
        warnings.warn("No SAM checkpoint loaded. Loading without pre-trained weights.")
        # raise RuntimeError("No SAM checkpoint loaded. Loading without pre-trained weights.")

    # Build decoder
    mask_decoder = MASK_DECODER_REGISTRY[args.mask_decoder]().build(args)
    try:
        mask_decoder.load_checkpoint(args.sam_checkpoint)
    except AttributeError:
        # raise RuntimeError("No SAM checkpoint loaded. Loading without pre-trained weights.")
        warnings.warn("No SAM checkpoint loaded. Loading without pre-trained weights.")

    # Build prompt encoder
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(args.image_size, args.image_size),
        mask_in_chans=16,
    )
    try:
        state_dict = torch.load(args.sam_checkpoint)
        # Extract the prompt encoder state dict from model checkpoint
        state_dict = {
            k.replace("prompt_encoder.", ""): v
            for k, v in state_dict.items()
            if "prompt_encoder" in k
        }
        prompt_encoder.load_state_dict(state_dict)

        logging.info(f"Loaded SAM prompt encoder from {args.sam_checkpoint}.")
    except AttributeError:
        logging.info("No SAM checkpoint loaded. Loading without pre-trained weights.")
        # raise RuntimeError("No SAM checkpoint loaded. Loading without pre-trained weights.")
        warnings.warn("No SAM checkpoint loaded. Loading without pre-trained weights.")

    prompt_sampler = PromptSampler(
        prompt_encoder=prompt_encoder,
        prompt_learner=LightweightAttention(
            prompt_encoder=prompt_encoder,
            embedding_dim=prompt_embed_dim,
            num_heads=1,
            mlp_dim=256,
        ),
    )

    model = Model(
        image_encoder=image_encoder,
        mask_decoder=mask_decoder,
        prompt_sampler=prompt_sampler,
    )

    return model
