# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch.nn as nn
from functools import partial

from timm.models.deit import _create_deit, VisionTransformerDistilled
from timm.models.vision_transformer import VisionTransformer
# from timm.models.registry import register_model

# __all__ = [
#     'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
#     'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
#     'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
#     'deit_base_distilled_patch16_384',
# ]


# @register_model
# def deit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
#     model_args = dict(
#         patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     )
#     model = _create_deit('deit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def deit_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
#     model_args = dict(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     )
#     model = _create_deit('deit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def deit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
#     model_args = dict(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     )
#     model = _create_deit('deit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def deit_base_patch16_384(pretrained: bool = False, **kwargs) -> VisionTransformer:
#     model_args = dict(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     )
#     model = _create_deit('deit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def deit_tiny_distilled_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformerDistilled:
#     model_args = dict(
#         patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     )
#     model = _create_deit(
#         'deit_tiny_distilled_patch16_224',
#         pretrained=pretrained,
#         distilled=True,
#         **dict(model_args, **kwargs),
#     )
#     return model


# @register_model
# def deit_small_distilled_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformerDistilled:
#     model_args = dict(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     )
#     model = _create_deit(
#         'deit_small_distilled_patch16_224',
#         pretrained=pretrained,
#         distilled=True,
#         **dict(model_args, **kwargs),
#     )
#     return model


# @register_model
# def deit_base_distilled_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformerDistilled:
#     model_args = dict(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     )
#     model = _create_deit(
#         'deit_base_distilled_patch16_224',
#         pretrained=pretrained,
#         distilled=True,
#         **dict(model_args, **kwargs),
#     )
#     return model


# @register_model
# def deit_base_distilled_patch16_384(pretrained: bool = False, **kwargs) -> VisionTransformerDistilled:
#     model_args = dict(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     )
#     model = _create_deit(
#         'deit_base_distilled_patch16_384',
#         pretrained=pretrained,
#         distilled=True,
#         **dict(model_args, **kwargs),
#     )
#     return model
