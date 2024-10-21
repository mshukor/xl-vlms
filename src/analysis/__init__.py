import argparse
import os
from typing import Any, Callable, Dict, List, Union

import torch

from analysis.feature_decomposition import (decompose_activations,
                                            get_feature_matrix)
from analysis.multimodal_grounding import get_multimodal_grounding
from analysis.utils import get_token_of_interest_features

__all__ = ["load_features", "analyse_features"]

SUPPORTED_ANALYSIS = [
    "decompose_activations",
]


def load_features(
    features_path: Union[str, List[str]],
    logger: Callable = None,
    feature_key: str = "hidden_states",
    args: argparse.Namespace = None,
    keep_only_token_of_interest: bool = True,
) -> List[Dict[str, Any]]:  #

    if isinstance(features_path, str):
        features_path = [features_path]
    features = {}
    meta_data = {}

    for feat_path in features_path:
        if logger is not None:
            logger.info(f'Loading features from: {feat_path}')
        data = torch.load(feat_path, map_location="cpu")
        assert feature_key in data, f"{feature_key} not found, got {data.keys()}."
        data_ = data[feature_key]
        feat = get_feature_matrix(
            data_, module_name=args.module_to_decompose, args=args
        )
        meta = {k: v for k, v in data.items() if feature_key not in k}
        feat_key = os.path.basename(feat_path)

        if keep_only_token_of_interest:
            feat = get_token_of_interest_features(
                feat, meta.get("token_of_interest_mask", None)
            )
        features[feat_key] = feat
        meta_data[feat_key] = meta
        if logger is not None:
            logger.info(
                f"Loading data from {feat_path}.\n keys: {data.keys()}.\n Extracted features of shape {feat.shape}"
            )

    return features, meta_data


@torch.no_grad()
def analyse_features(
    features: Dict[str, torch.Tensor],
    analysis_name: str = "decompose_activations",
    model_class: Callable = None,
    logger: Callable = None,
    metadata: Dict[str, Any] = {},
    args: argparse.Namespace = None,
    **kwargs: Any,
) -> None:
    if "decompose_activations" in analysis_name:
        features = list(features.values())[0]
        metadata = list(metadata.values())[0]
        concepts, activations, _ = decompose_activations(
            mat=features,
            num_concepts=args.num_concepts,
            decomposition_method=args.decomposition_method,
            args=args,
        )
        if logger is not None:
            logger.info(
                f"\nDecomposition type {args.decomposition_method}, Components/concepts shape: {concepts.shape}, Activations shape: {activations.shape}"
            )
        if "grounding" in analysis_name:
            text_grounding = "text_grounding" in analysis_name
            image_grounding = "image_grounding" in analysis_name
            grounding_dict = get_multimodal_grounding(
                concepts=concepts,
                activations=activations,
                model_class=model_class,
                text_grounding=text_grounding,
                image_grounding=image_grounding,
                module_to_decompose=args.module_to_decompose,
                save_mas_dir=args.save_dir,
                num_grounded_text_tokens=args.num_grounded_text_tokens,
                num_most_activating_samples=args.num_most_activating_samples,
                metadata=metadata,
                logger=logger,
            )
    else:
        raise NotImplementedError(
            f"Only the following analysis are supported: {SUPPORTED_ANALYSIS}"
        )
