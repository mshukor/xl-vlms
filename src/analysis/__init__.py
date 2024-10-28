import argparse
import os
from typing import Any, Callable, Dict, List, Union

import torch

from analysis.feature_decomposition import (decompose_and_ground_activations,
                                            get_feature_matrix)
from analysis.utils import get_token_of_interest_features
from metrics import concept_dictionary_evaluation

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
            logger.info(f"Loading features from: {feat_path}")
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
    metadata: Dict[str, Any] = {},
    analysis_name: str = "decompose_activations",
    model_class: Callable = None,
    logger: Callable = None,
    device: torch.device = torch.device("cpu"),
    args: argparse.Namespace = None,
    **kwargs: Any,
) -> None:
    if "decompose_activations" in analysis_name:
        results_dict = decompose_and_ground_activations(
            features,
            metadata,
            analysis_name=analysis_name,
            model_class=model_class,
            logger=logger,
            args=args,
        )
    elif "concept_dictionary_evaluation" in analysis_name:
        results_dict = concept_dictionary_evaluation(
            metric_name=analysis_name,
            features=features,
            metadata=metadata,
            model_class=model_class,
            concepts_decomposition_path=args.concepts_decomposition_path,
            logger=logger,
            args=args,
            device=device,
        )
    else:
        raise NotImplementedError(
            f"Only the following analysis are supported: {SUPPORTED_ANALYSIS}"
        )
    if results_dict:
        file_name = os.path.join(
            args.save_dir, f"{analysis_name}_{args.save_filename}.pth"
        )
        torch.save(results_dict, file_name)
        if logger is not None:
            logger.info(f"Saving analysis results to: {file_name}")
