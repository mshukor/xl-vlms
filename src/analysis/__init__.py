import argparse
import os
from typing import Any, Callable, Dict, List, Union

import torch

from analysis.cluster_analysis import analyse_clusters
from analysis.feature_decomposition import (decompose_activations,
                                            get_feature_matrix)
from analysis.model_steering import get_steering_vector
from analysis.multimodal_grounding import get_multimodal_grounding
from analysis.utils import (get_matched_token_of_interest_mask,
                            get_token_of_interest_features)

__all__ = ["load_features", "analyse_features"]

SUPPORTED_ANALYSIS = [
    "decompose_activations",
    "steering_vector",
    "analyse_clusters",
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
    token_of_interest_mask = None
    if args.load_matched_features and len(features_path) > 1:
        token_of_interest_mask = get_matched_token_of_interest_mask(features_path)

    for feat_path in features_path:
        data = torch.load(feat_path, map_location="cpu")
        assert feature_key in data, f"{feature_key} not found, got {data.keys()}."
        data_ = data[feature_key]
        feat = get_feature_matrix(
            data_,
            module_name=args.module_to_decompose,
            token_idx=args.decomposition_extract_pos,
        )
        meta = {k: v for k, v in data.items() if feature_key not in k}
        feat_key = os.path.basename(feat_path)

        if keep_only_token_of_interest:
            if token_of_interest_mask is None:
                feat = get_token_of_interest_features(
                    feat, meta.get("token_of_interest_mask", None)
                )
            else:
                feat = get_token_of_interest_features(feat, token_of_interest_mask)
        features[feat_key] = feat
        meta_data[feat_key] = meta
        if logger is not None:
            logger.info(
                f"Loading data from {feat_path}.\n keys: {data.keys()}.\n Extracted features of shape {feat.shape}"
            )

    return features, meta_data


def load_analysis(
    analysis_path: str,
    logger: Callable = None,
    analysis_keys: List[str] = ["text_grounding"],
    args: argparse.Namespace = None,
) -> List[Dict[str, Any]]:
    data = torch.load(analysis_path, map_location="cpu")
    analysis_data_ = {}
    for analysis_key in analysis_keys:
        assert analysis_key in data, f"{analysis_key} not found, got {data.keys()}."
        analysis_data_[analysis_key] = data[analysis_key]
    if logger is not None:
        logger.info(
            f"Loading data from {analysis_path}.\n Data size: {len(data)}, keys: {data[0].keys()}"
        )
    meta_data = {k: v for k, v in data.items() if k not in analysis_keys}

    return analysis_data_, meta_data


@torch.no_grad()
def analyse_features(
    analysis_name: str = "decompose_activations",
    model_class: Callable = None,
    logger: Callable = None,
    args: argparse.Namespace = None,
    **kwargs: Any,
) -> None:

    if args.features_path is not None and args.base_features_key is not None:
        features, metadata = load_features(
            features_path=args.features_path,
            feature_key=args.base_features_key,
            logger=logger,
            args=args,
        )
    else:
        assert (args.origin_model_feature_path is not None) and (
            args.dest_model_feature_path is not None
        ), "features_path and base_features_key should be provided when analyzing features from a single model"

    num_concepts = [int(n) for n in args.num_concepts] if args.num_concepts else None
    if "decompose_activations" in analysis_name:
        features = list(features.values())[0]
        metadata = list(metadata.values())[0]
        concepts, activations, _ = decompose_activations(
            mat=features,
            num_concepts=num_concepts[0],
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
                save_analysis=args.save_analysis,
                save_dir=args.save_dir,
                save_name=args.save_filename,
                num_grounded_text_tokens=args.num_grounded_text_tokens,
                num_most_activating_samples=args.num_most_activating_samples,
                metadata=metadata,
                logger=logger,
                args=args,
            )

    elif "steering_vector" in analysis_name:
        get_steering_vector(
            features=features,
            steering_method=args.steering_method,
            base_features_key=args.base_features_key,
            num_concepts=num_concepts,
            save_dir=args.save_dir,
            save_name=args.save_filename,
            logger=logger,
            args=args,
        )

    elif "analyse_clusters" in analysis_name:

        features, metadatas = load_features(
            features_path=[
                args.origin_model_feature_path,
                args.dest_model_feature_path,
            ],
            feature_key=args.base_features_key,
            args=args,
        )

        # Load analysis data for origin model if the path is provided, else pass None
        if args.origin_model_analysis_path:
            analysis_data_original, meta_data_original_analysis = load_analysis(
                analysis_path=args.origin_model_analysis_path,
                analysis_keys=[
                    "image_grounding_paths",
                    "text_grounding",
                    "concepts",
                    "activations",
                ],
                args=args,
            )
        else:
            analysis_data_original, meta_data_original_analysis = None, None

        analyse_clusters(
            features=features,
            metadatas=metadatas,
            analysis_data_original=analysis_data_original,
            model_class=model_class,
            analysis_name=analysis_name,
            num_concepts=num_concepts[0],
            save_analysis=args.save_analysis,
            save_dir=args.save_dir,
            save_name=args.save_filename,
            logger=logger,
            args=args,
        )

    else:
        raise NotImplementedError(
            f"Only the following analysis are supported: {SUPPORTED_ANALYSIS}"
        )
