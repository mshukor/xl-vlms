import argparse
import os
from typing import Any, Callable, Dict, List, Union

import torch

import metrics
from analysis.cluster_analysis import analyse_clusters
from analysis.feature_decomposition import (decompose_and_ground_activations,
                                            get_feature_matrix)
from analysis.model_steering import get_steering_vector
from analysis.learnable_model_steering import LearnableSteering
from analysis.utils import (get_matched_token_of_interest_mask,
                            get_token_of_interest_features)

__all__ = ["load_features", "analyse_features"]

SUPPORTED_ANALYSIS = [
    "decompose_activations",
    "steering_vector",
    "analyse_clusters",
]



def model_name(
    model_name_or_path: str = None,
) -> str:
    if model_name_or_path=="llava-hf/llava-1.5-7b-hf":
        return "llava"
    else:
        NotImplementedError


def load_features(
    features_path: Union[str, List[str]],
    logger: Callable = None,
    feature_key: str = "hidden_states",
    args: argparse.Namespace = None,
    keep_only_token_of_interest: bool = True,
) -> List[Dict[str, Any]]:

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


def load_features_helper(
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Any:
    
    if args.features_path is not None:
        features, metadata = load_features(
            features_path=args.features_path,
            logger=logger,
            args=args,
        )
    else:
        assert (args.origin_model_feature_path is not None) and (
            args.dest_model_feature_path is not None
        ), "features_path and base_features_key should be provided when analyzing features from a single model"

        features, metadata = load_features(
            features_path=[
                args.origin_model_feature_path,
                args.dest_model_feature_path,
            ],
            args=args,
        )
    

    return features, metadata



    

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
    device: torch.device = torch.device("cpu"),
    args: argparse.Namespace = None,
    **kwargs: Any,
) -> None:

    
    num_concepts = [int(n) for n in args.num_concepts] if args.num_concepts else None
    results_dict = {}
    if "decompose_activations" in analysis_name:

        features, metadata = load_features_helper(logger=logger, args=args)

        results_dict = decompose_and_ground_activations(
            features,
            metadata,
            analysis_name=analysis_name,
            model_class=model_class,
            logger=logger,
            args=args,
        )
    elif "concept_dictionary_evaluation" in analysis_name:

        features, metadata = load_features_helper(logger=logger, args=args)

        results_dict = metrics.concept_dictionary_evaluation(
            metric_name=analysis_name,
            features=features,
            metadata=metadata,
            model_class=model_class,
            concepts_decomposition_path=args.analysis_saving_path,
            logger=logger,
            args=args,
            device=device,
        )
    elif "steering_vector" in analysis_name:

        features, metadata = load_features_helper(logger=logger, args=args)

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

    elif "learnable_steering" in analysis_name:

        pos_path = [p for p in args.features_path if "pos" in p][0]
        neg_path = [p for p in args.features_path if "neg" in p][0]

        model_name_str = model_name(args.model_name_or_path)

        learnable_steering = LearnableSteering(
            pos_path=pos_path,
            neg_path=neg_path,
            module=args.modules_to_hook[0][0],
            shift_type=args.shift_type,
            save_dir=args.save_dir,
            save_name=args.save_filename,
            model_name=model_name_str,
            model_class=model_class,
            args=args,
            logger=logger,
        )

        learnable_steering.compute_contrastive_vectors()

        if "train" in pos_path:
            with torch.enable_grad():
                learnable_steering.train_model()


    elif "analyse_clusters" in analysis_name:

        features, metadata = load_features_helper(logger=logger, args=args)

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
            metadatas=metadata,
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
    if results_dict:
        file_name = os.path.join(
            args.save_dir, f"{analysis_name}_{args.save_filename}.pth"
        )
        torch.save(results_dict, file_name)
        if logger is not None:
            logger.info(f"Saving analysis results to: {file_name}")
