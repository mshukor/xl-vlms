import argparse
from typing import Any, Callable, Dict

import torch

from analysis.feature_decomposition import decompose_and_ground_activations
from metrics.dictionary_learning_metrics import (
    compute_grounding_words_overlap, get_clip_score)

__all__ = ["concept_dictionary_evaluation"]


def concept_dictionary_evaluation(
    metric_name: str,
    features: Dict[str, torch.Tensor] = None,
    metadata: Dict[str, Any] = {},
    model_class: Callable = None,
    concepts_decomposition_path: str = None,
    args: argparse.Namespace = None,
    logger: Callable = None,
    device=torch.device("cpu"),
    **kwargs: Any,
) -> None:
    scores = {}
    if concepts_decomposition_path is None:
        concepts_dict = decompose_and_ground_activations(
            features,
            metadata,
            analysis_name="decompose_activations_image_grounding_text_grounding",
            model_class=model_class,
            logger=logger,
            args=args,
        )
    else:
        concepts_dict = torch.load(concepts_decomposition_path)

    if logger is not None:
        # log info about concept dictionary
        logger.info(f"Concept dictionary is decomposition type: {concepts_dict['decomposition_method']}")
        logger.info(f"Number of concepts in given concept dictionary: {concepts_dict['concepts'].shape[0]}")

    if "clipscore" in metric_name:
        clipscore_dict = get_clip_score(
            features,
            metadata,
            concepts_dict=concepts_dict,
            model_class=model_class,
            device=device,
            logger=logger,
            args=args,
        )
        scores.update(clipscore_dict)

    if "bertscore" in metric_name:
        logger.info("BERTScore under construction")

    if "overlap" in metric_name:
        grounding_words = concepts_dict["text_grounding"]
        overlap_scores = compute_grounding_words_overlap(grounding_words, logger=logger)
        scores.update(overlap_scores)
    return scores
