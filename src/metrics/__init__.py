import argparse
import os
from functools import partial
from typing import Any, Callable, Dict

import torch

from metrics.captioning_metrics import compute_captioning_metrics
from metrics.hallucination_metrics import compute_hallucination_metrics
from metrics.dictionary_learning_metrics import (
    compute_grounding_words_overlap, get_clip_score)
from metrics.vqa_accuracy import compute_vqav2_accuracy

__all__ = ["get_metric"]


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
        logger.info(
            f"Concept dictionary is decomposition type: {concepts_dict['decomposition_method']}"
        )
        logger.info(
            f"Number of concepts in given concept dictionary: {concepts_dict['concepts'].shape[0]}"
        )

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


def get_metric(
    metric_name: str,
    args: argparse.Namespace = None,
    **kwargs: Any,
) -> Callable:
    metric = None

    save_filename = (
        os.path.join(args.save_dir, f"{metric_name}_{args.save_filename}.json")
        if args.save_filename and metric_name is not None
        else None
    )
    if "vqav2_accuracy" in metric_name:
        metric = partial(
            compute_vqav2_accuracy,
            token_of_interest=args.token_of_interest,
            category_of_interest=args.category_of_interest,
            answer_type_to_answer=args.answer_type_to_answer,
            preds_token_of_interests=args.predictions_token_of_interest,
            targets_token_of_interests=args.targets_token_of_interest,
            save_filename=save_filename,
            save_predictions=args.save_predictions,
            predictions_path=args.predictions_path,
        )
    elif "captioning_metrics" in metric_name:
        metric = partial(
            compute_captioning_metrics,
            metrics=args.captioning_metrics,
            preds_token_of_interests=args.predictions_token_of_interest,
            targets_token_of_interests=args.targets_token_of_interest,
            save_filename=save_filename,
            save_predictions=args.save_predictions,
            predictions_path=args.predictions_path,
        )

    elif "hallucination_metrics" in metric_name:
        metric = partial(
            compute_hallucination_metrics,
            save_filename=save_filename,
            save_predictions=args.save_predictions,
            predictions_path=args.predictions_path,
            model_name=args.model_name_or_path,
        )

    return metric
