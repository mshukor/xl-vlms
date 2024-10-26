import argparse
from functools import partial
from typing import Any, Callable

import torch

from metrics.clipscore_overlap import compute_clipscore, compute_overlap

__all__ = ["dictionary_learning_evaluation"]


def dictionary_learning_evaluation(
    metric_name: str,
    logger: Callable = None,
    args: argparse.Namespace = None,
    **kwargs: Any,
) -> None:
    concepts_dict = torch.load(args.decomposition_path)
    if "overlap" in metric_name:
        grounded_words = concepts_dict['text_grounding']
        overlap_metric, _ = compute_overlap(grounded_words)
        logger.info(f"Overlap metric: {overlap_metric: .3f}")
        
    if "clipscore" in metric_name:
        logger.info("CLIPScore under construction")
        
    if "bertscore" in metric_name:
        logger.info("BERTScore under construction")

