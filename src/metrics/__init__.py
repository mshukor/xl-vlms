import argparse
from functools import partial
from typing import Any, Callable, Dict, List, Union

import torch

from metrics.clipscore_overlap import compute_clipscore, compute_overlap
from analysis.feature_decomposition import project_test_samples

__all__ = ["dictionary_learning_evaluation"]


def dictionary_learning_evaluation(
    metric_name: str,
    features: Dict[str, torch.Tensor] = None,
    loader: Callable = None,
    metadata: Dict[str, Any] = {},
    logger: Callable = None,
    args: argparse.Namespace = None,
    device = torch.device('cpu'),
    **kwargs: Any,
) -> None:
    concepts_dict = torch.load(args.decomposition_path)
        
    if "clipscore" in metric_name:
        features = list(features.values())[0]
        metadata = list(metadata.values())[0]
        analysis_model = concepts_dict['analysis_model']
        projections = project_test_samples(
            sample=features,
            analysis_model=analysis_model,
            decomposition_type=concepts_dict['decomposition_method'],
        )
        clipscore_dict = compute_clipscore(
            loader=loader,
            projections=projections,
            grounding_words=concepts_dict['text_grounding'],
            device=device,
            metadata=metadata,
        )
        logger.info(f"top-1 test CLIPScore (mean, std) {clipscore_dict['top_1_mean']: .3f} +/- {clipscore_dict['top_1_std']: .3f}")
                            
    if "overlap" in metric_name:
        grounded_words = concepts_dict['text_grounding']
        overlap_metric, _ = compute_overlap(grounded_words)
        logger.info(f"Overlap metric (lower is better): {overlap_metric: .3f}")
        
    if "bertscore" in metric_name:
        logger.info("BERTScore under construction")

