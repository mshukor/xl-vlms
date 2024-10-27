import argparse
from functools import partial
from typing import Any, Callable, Dict, List, Union

import torch

from metrics.dictionary_learning_metrics import (compute_test_clipscore, 
                                                 get_random_words,
                                                 compute_overlap)
from analysis.feature_decomposition import project_test_samples

__all__ = ["dictionary_learning_evaluation"]


def dictionary_learning_evaluation(
    metric_name: str,
    features: Dict[str, torch.Tensor] = None,
    metadata: Dict[str, Any] = {},
    model_class: Callable = None,
    args: argparse.Namespace = None,
    logger: Callable = None,
    device = torch.device('cpu'),
    **kwargs: Any,
) -> None:
    concepts_dict = torch.load(args.decomposition_path)
        
    if "clipscore" in metric_name:
        features = list(features.values())[0]
        metadata = list(metadata.values())[0]
        analysis_model = concepts_dict['analysis_model']
        grounding_words=concepts_dict['text_grounding']
        projections = project_test_samples(
            sample=features,
            analysis_model=analysis_model,
            decomposition_type=concepts_dict['decomposition_method'],
        )
        if args.use_random_words:
            lm_head = model_class.get_lm_head().float()
            tokenizer = model_class.get_tokenizer()
            grounding_words = get_random_words(
                lm_head=lm_head,
                tokenizer=tokenizer,
                grounding_words=grounding_words,
            )
            logger.info(
                f"Random words usage is True. Only for CLIPScore evaluation")
                
        clipscore_dict = compute_test_clipscore(
            projections=projections,
            grounding_words=grounding_words,
            device=device,
            metadata=metadata,
        )
        logger.info(f"top-1 test CLIPScore (mean, std) {clipscore_dict['top_1_mean']: .3f} +/- {clipscore_dict['top_1_std']: .3f}")
                            
    if "bertscore" in metric_name:
        logger.info("BERTScore under construction")
        
    if "overlap" in metric_name:
        grounding_words = concepts_dict['text_grounding']
        overlap_metric, _ = compute_overlap(grounding_words)
        if args.use_random_words:
            logger.info("Overlap only computed for meaningful dictionaries, not random words")
            logger.info(f"Computing overlap for concept dictionary: {args.decomposition_path}")
        logger.info(f"Overlap metric (lower is better): {overlap_metric: .3f}")
        
    

