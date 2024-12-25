import argparse
import os
from functools import partial
from typing import Any, Callable

from metrics.captioning_metrics import compute_captioning_metrics
from metrics.vqa_accuracy import compute_vqav2_accuracy

__all__ = ["get_metric"]


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

    return metric
