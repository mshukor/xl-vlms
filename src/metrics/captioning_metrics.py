# Adapted from https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/_task_utils/vqa_eval_metric.py#L4
import itertools
import json
from typing import Any, Callable, Dict, List

import language_evaluation
import torch

from metrics.utils import (get_number_predictions_with_token_of_interest,
                           get_words_frequency)

__all__ = ["compute_captioning_metrics"]


def compute_captioning_metrics(
    data: Dict[str, Any],
    metrics: List[str] = None,
    preds_token_of_interests: List[str] = None,
    targets_token_of_interests: List[str] = None,
    save_filename: str = None,
    save_predictions: bool = False,
    predictions_path: str = None,
    logger: Callable = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    for idx in range(len(data["targets"])):
        for i in range(len(data["targets"][idx])):
            data["targets"][idx][i] = data["targets"][idx][i].split("$$")

    evaluator = language_evaluation.CocoEvaluator(
        verbose=True, coco_types=metrics
    )  # coco_types=["BLEU", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
    predictions, targets = list(itertools.chain(*data["model_predictions"])), list(
        itertools.chain(*data["targets"])
    )
    results = {}
    results["scores"] = evaluator.run_evaluation(predictions, targets)

    answer_counts = get_words_frequency(data["model_predictions"])
    # For better visualization
    top_k_items = sorted(answer_counts.items(), key=lambda item: item[1])
    topk_dict = {}
    for item in top_k_items:
        topk_dict[item[0]] = item[1]
    results["answer_counts"] = topk_dict

    (
        num_preds_with_toi,
        num_preds_and_targets_with_toi,
        num_preds_and_baseline_preds_with_toi,
        num_preds_changed,
    ) = (0, 0, 0, 0)
    if preds_token_of_interests is not None and targets_token_of_interests is not None:
        (
            num_preds_with_toi,
            num_preds_and_targets_with_toi,
            num_preds_and_baseline_preds_with_toi,
            num_preds_changed,
        ) = get_number_predictions_with_token_of_interest(
            data["model_predictions"],
            data["targets"],
            ids=data["img_id"],
            preds_token_of_interests=preds_token_of_interests,
            targets_token_of_interests=targets_token_of_interests,
            predictions_path=predictions_path,
        )
    results["num_preds_with_toi"] = num_preds_with_toi
    results["num_preds_and_targets_with_toi"] = num_preds_and_targets_with_toi
    results["num_preds_and_baseline_preds_with_toi"] = (
        num_preds_and_baseline_preds_with_toi
    )
    results["num_preds_changed"] = num_preds_changed

    results["dataset_size"] = sum([len(p) for p in data["model_predictions"]])
    if logger is not None:
        logger.info(f"Answer counts: {results['answer_counts']}")
        examples = data["model_predictions"][:10]
        logger.info(f"Captioning prediction examples: {examples}")
        logger.info(
            f"num_preds_with_toi {preds_token_of_interests}: {results['num_preds_with_toi']}"
        )
        logger.info(
            f"num_preds_with_toi {preds_token_of_interests}: {num_preds_with_toi}"
        )
        logger.info(
            f"num_preds_and_targets_with_toi {targets_token_of_interests}: {num_preds_and_targets_with_toi}"
        )
        logger.info(
            f"num_preds_and_baseline_preds_with_toi {targets_token_of_interests}: {num_preds_and_baseline_preds_with_toi}"
        )
        logger.info(f"num_preds_changed: {num_preds_changed}")
        logger.info(f"Captioning metrics: {results['scores']}")

    if save_filename:
        with open(save_filename, "w") as json_file:
            json.dump(results, json_file, indent=4)
        if logger is not None:
            logger.info(f"Saving data to: {save_filename}")
        if save_predictions:
            id_to_answer = {}
            for i in range(len(data["model_predictions"])):
                preds = data["model_predictions"][i]
                ids = data["img_id"][i]
                for id, pred in zip(ids, preds):
                    if isinstance(id, torch.Tensor):
                        id = id.item()
                    id_to_answer[id] = pred
            save_filename = save_filename.split(".json")[0] + "_model_prediction.json"
            with open(save_filename, "w") as json_file:
                json.dump(id_to_answer, json_file, indent=4)
            if logger is not None:
                logger.info(
                    f"Saving {len(id_to_answer)} predictions to: {save_filename}"
                )

    return results
