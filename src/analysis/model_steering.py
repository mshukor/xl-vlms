import argparse
import json
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from analysis.feature_decomposition import decompose_activations
from analysis.utils import get_dict_of_top_k_items

__all__ = ["get_steering_vector", "get_shift_vector_scores"]

SUPPORTED_STEERING_METHODS = [
    "shift_of_means",
    "shift_between_intra_clusters",
]

def get_steering_vector(
    features: Dict[str, torch.Tensor],
    steering_method: str = "shift_of_means",
    base_features_key: str = "",
    num_concepts: List[int] = [],
    logger: Callable = None,
    save_dir: str = "",
    save_name: str = "",
    args: argparse.Namespace = None,
    **kwargs: Any,
) -> None:
    assert (
        len(features.values()) == 2
    ), f"There is only {len(features.values())} features, 2 must be given."
    assert (
        base_features_key in features
    ), f"{base_features_key} not found in features. Only got {features.keys()}"

    feat1 = features[base_features_key]
    other_features_key = [k for k in features if k != base_features_key][0]
    feat2 = features[other_features_key]  # (N, D)

    if logger is not None:
        logger.info(
            f"Computing steering vectors for feat1 {feat1.shape} and feat2 {feat2.shape}"
        )
    meta_data = {}
    if steering_method == "shift_of_means":
        vector = feat2.mean(0) - feat1.mean(0)
    elif "cluster" in steering_method:
        concepts, activations, _ = decompose_activations(
            mat=feat1,
            num_concepts=num_concepts[0],
            decomposition_method=args.decomposition_method,
            args=args,
        )
        concepts, activations = torch.tensor(concepts), torch.tensor(activations)

        meta_data["concepts1"] = concepts
        meta_data["activations1"] = activations
        # concepts (C, D), activations (N, C)
        cluster_idx = None

        if steering_method == "shift_between_intra_clusters":
            vector = concepts.unsqueeze(1) - concepts.unsqueeze(0)  # (C, C, D)
            for i in range(vector.shape[0]):
                for j in range(vector.shape[1]):
                    if i != j:
                        meta_data[f"steering_vector_concept_{i}_to_{j}"] = vector[i][j]
            meta_data[f"steering_vector_shift_of_means"] = feat2.mean(0) - feat1.mean(0)
            meta_data[f"steering_vector_mean_of_directions"] = vector.mean(0).mean(0)
        else:
            vector = feat2.mean(0) - concepts.mean(0)

        logger.info(
            f"Computing steering vector {steering_method}, cluster idx: {cluster_idx}"
        )
    else:
        raise NotImplementedError(
            f"Only the following steering methods are supported: {SUPPORTED_STEERING_METHODS}, got {steering_method}"
        )

    meta_data["steering_vector"] = vector
    file_name = os.path.join(save_dir, f"{steering_method}_{save_name}.pth")
    torch.save(meta_data, file_name)
    if logger is not None:
        logger.info(f"Saving steering vector to: {file_name}")
    return vector


def get_shift_vector_scores(
    results: Dict[str, Any],
    topk: int = 5,
    score_key: str = "",
    keep_first_word: bool = False,
    reference_dict: Dict[str, Any] = {},
) -> Tuple[List, int]:

    if score_key:
        answer_counts = results["answer_counts"][score_key]
        ref_dict = reference_dict.get("answer_counts", {}).get(score_key, {})
    else:
        answer_counts = results["answer_counts"]
        ref_dict = reference_dict.get("answer_counts", {})

    answer_counts = {k: v for k, v in answer_counts.items() if k}
    if keep_first_word:
        answer_counts_ = {}
        ref_dict_ = {}
        for k, v in answer_counts.items():
            k_ = k.split(" ")[0]
            if k_ in answer_counts_:
                answer_counts_[k_] += v
            else:
                answer_counts_[k_] = v

            if k in ref_dict:
                if k_ in ref_dict_:
                    ref_dict_[k_] += ref_dict[k]
                else:
                    ref_dict_[k_] = ref_dict[k]

        answer_counts = answer_counts_
    top_k_items = get_dict_of_top_k_items(answer_counts, topk, reference_dict=ref_dict_)
    if len(top_k_items) > 1:
        counts = np.array(list(top_k_items.values()))[:, None]
        model = KMeans(n_clusters=2, max_iter=50).fit(counts)

        predicted_clusters = [
            (k, model.predict(np.array([v])[:, None])) for k, v in top_k_items.items()
        ]
        main_cluster_idx = model.predict(
            np.array([max([v for v in top_k_items.values()])])[:, None]
        )

        main_answers = [k for k in predicted_clusters if k[1] == main_cluster_idx]
        main_answers_keys = [k[0] for k in main_answers]
        main_answers_values = [top_k_items[k[0]] for k in main_answers]

        counts_difference_to_main_answers = [
            min(main_answers_values) - top_k_items[k[0]]
            for k in predicted_clusters
            if k[1] != main_cluster_idx
        ]
        counts_difference_to_main_answers = (
            min(counts_difference_to_main_answers)
            if counts_difference_to_main_answers
            else 0
        )
    else:
        main_answers_keys = list(top_k_items.keys())[0]
        counts_difference_to_main_answers = top_k_items[main_answers_keys]

    return main_answers_keys, counts_difference_to_main_answers


def get_topk_shift_vectors(
    results_paths: List[str],
    answer_types: List[str],
    number_target_concepts: int = 1,
    score_keys: Dict[str, str] = {},
    topk: int = 5,
    reference_dict_path: str = "",
    num_shift_vectors: int = 5,
    unique_vectors: bool = True,
    keep_first_word: bool = False,
) -> Dict[str, Any]:

    reference_dict = json.load(open(reference_dict_path)) if reference_dict_path else {}
    all_scores = {}
    for i, results_path in enumerate(results_paths):
        if os.path.exists(results_path):
            results = json.load(open(results_path))
            if answer_types:
                score_key = score_keys.get(answer_types[i], "other")
                ans_type = answer_types[i]
            else:
                score_key = None
                ans_type = ""
            main_answers, counts_difference_to_main_answers = get_shift_vector_scores(
                results,
                topk=topk,
                score_key=score_key,
                reference_dict=reference_dict,
                keep_first_word=keep_first_word,
            )
            if len(main_answers) <= number_target_concepts and len(main_answers) > 0:
                all_scores[results_path] = {
                    "main_answers": main_answers,
                    "counts_difference_to_main_answers": counts_difference_to_main_answers,
                    "answer_type": ans_type,
                }
    if unique_vectors:
        sorted_dict = dict(
            sorted(
                all_scores.items(),
                key=lambda item: item[1]["counts_difference_to_main_answers"],
                reverse=True,
            )
        )
        shift_targets = []
        all_scores_filtered = {}
        for k, v in sorted_dict.items():
            main_answers = v["main_answers"]
            if main_answers not in shift_targets:
                all_scores_filtered[k] = v
                shift_targets.append(main_answers)
            if len(all_scores_filtered) >= num_shift_vectors:
                break
        sorted_dict = all_scores_filtered
    else:
        sorted_dict = dict(
            sorted(
                all_scores.items(),
                key=lambda item: item[1]["counts_difference_to_main_answers"],
                reverse=True,
            )[:num_shift_vectors]
        )

    return sorted_dict
