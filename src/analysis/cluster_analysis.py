import argparse
import os
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image
from scipy import stats
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

from analysis.feature_decomposition import decompose_activations
from analysis.multimodal_grounding import (concept_text_grounding,
                                           get_multimodal_grounding)
from analysis.utils import cosine_similarity, l2_distance

__all__ = ["analyse_clusters"]


def analyse_clusters(
    features: Dict[str, Any],
    metadatas: Dict[str, Any],
    analysis_data_original: Dict[str, torch.Tensor] = None,
    model_class: Callable = None,
    save_analysis: bool = False,
    num_concepts: int = 20,
    save_dir: str = "",
    save_name: str = "",
    logger: Callable = None,
    args: argparse.Namespace = None,
    **kwargs: Any,
) -> None:

    save_data = {}

    keys = list(features.keys())
    assert len(keys) == 2, "Exactly two keys should be in the features dictionary."

    original_key = next(key for key in keys if "original" in key)
    dest_key = next(key for key in keys if key != original_key)

    features_original, metadata_original_feature = (
        features[original_key],
        metadatas[original_key],
    )
    features_destination, metadata_destination_feature = (
        features[dest_key],
        metadatas[dest_key],
    )

    # Computing the analysis of original features if analysis_data_original is None
    if analysis_data_original is None:

        analysis_data_original = compute_analysis_features(
            features_original,
            metadata_original_feature,
            model_class,
            num_concepts,
            logger,
            args,
        )

    (
        map_original_destination,
        common_local_data_indices_original,
        common_local_data_indices_destination,
    ) = common_local_indices(metadata_original_feature, metadata_destination_feature)

    (
        per_sample_shift_original_to_finetune,
        mean_shift_original_to_finetune,
        shifted_comps,
    ) = compute_shift_concepts(
        features_original,
        features_destination,
        analysis_data_original,
        mean_method=1,
        normalize_shift_vectors=False,
        normalize_shifted_clusters=False,
        shift_alpha=1.0,
        map_original_destination=map_original_destination,
        samples_mask_original=common_local_data_indices_original,
        args=None,
    )

    shift_grounding_words = compute_shift_grounding_words(
        model_class, mean_shift_original_to_finetune
    )

    _, shifted_activations, _ = decompose_activations(
        mat=features_destination,
        num_concepts=num_concepts,
        decomposition_method=args.decomposition_method,
        concepts=shifted_comps,
        args=args,
    )
    if logger is not None:
        logger.info(
            f"\nDecomposition type {args.decomposition_method}, Components/concepts shape: {shifted_comps.shape}, Activations shape: {shifted_activations.shape}"
        )
    # the activations that are apssed through this are those corresponding to the destination model
    # so, what should be the mask in metadata_destination_feature ... # TODO : for now only pass the activations where the masks are the same
    shifted_grounding_dict = get_multimodal_grounding(
        concepts=shifted_comps,
        activations=shifted_activations,
        model_class=model_class,
        text_grounding=True,
        image_grounding=True,
        module_to_decompose=args.module_to_decompose,
        num_grounded_text_tokens=args.num_grounded_text_tokens,
        num_most_activating_samples=args.num_most_activating_samples,
        metadata=metadata_destination_feature,
        logger=logger,
        args=args,
    )

    if args.visualize_concepts or args.compute_recovery_metrics:
        analysis_data_destination = compute_analysis_features(
            features_destination,
            metadata_destination_feature,
            model_class,
            num_concepts,
            logger,
            args,
        )

    if args.visualize_concepts:

        visualize_concepts(
            num_concepts=num_concepts,
            shift_grounding_dict=shifted_grounding_dict,
            analysis_data_original=analysis_data_original,
            analysis_data_destination=analysis_data_destination,
            save_dir=save_dir,
            save_name=save_name,
        )

    if args.compute_recovery_metrics:

        metrics = {}

        # word recovery
        metrics["word_intersection_original"], metrics["word_intersection_shifted"] = (
            metric_recovery(
                num_concepts,
                metric_key="text_grounding",
                shift_grounding_dict=shifted_grounding_dict,
                analysis_data_original=analysis_data_original,
                analysis_data_destination=analysis_data_destination,
            )
        )

        # MAS recovery
        metrics["mas_intersection_original"], metrics["mas_intersection_shifted"] = (
            metric_recovery(
                num_concepts,
                metric_key="image_grounding_paths",
                shift_grounding_dict=shifted_grounding_dict,
                analysis_data_original=analysis_data_original,
                analysis_data_destination=analysis_data_destination,
            )
        )

        # Cosine distance recovery and l2 distance recovery by shifting
        similarity_matrix = cosine_similarity(
            analysis_data_original["concepts"], analysis_data_destination["concepts"]
        )
        cost_matrix = -similarity_matrix
        _, col_ind_matching = linear_sum_assignment(cost_matrix)
        rearranged_destination_concepts = torch.stack(
            [analysis_data_destination["concepts"][i] for i in col_ind_matching]
        )

        distance_shifted_to_finetuned_decomposition = torch.norm(
            rearranged_destination_concepts - shifted_grounding_dict["concepts"], dim=1
        )
        distance_shifted_to_original_decomposition = torch.norm(
            rearranged_destination_concepts - analysis_data_original["concepts"], dim=1
        )

        cosine_sim_shifted_to_finetuned_decomposition = cosine_similarity(
            rearranged_destination_concepts, shifted_grounding_dict["concepts"]
        ).diagonal()
        cosine_sim_shifted_to_original_decomposition = cosine_similarity(
            rearranged_destination_concepts, analysis_data_original["concepts"]
        ).diagonal()

        improved_distance = (
            distance_shifted_to_original_decomposition
            - distance_shifted_to_finetuned_decomposition
        )
        improved_cosine = (
            cosine_sim_shifted_to_finetuned_decomposition
            - cosine_sim_shifted_to_original_decomposition
        )

        metrics["improved_distance"] = improved_distance
        metrics["improved_cosine"] = improved_cosine
        metrics["normalized_improved_distance"] = (
            improved_distance / distance_shifted_to_original_decomposition
        )
        metrics["normalized_improved_cosine"] = (
            improved_cosine / cosine_sim_shifted_to_original_decomposition
        )

        save_data["recovery_metrics"] = metrics

    if args.compute_stat_shift_vectors:

        stats = {}

        (
            stats["cosine_sim_scores"],
            stats["dot_prod_scores"],
            stats["average_cosine_sim_w_mean_scores"],
            stats["average_dot_product_w_mean_scores"],
            stats["mag_mean_shift_score"]
        ) = process_shift_vectors(per_sample_shift_original_to_finetune)

        save_data["shift_stats"] = stats

    if save_analysis:

        save_data["per_sample_shift"] = per_sample_shift_original_to_finetune
        save_data["shifted_components"] = shifted_comps
        save_data["shift_text_grounding"] = shift_grounding_words
        save_data["shifted_components_text_grounding"] = shifted_grounding_dict[
            "text_grounding"
        ]
        analysis_dir = os.path.join(save_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        file_name = os.path.join(
            analysis_dir, f"analyse_clusters_{save_name}.pth"
        )
        
        torch.save(save_data, file_name)
        if logger is not None:
            logger.info(f"Saving cluster analysis to: {file_name}")

    return


def compute_analysis_features(
    features: torch.Tensor = None,
    metadata: Dict[str, Any] = None,
    model_class: Callable = None,
    num_concepts: int = 20,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Dict[str, Any]:

    analysis_data = {}

    concepts, activations, _ = decompose_activations(
        mat=features,
        num_concepts=num_concepts,
        decomposition_method=args.decomposition_method,
        args=args,
    )
    if logger is not None:
        logger.info(
            f"\nDecomposition type {args.decomposition_method}, Components/concepts shape: {concepts.shape}, Activations shape: {activations.shape}"
        )
    grounding_dict = get_multimodal_grounding(
        concepts=concepts,
        activations=activations,
        model_class=model_class,
        text_grounding=True,
        image_grounding=True,
        module_to_decompose=args.module_to_decompose,
        num_grounded_text_tokens=args.num_grounded_text_tokens,
        num_most_activating_samples=args.num_most_activating_samples,
        metadata=metadata,
        logger=logger,
        args=args,
    )

    analysis_keys = [
        "image_grounding_paths",
        "text_grounding",
        "concepts",
        "activations",
    ]
    for analysis_key in analysis_keys:
        assert (
            analysis_key in grounding_dict
        ), f"{analysis_key} not found, got {grounding_dict.keys()}."
        analysis_data[analysis_key] = grounding_dict[analysis_key]

    return analysis_data


def compute_reference_clusters(
    concepts: torch.Tensor,
    concept_activations: torch.Tensor,
    features: torch.Tensor,
    mask_samples: torch.Tensor,
    similarity: str = "cosine",
) -> torch.Tensor:
    if similarity == "cosine":
        sim = cosine_similarity(features, concepts)  # (N, C)
        features_clusters_association_idx = sim.argmax(dim=-1)  # (N)
    elif similarity == "concept_activation":
        features_clusters_association_idx = torch.argmax(
            concept_activations, dim=1
        )  # (N)
    elif similarity == "l2":
        dist = l2_distance(features, concepts)
        features_clusters_association_idx = dist.argmin(dim=-1)  # (N)

    else:
        raise NotImplementedError(f"{similarity} not implemented.")

    indices_associated_to_concepts = [
        [
            idx
            for idx in torch.where(features_clusters_association_idx == i)[0]
            if idx.item() in mask_samples
        ]
        for i in range(concept_activations.shape[1])
    ]
    clusters_reference = concepts[features_clusters_association_idx]  # (N, D)

    return indices_associated_to_concepts, clusters_reference


def compute_shift_vector(
    clusters: torch.Tensor,
    features: torch.Tensor,
    num_concepts: int,
    features_clusters_association_idx: torch.Tensor,
) -> torch.Tensor:
    N, D = features.shape
    vector = torch.zeros(num_concepts, D)
    for concept_idx in range(num_concepts):
        concept_mask = features_clusters_association_idx == concept_idx
        vector[concept_idx] = (
            features[concept_mask].reshape(-1, D).mean(0) - clusters[concept_idx]
        )

    return vector


def common_local_indices(metadata_original_feature, meta_data_destination_feature):

    original_TOI_presence = metadata_original_feature.get(
        "token_of_interest_mask", None
    )
    dest_TOI_presence = meta_data_destination_feature.get(
        "token_of_interest_mask", None
    )

    if original_TOI_presence is None or dest_TOI_presence is None:
        assert (
            original_TOI_presence is None and dest_TOI_presence is None
        ), "both or none of of saved analysis should have token_of_interest_mask."

        # usually the presence arrays should not be None, but just in case:
        len_analysis_original = len(list(metadata_original_feature.values())[0])
        len_analysis_dest = len(list(meta_data_destination_feature.values())[0])
        assert (
            len_analysis_original == len_analysis_dest
        ), "length of both hidden states in original and destination should be the same if their presence mask is not provided."
        original_TOI_presence, dest_TOI_presence = torch.ones(
            len_analysis_original
        ), torch.ones(len_analysis_original)

    cnt1 = -1
    cnt2 = -1
    common_local_data_indices_original = []
    common_local_data_indices_destination = []
    # we want a map from the cnt of original to cnt of the finetune
    map_original_destination = {}
    for idx in range(len(original_TOI_presence)):
        if original_TOI_presence[idx] == 1:
            cnt1 += 1
        if dest_TOI_presence[idx] == 1:
            cnt2 += 1
            if original_TOI_presence[idx] == 1:
                common_local_data_indices_destination.append(cnt2)
                common_local_data_indices_original.append(cnt1)
                map_original_destination[cnt1] = cnt2

    return (
        map_original_destination,
        common_local_data_indices_original,
        common_local_data_indices_destination,
    )


def compute_mean_per_concept(
    features_associated_to_clusters: List[torch.tensor], mean_method: int = 1
):
    if mean_method == 1:
        mean_features_associated_to_clusters = [
            torch.mean(component_elements, dim=0)
            for component_elements in features_associated_to_clusters
        ]

    elif mean_method == 2:
        mean_features_associated_to_clusters = [
            torch.median(component_elements, dim=0)[0]
            for component_elements in features_associated_to_clusters
        ]

    else:
        mean_features_associated_to_clusters = [
            torch.tensor(
                stats.trim_mean(
                    component_elements.cpu().numpy(), proportiontocut=0.1, axis=0
                )
            )
            for component_elements in features_associated_to_clusters
        ]

    return mean_features_associated_to_clusters


def compute_shift_concepts(
    features_original,
    features_destination,
    analysis_data_original,
    mean_method=1,
    normalize_shift_vectors=False,
    normalize_shifted_clusters=False,
    shift_alpha=1.0,
    map_original_destination=None,
    samples_mask_original=None,
    args=None,
) -> Any:

    # from analysis of features
    original_comp = analysis_data_original["concepts"]
    original_decomp = analysis_data_original["activations"]

    indices_associated_to_concepts, clusters_reference = compute_reference_clusters(
        concepts=original_comp,
        concept_activations=original_decomp,
        features=features_original,
        similarity="concept_activation",  # cosine
        mask_samples=samples_mask_original,
    )

    original_features_associated_to_concepts = [
        torch.stack([features_original[tmp_idx].cpu() for tmp_idx in idx_comp_original])
        for idx_comp_original in indices_associated_to_concepts
    ]
    destination_features_associated_to_concepts = [
        torch.stack(
            [
                features_destination[map_original_destination[tmp_idx.item()]].cpu()
                for tmp_idx in idx_comp_original
            ]
        )
        for idx_comp_original in indices_associated_to_concepts
    ]

    per_sample_shift_original_to_finetune = [
        destination_features_associated_to_concepts[comp_idx]
        - original_features_associated_to_concepts[comp_idx]
        for comp_idx in range(original_comp.shape[0])
    ]

    mean_original_features_associated_to_concepts = compute_mean_per_concept(
        original_features_associated_to_concepts, mean_method=mean_method
    )
    mean_destination_features_associated_to_concepts = compute_mean_per_concept(
        destination_features_associated_to_concepts, mean_method=mean_method
    )

    mean_shift_original_to_finetune = torch.stack(
        mean_destination_features_associated_to_concepts
    ) - torch.stack(mean_original_features_associated_to_concepts)

    if normalize_shift_vectors:
        mean_shift_original_to_finetune /= torch.linalg.norm(
            mean_shift_original_to_finetune
        )

    shifted_comps = original_comp + shift_alpha * mean_shift_original_to_finetune
    if normalize_shifted_clusters:
        shifted_comps = shifted_comps / torch.linalg.norm(
            shifted_comps, axis=1, keepdims=True
        )

    return (
        per_sample_shift_original_to_finetune,
        mean_shift_original_to_finetune,
        shifted_comps,
    )


def compute_shift_grounding_words(
    model_class: Callable = None,
    mean_shift_original_to_finetune: torch.Tensor = None,
) -> Any:

    lm_head = model_class.get_lm_head().float()
    tokenizer = model_class.get_tokenizer()

    device = torch.device("cpu")

    shift_grounding_words = concept_text_grounding(
        mean_shift_original_to_finetune.to(device),
        lm_head,
        tokenizer,
        num_top_tokens=30,
    )
    shift_grounding_words = [list(set(a)) for a in shift_grounding_words]

    return shift_grounding_words


def comp_vis(components, mas_paths, grounding_words, save_dir, save_name):
    n_MAS = 5
    n_word = 10
    width = n_MAS * 2 + 0.1

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    trasnform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )

    fig = plt.figure(figsize=(width, len(components) * width / 5), dpi=150)
    for idx_component, comp_idx in enumerate(components):
        words = grounding_words[comp_idx][-n_word:]
        words.reverse()
        plot_str = ""
        for idx_word, word in enumerate(words):
            if idx_word % 2 == 0:
                plot_str = plot_str + "'" + word + "'" + ","

            else:
                plot_str = plot_str + "'" + word + "'" + "\n"
        plot_str = plot_str[:-1]
        mas = []
        for img_path in mas_paths[comp_idx][:n_MAS]:
            img = trasnform(Image.open(img_path).convert("RGB"))
            img_unnorm = ((img / img.max() + 1) / 2).permute(
                1, 2, 0
            )  # Bringing in range [0, 1] and numpy channel order
            mas.append(img_unnorm)

        for i in range(len(mas)):
            fig.add_subplot(len(components), n_MAS, n_MAS * idx_component + i + 1)
            plt.imshow(mas[i])
            plt.axis("off")
        plt.text(
            -224 * (n_MAS - 1) - 320,
            112 + idx_component * 0,
            plot_str,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
            linespacing=1.8,
        )
        plt.subplots_adjust(wspace=0.02)

    plt.savefig(
        os.path.join(save_dir, "analysis", f"{save_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )


def visualize_concepts(
    num_concepts: int = 20,
    shift_grounding_dict: Dict[str, Any] = None,
    analysis_data_original: Dict[str, Any] = None,
    analysis_data_destination: Dict[str, Any] = None,
    save_dir: str = None,
    save_name: str = None,
):

    similarity_matrix = cosine_similarity(
        analysis_data_original["concepts"], analysis_data_destination["concepts"]
    )
    cost_matrix = -similarity_matrix
    _, col_ind_matching = linear_sum_assignment(cost_matrix)

    comp_vis(
        range(num_concepts),
        analysis_data_original["image_grounding_paths"],
        analysis_data_original["text_grounding"],
        save_name="original_concepts_" + save_name,
        save_dir=save_dir,
    )

    comp_vis(
        range(num_concepts),
        shift_grounding_dict["image_grounding_paths"],
        shift_grounding_dict["text_grounding"],
        save_name="shifted_concepts_" + save_name,
        save_dir=save_dir,
    )

    comp_vis(
        range(num_concepts),
        [
            analysis_data_destination["image_grounding_paths"][i]
            for i in col_ind_matching
        ],
        [analysis_data_destination["text_grounding"][i] for i in col_ind_matching],
        save_name="finetuned_concepts_" + save_name,
        save_dir=save_dir,
    )


# Recovery metrics


def metric_recovery(
    num_concepts: int = 20,
    metric_key: str = "text_grounding",
    shift_grounding_dict: Dict[str, Any] = None,
    analysis_data_original: Dict[str, Any] = None,
    analysis_data_destination: Dict[str, Any] = None,
) -> Tuple[List[int], List[int]]:

    similarity_matrix = cosine_similarity(
        analysis_data_original["concepts"], analysis_data_destination["concepts"]
    )
    cost_matrix = -similarity_matrix
    _, col_ind_matching = linear_sum_assignment(cost_matrix)

    original_dest_intersection, shifted_dest_intersection = [], []

    for concept_idx in range(num_concepts):
        original_dest_intersection.append(
            len(
                list(
                    set(
                        analysis_data_destination[metric_key][
                            col_ind_matching[concept_idx]
                        ]
                    ).intersection(set(analysis_data_original[metric_key][concept_idx]))
                )
            )
        )
        shifted_dest_intersection.append(
            len(
                list(
                    set(
                        analysis_data_destination[metric_key][
                            col_ind_matching[concept_idx]
                        ]
                    ).intersection(set(shift_grounding_dict[metric_key][concept_idx]))
                )
            )
        )

    return (original_dest_intersection, shifted_dest_intersection)


def process_shift_vectors(per_samples_shifts: List[torch.Tensor]) -> Any:

    def process_one_concept_shifts(directions):

        dot_product_matrix = torch.matmul(directions, directions.T)

        lower_triangular = torch.tril(dot_product_matrix, -1)
        mask = lower_triangular != 0

        average_dot_product = torch.mean(dot_product_matrix[mask])

        cosine_sim_matrix = cosine_similarity(directions, directions)
        average_cosine_sim = torch.mean(cosine_sim_matrix[mask])

        mean_shift = torch.mean(directions, axis=0)
        std_shift = torch.std(directions, axis=0)
        mag_mean_shift_score = torch.norm(mean_shift)
        mag_std_shift_score = torch.norm(std_shift)

        mag_directions = torch.norm(directions, dim=1)
        mean_mag_shift_score = torch.mean(mag_directions)
        std_mag_shift_score = torch.std(mag_directions)

        cosine_sim_with_mean_shift = cosine_similarity(
            directions, mean_shift.reshape(1, -1)
        ).flatten()
        average_cosine_sim_w_mean = torch.mean(cosine_sim_with_mean_shift)

        dot_product_with_mean_shift = torch.matmul(directions, mean_shift)
        average_dot_product_w_mean = torch.mean(dot_product_with_mean_shift)

        return (
            average_cosine_sim,
            average_dot_product,
            mag_mean_shift_score,
            mean_mag_shift_score,
            mag_std_shift_score,
            std_mag_shift_score,
            average_cosine_sim_w_mean,
            average_dot_product_w_mean,
            std_shift,
        )

    (
        mag_mean_shift_scores, # this is the magnitude of each global shift vector
        cosine_sim_scores,  # this is the pairwise average
        average_cosine_sim_w_mean_scores,  # this is the average of cosine sim btw the shift vector and the mean of shift vectors
        dot_prod_scores,  # this is the pairwise average
        average_dot_product_w_mean_scores,  # this is the average of the dot prod btw the shift vector and the mean of shift vectors
    ) = (
        [],
        [],
        [],
        [],
        [],
    )

    for i, directions in enumerate(per_samples_shifts):

        (
            cosine_sim_score,
            dot_prod_score,
            mag_mean_shift_score,
            mean_mag_shift_score,  # Ignoring for now, but may be useful
            mag_std_shift_score,  # Ignoring for now, but may be useful
            std_mag_shift_score,  # Ignoring for now, but may be useful
            average_cosine_sim_w_mean,
            average_dot_product_w_mean,
            std_shift,  # Ignoring for now, but may be useful
        ) = process_one_concept_shifts(directions)

        mag_mean_shift_scores.append(mag_mean_shift_score)
        cosine_sim_scores.append(cosine_sim_score)
        dot_prod_scores.append(dot_prod_score)
        average_cosine_sim_w_mean_scores.append(average_cosine_sim_w_mean)
        average_dot_product_w_mean_scores.append(average_dot_product_w_mean)

    return (
        cosine_sim_scores,
        dot_prod_scores,
        average_cosine_sim_w_mean_scores,
        average_dot_product_w_mean_scores,
        mag_mean_shift_scores
    )
