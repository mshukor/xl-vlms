import argparse
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, DictionaryLearning

from analysis.multimodal_grounding import get_multimodal_grounding

__all__ = [
    "get_feature_matrix",
    "decompose_activations",
    "project_test_sample",
    "decompose_and_ground_activations",
]


def decompose_and_ground_activations(
    features: Dict[str, torch.Tensor],
    metadata: Dict[str, Any] = {},
    analysis_name: str = "decompose_activations",
    model_class: Callable = None,
    logger: Callable = None,
    args: argparse.Namespace = None,
):
    results_dict = {}
    features = list(features.values())[0]
    metadata = list(metadata.values())[0]
    concepts, activations, decomposition_model = decompose_activations(
        mat=features,
        num_concepts=args.num_concepts,
        decomposition_method=args.decomposition_method,
        args=args,
    )
    results_dict["concepts"] = concepts
    results_dict["activations"] = activations
    results_dict["decomposition_method"] = args.decomposition_method
    if logger is not None:
        logger.info(
            f"\nDecomposition type {args.decomposition_method}, Components/concepts shape: {concepts.shape}, Activations shape: {activations.shape}"
        )
    if "grounding" in analysis_name:
        text_grounding = "text_grounding" in analysis_name
        image_grounding = "image_grounding" in analysis_name
        grounding_dict = get_multimodal_grounding(
            concepts=concepts,
            activations=activations,
            model_class=model_class,
            text_grounding=text_grounding,
            image_grounding=image_grounding,
            module_to_decompose=args.module_to_decompose,
            num_grounded_text_tokens=args.num_grounded_text_tokens,
            num_most_activating_samples=args.num_most_activating_samples,
            metadata=metadata,
            logger=logger,
            args=args,
        )
        grounding_dict["analysis_model"] = decomposition_model
        results_dict.update(grounding_dict)
    return results_dict


def get_feature_matrix(
    features: List[Dict[str, Any]],
    module_name: str,
    token_idx: int = None,
) -> torch.Tensor:
    assert (
        module_name in features[0].keys()
    ), f"Given module name {module_name} not found among stored modules: {features[0].keys()}"

    feat_dim = features[0][module_name].ndim
    if feat_dim == 2:
        # Assuming feat shape [Num_token_positions, Representation_size]
        for feat in features:
            feat[module_name] = feat[module_name].unsqueeze(
                0
            )  # For unifying all shapes

    if token_idx is not None:
        matrix = torch.cat(
            [feat[module_name][:, token_idx] for feat in features], dim=0
        )
    else:
        # Take average over token positions
        matrix = torch.cat(
            [feat[module_name].mean(dim=1, keepdim=False) for feat in features], dim=0
        )

    return matrix


def decompose_activations(
    mat,
    num_concepts: int,
    decomposition_method: str = "snmf",
    concepts: torch.tensor = None,
    args: argparse.Namespace = None,
):
    """
    Input:
        mat: torch tensor or numpy array object of shape (N_samples, Representation_dim).
        num_concepts: Number of components/concepts
        decomposition_method: Decomposition/Dictionary learning model type (Options: PCA, KMeans, Semi-NMF/Non-negative dict learning, Simple)
        concepts: if not None, concepts will be considered to handle the decomposition.
    Output:
        components: Numpy array of shape (num_concepts, Representation_dim). Stores underlying dictionary elements / concept vectors
        comp_activ: Numpy array of shape (N_samples, num_concepts). Activations/Coefficients matrix.
        model: sklearn dictionary learning / clustering object.

    This function performs dictionary learning for given representation matrix and extracts concept vectors and their activations.
    """

    assert num_concepts is not None, "Number of components is None!"
    assert decomposition_method is not None, "Decomposition method specified is None!"

    if torch.is_tensor(mat):
        # Convert to numpy array for sklearn processing
        mat = mat.cpu().data.numpy()

    assert (
        len(mat.shape) == 2
    ), f"Given feature matrix needs to be 2D. Shape encountered {mat.shape}"

    if decomposition_method == "pca":
        model = PCA(n_components=num_concepts)
        if concepts is not None:
            model.components_ = concepts
            comp_activ = model.transform(mat)
            components = concepts
        else:
            comp_activ = model.fit_transform(mat)
            components = model.components_
    elif decomposition_method == "kmeans":
        model = KMeans(n_clusters=num_concepts, max_iter=args.dl_max_iter)
        # Kmeans transforms to cluster distances and not "activations". 1/(1+x) transformation to view distances as activations
        if concepts is not None:
            model.cluster_centers_ = concepts
            comp_activ = 1 / (1 + model.transform(mat))
            components = concepts
        else:
            comp_activ = 1 / (1 + model.fit_transform(mat))
            components = model.cluster_centers_
    elif decomposition_method in ["nndl", "snmf"]:
        model = DictionaryLearning(
            n_components=num_concepts,
            positive_code=True,
            fit_algorithm="cd",
            transform_algorithm="lasso_cd",
            max_iter=args.dl_max_iter,
        )
        if concepts is not None:
            model.components_ = concepts
            comp_activ = model.transform(mat)
            components = concepts
        else:
            comp_activ = model.fit_transform(mat)
            components = model.components_
    elif decomposition_method == "simple":
        if concepts is not None:
            raise NotImplementedError

        model = KMeans(n_clusters=num_concepts, max_iter=args.dl_max_iter)
        norms = np.linalg.norm(mat, axis=1)
        indices = norms.argsort()[-num_concepts:]
        components = mat[indices]
        model.cluster_centers_ = components + 0
        # Kmeans transforms to cluster distances and not "activations". 1/(1+x) transformation to view distances as activations
        comp_activ = 1 / (1 + model.transform(mat))

    return components, comp_activ, model


def project_test_sample(
    sample: torch.Tensor, analysis_model: Callable, decomposition_type: str = "nndl"
) -> np.ndarray:
    """
    Input:
        sample: torch tensor or numpy array object of shape (N_samples, Representation_dim). Should contain test representations
        analysis_model: Already learnt sklearn dictionary learning / clustering object.
        decomposition_type: Dictionary learning model type (Options: PCA, KMeans, Semi-NMF/Non-negative dict learning, Simple)
    Output:
        proj: numpy array of shape (N_samples, # components of analysis_model)
    """
    if isinstance(sample, torch.Tensor):
        sample = sample.cpu().numpy()

    assert isinstance(sample, np.ndarray), "sample should be of type np.ndarray"

    projected_sample = analysis_model.transform(sample)
    if decomposition_type in ["kmeans", "simple"]:
        # Kmeans transforms to cluster distances and not "activations". 1/(1+x) transformation to view distances as activations
        projected_sample = 1 / (1 + projected_sample)
    return projected_sample
