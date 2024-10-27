import argparse
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, DictionaryLearning

__all__ = ["get_feature_matrix", "decompose_activations", "project_test_sample"]


def get_feature_matrix(
    features: List[Dict[str, Any]],
    module_name: str,
    args: argparse.Namespace = None,
) -> torch.Tensor:
    assert (
        module_name in features[0].keys()
    ), f"Given module name {module_name} not found among stored modules: {features[0].keys()}"

    feat_dim = features[0][module_name].ndim
    token_idx = args.decomposition_extract_pos
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
    args: argparse.Namespace = None,
):
    """
    Input:
        mat: torch tensor or numpy array object of shape (N_samples, Representation_dim).
        num_concepts: Number of components/concepts
        decomposition_method: Decomposition/Dictionary learning model type (Options: PCA, KMeans, Semi-NMF/Non-negative dict learning, Simple)
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
        comp_activ = model.fit_transform(mat)
        components = model.components_
    elif decomposition_method == "kmeans":
        model = KMeans(n_clusters=num_concepts, max_iter=args.dl_max_iter)
        # Kmeans transforms to cluster distances and not "activations". 1/(1+x) transformation to view distances as activations
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
        comp_activ = model.fit_transform(mat)
        components = model.components_
    elif decomposition_method == "simple":
        model = KMeans(n_clusters=num_concepts, max_iter=args.dl_max_iter)
        norms = np.linalg.norm(mat, axis=1)
        indices = norms.argsort()[-num_concepts:]
        components = mat[indices]
        model.cluster_centers_ = components + 0
        # Kmeans transforms to cluster distances and not "activations". 1/(1+x) transformation to view distances as activations
        comp_activ = 1 / (1 + model.transform(mat))

    return components, comp_activ, model


def project_test_samples(
    sample: torch.Tensor, analysis_model: Callable, decomposition_type: str = "snmf"
):
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
