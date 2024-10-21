from typing import List

import torch

__all__ = ["cosine_similarity", "l2_distance"]


def cosine_similarity(
    batch1: torch.Tensor, batch2: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    batch1, batch2 = batch1.float(), batch2.float()
    dot_product = batch1 @ batch2.T  # Shape: (B, B)

    norm_batch1 = torch.norm(batch1, p=2, dim=-1, keepdim=True)
    norm_batch2 = torch.norm(batch2, p=2, dim=-1, keepdim=True)

    cosine_sim = dot_product / (norm_batch1 * norm_batch2.T + eps)

    return cosine_sim


def l2_distance(batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
    batch1, batch2 = batch1.float(), batch2.float()
    return (batch1 - batch2).pow(2).sum(-1).sqrt()


def get_token_of_interest_features(
    features: torch.Tensor, token_of_interest_mask: torch.Tensor = None
) -> torch.Tensor:

    if token_of_interest_mask is not None:
        if isinstance(token_of_interest_mask, list):
            token_of_interest_mask = torch.cat(token_of_interest_mask, dim=0)
        features = features[token_of_interest_mask].reshape(-1, features.shape[1])

    return features


def get_matched_token_of_interest_mask(features_path: List[str]) -> torch.Tensor:

    masks = []
    token_of_interest_mask = None
    for feat_path in features_path:
        data = torch.load(feat_path, map_location="cpu")
        token_of_interest_mask = data.get("token_of_interest_mask", None)
        if token_of_interest_mask is not None:
            masks.append(torch.cat(token_of_interest_mask, dim=0))
        else:
            # All features should have token_of_interest_mask for this function to be valid
            return None
    masks = torch.stack(masks, dim=0)
    token_of_interest_mask = torch.all(masks, dim=0)

    return token_of_interest_mask
