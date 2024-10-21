from typing import List

import torch

__all__ = ["get_token_of_interest_features"]



def get_token_of_interest_features(
    features: torch.Tensor, token_of_interest_mask: torch.Tensor = None
) -> torch.Tensor:

    if token_of_interest_mask is not None:
        if isinstance(token_of_interest_mask, list):
            token_of_interest_mask = torch.cat(token_of_interest_mask, dim=0)
        features = features[token_of_interest_mask].reshape(-1, features.shape[1])

    return features


