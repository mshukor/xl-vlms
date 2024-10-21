import argparse
from typing import Callable, Tuple

import torch

from models.llava import LLaVA

__all__ = ["get_model_class"]


SUPPORTED_MODELS = [
    "llava-1.5",
]


def get_model_class(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    processor_name: str = "llava-hf/llava-1.5-7b-hf",
    device: torch.device = torch.device("cuda"),
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Tuple[Callable]:

    if "llava-1.5" in model_name:
        model_class = LLaVA(
            model_name=model_name,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
        )
    else:
        raise NotImplementedError(f"Only these models are supported {SUPPORTED_MODELS}")

    model_class.model_.to(device)

    return model_class
