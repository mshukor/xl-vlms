import argparse
from typing import Callable, Tuple

import torch

__all__ = ["get_model_class"]


SUPPORTED_MODELS = [
    "llava-1.5",
    "Qwen/Qwen2-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "allenai/Molmo-7B-D-0924",
]


def get_model_class(
    model_name_or_path: str = "llava-hf/llava-1.5-7b-hf",
    processor_name: str = "llava-hf/llava-1.5-7b-hf",
    device: torch.device = torch.device("cuda"),
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Tuple[Callable]:

    if "llava-1.5" in model_name_or_path:
        from models.llava import LLaVA

        model_class = LLaVA(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
            cache_dir=args.cache_dir,
        )
    elif "Qwen" in model_name_or_path:
        from models.qwen_vl import QwenVL

        model_class = QwenVL(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
        )
    elif "idefics" in model_name_or_path:
        from models.idefics2 import IDEFICS

        model_class = IDEFICS(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
        )
    elif "Molmo" in model_name_or_path:
        from models.molmo import Molmo

        model_class = Molmo(
            model_name_or_path=model_name_or_path,
            processor_name=processor_name,
            local_files_only=args.local_files_only,
        )
    else:
        raise NotImplementedError(
            f"Got {model_name_or_path}, but only these models are supported {SUPPORTED_MODELS}"
        )

    if logger is not None:
        logger.info(f"Successfully loaded {model_name_or_path}, device: {device}")

    model_class.model_.to(device)

    return model_class
