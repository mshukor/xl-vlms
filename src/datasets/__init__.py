import argparse
from typing import Any, Callable, Tuple

from torch.utils.data import DataLoader, Subset

from datasets.image_text_dataset import COCODataset

__all__ = ["get_dataset_loader"]


def get_dataset_loader(
    dataset_name: str,
    args: argparse.Namespace = None,
    logger: Callable = None,
    **kwargs: Any,
) -> Tuple[Callable, list]:
    """
    General function to return a DataLoader for a given dataset.

    This function also has the functionality of extracting of a subset of the dataset based on

    Args:
        dataset_name (str): The name of the dataset.
        args (argparse.Namespace): Arguments including paths, flags, and dataset options.

    Returns:
        loader (DataLoader): The DataLoader object for the (subset of) dataset.
        indices (list): The indices of the subset (if applicable) or an empty list.
    """

    batch_size = getattr(args, "batch_size", 1)

    if dataset_name == "coco":
        dataset_cls = COCODataset
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented.")

    dataset = dataset_cls(
        data_dir=args.data_dir,
        annotation_file=args.annotation_file,
        split=args.split,
        dataset_size=args.dataset_size,
        seed=args.seed,
        dataset_name=args.dataset_name,
        mode=("val" if args.generation_mode else "train"),
        prompt_template=args.prompt_template,
    )

    loader = None
    token_of_interest_indices = []

    if args.select_token_of_interest_samples:
        token_of_interest = args.token_of_interest
        token_of_interest_indices = dataset.token_of_interest_idx_extractor(
            token_of_interest
        )
        dataset_subset = Subset(dataset, token_of_interest_indices)

        loader = DataLoader(
            dataset_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    if logger is not None:
        logger.info(f"Reading dataset: {dataset_name} of size: {len(loader)}")
    assert len(loader) > 0, f"loader is empty"
    return loader
