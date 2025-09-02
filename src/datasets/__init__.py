import argparse
import pickle
from typing import Any, Callable, Tuple

from torch.utils.data import DataLoader, Subset, Dataset

from datasets.image_text_dataset import (COCODataset, VQAv2Dataset,
                                         POPE_test_Dataset, POPE_train_Dataset, MMSafetyBench)

__all__ = ["get_dataset_loader"]


def get_dataset_loader(
    dataset_name: str,
    args: argparse.Namespace = None,
    logger: Callable = None,
    **kwargs: Any,
) -> Tuple[DataLoader, Dataset]:
    """
    General function to return a DataLoader for a given dataset.

    This function also has the functionality of extracting of a subset of the dataset based on

    Args:
        dataset_name (str): The name of the dataset.
        args (argparse.Namespace): Arguments including paths, flags, and dataset options.

    Returns:
        loader (DataLoader): The DataLoader object for the (subset of) dataset.
    """

    batch_size = getattr(args, "batch_size", 1)

    if dataset_name == "coco":
        dataset_cls = COCODataset
    elif dataset_name == "vqav2":
        dataset_cls = VQAv2Dataset
    elif dataset_name == "pope_test":
        dataset_cls = POPE_test_Dataset
    elif dataset_name == "pope_train":
        dataset_cls = POPE_train_Dataset
    elif "mmsb" in dataset_name:
        dataset_cls = MMSafetyBench
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented.")

    dataset = dataset_cls(
        data_dir=args.data_dir,
        annotation_file=args.annotation_file,
        questions_file=args.questions_file,
        split=args.split,
        dataset_size=args.dataset_size,
        seed=args.seed,
        dataset_name=args.dataset_name,
        mode=("val" if args.generation_mode else "train"),
        prompt_template=args.prompt_template,
        token_of_interest_num_samples=args.token_of_interest_num_samples,
    )

    loader = None
    final_dataset = None
    token_of_interest_indices = []

    if logger is not None:
        logger.info(f"Successfully loaded {dataset_name}")

    if args.select_token_of_interest_samples:
        token_of_interest_indices = dataset.token_of_interest_idx_extractor(
            token_of_interest=args.token_of_interest,
            token_of_interest_key=args.token_of_interest_key,
            allow_different_variations=args.allow_different_variations_of_token_of_interest,
            token_of_interest_class=args.token_of_interest_class,
            logger=logger,
        )
        dataset_subset = Subset(dataset, token_of_interest_indices)

        loader = DataLoader(
            dataset_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        final_dataset = dataset_subset
    elif args.select_samples_from_ids:

        assert (
            args.path_to_samples_ids is not None
        ), 'A path to ids should be given when passing the flag "select_samples_from_ids"'
        with open(args.path_to_samples_ids, "rb") as handle:
            img_ids = pickle.load(handle)

        filtered_indices = dataset.indices_from_ids_extractor(img_ids=img_ids)
        dataset_subset = Subset(dataset, filtered_indices)

        loader = DataLoader(
            dataset_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        final_dataset = dataset_subset
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        final_dataset = dataset

        
    if logger is not None:
        logger.info(f"Reading dataset: {dataset_name} of size: {len(loader)}")
    assert len(loader) > 0, f"loader is empty"

    return loader, final_dataset
