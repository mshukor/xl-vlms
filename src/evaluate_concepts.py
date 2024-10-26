import os

import torch

from analysis import load_features
from datasets import get_dataset_loader
from metrics import dictionary_learning_evaluation
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from models import get_model_class

if __name__ == "__main__":

    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, f"logs.log"))
    log_args(args, logger)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loader = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args
    )

    if args.features_path is not None:
        features, metadata = load_features(
            features_path=args.features_path,
            feature_key=args.feature_key,
            logger=logger,
            args=args,
        )

    dictionary_learning_evaluation(
        metric_name=args.evaluation_name,
        features=features,
        loader=loader,
        metadata=metadata,
        logger=logger,
        args=args,
        device=device,
    )

    
