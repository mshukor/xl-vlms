import os

import torch

from analysis import analyse_features, load_features
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from models import get_model_class

if __name__ == "__main__":

    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, f"logs.log"))
    log_args(args, logger)

    features, metadata = load_features(
        features_path=args.features_path,
        feature_key=args.feature_key,
        logger=logger,
        args=args,
    )

    device = torch.device("cpu")

    model_class = get_model_class(
        args.model_name, device=device, logger=logger, args=args
    )

    analyse_features(
        features=features,
        analysis_name=args.analysis_name,
        logger=logger,
        model_class=model_class,
        metadata=metadata,
        device=device,
        args=args,
    )
