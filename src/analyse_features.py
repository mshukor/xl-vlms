import os

import torch

from analysis import analyse_features
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from models import get_model_class

if __name__ == "__main__":

    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, f"logs.log"))
    log_args(args, logger)

    device = torch.device("cpu")

    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
    )

    analyse_features(
        analysis_name=args.analysis_name,
        logger=logger,
        model_class=model_class,
        device=device,
        args=args,
    )
