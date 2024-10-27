import os

import torch

from analysis import load_features
from metrics import dictionary_learning_evaluation
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from models import get_model_class

if __name__ == "__main__":

    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, f"logs.log"))
    log_args(args, logger)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # GPU might be needed for CLIPScore/BERTScore

    if args.features_path is not None:
        features, metadata = load_features(
            features_path=args.features_path,
            feature_key=args.feature_key,
            logger=logger,
            args=args,
        )
        
    model_class = None
    if args.model_name is not None:
        # Don't need model on GPU 
        model_class = get_model_class(
            args.model_name, device=torch.device("cpu"), logger=logger, args=args
        )

    dictionary_learning_evaluation(
        metric_name=args.evaluation_name,
        features=features,
        metadata=metadata,
        model_class=model_class,
        logger=logger,
        args=args,
        device=device,
    )

    
