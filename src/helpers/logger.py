import logging
import os

__all__ = ["setup_logger", "log_args"]


def setup_logger(log_file=None, level=logging.INFO):
    # Create a custom logger
    logger = logging.getLogger("train_test_logger")
    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler()  # Log to console
    console_handler.setLevel(level)

    # Optional: Log to file as well
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

    # Create formatters and add them to handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    if log_file:
        file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)

    if log_file:
        logger.addHandler(file_handler)

    return logger


def log_args(args, logger):
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
