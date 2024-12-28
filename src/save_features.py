import argparse
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import torch

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables,
                           compute_time_left, set_seed, setup_hooks,
                           update_dict_of_list)
from models import get_model_class
from models.image_text_model import ImageTextModel


@torch.no_grad()
def inference(
    loader: Callable,
    model_class: ImageTextModel,
    hook_return_function: Callable,
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Tuple[List[Dict[str, Any]], List[bool]]:

    num_iterations = len(loader)
    hook_data = {}
    model = model_class.get_model()
    for name, module in model.named_modules():
        print (name)
    start_time = time.time()
    for i, item in enumerate(loader):

        text = item["text"][0]  # for now we support batch size = 1
        image_path = item["image"][0]
        inputs = model_class.preprocessor(
            instruction=text,
            image_file=image_path,
            response="",
            generation_mode=args.generation_mode,
        )

        if args.generation_mode:
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        else:
            out = model(**inputs).logits

        item["model_output"] = out
        input_len = (
            inputs["input_ids"].shape[1]
            if inputs["input_ids"].ndim > 1
            else inputs["input_ids"].shape[0]
        )
        item["model_generated_output"] = out[:, input_len:]
        item["model_predictions"] = model_class.get_tokenizer().batch_decode(
            out[:, input_len:], skip_special_tokens=True
        )

        if hook_return_functions is not None:
            for func in hook_return_functions:
                if func is not None:
                    hook_output = func(**item)
                    if hook_output:
                        item.update(hook_output)

        hook_data = update_dict_of_list(item, hook_data)
        clear_hooks_variables()
        if (i + 1) % 100 == 0:
            time_left = compute_time_left(start_time, i, num_iterations)
            logger.info(
                f"Iteration: {i}/{num_iterations},  Estimated time left: {time_left:.2f} mins"
            )
    return hook_data


if __name__ == "__main__":

    args = get_arguments()

    logger = setup_logger(log_file=os.path.join(args.save_dir, f"logs.log"))

    set_seed(args.seed)

    logger.info(f"Loading model: {args.model_name_or_path}")
    log_args(args, logger)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
    )

    hook_return_functions, hook_postprocessing_functions = setup_hooks(
        model=model_class.model_,
        modules_to_hook=args.modules_to_hook,
        hook_names=args.hook_names,
        tokenizer=model_class.get_tokenizer(),
        logger=logger,
        args=args,
    )
    loader = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args
    )

    hook_data = inference(
        loader=loader,
        model_class=model_class,
        device=device,
        hook_return_function=hook_return_functions,
        logger=logger,
        args=args,
    )

    clear_forward_hooks(model_class.model_)
    if hook_postprocessing_functions is not None:
        for func in hook_postprocessing_functions:
            if func is not None:
                func(data=hook_data, args=args, logger=logger)
