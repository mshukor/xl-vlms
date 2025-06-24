import argparse
import os
import time
from typing import Any, Callable, Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset
import torch

from datasets import get_dataset_loader
from datasets.constants import SAME_ANSWERS, OPPOSITE_ANSWERS
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables,
                           compute_time_left, set_seed, setup_hooks,
                           update_dict_of_list)
from models import get_model_class
from models.image_text_model import ImageTextModel
import gc, psutil


@torch.no_grad()
def inference(
    loader: DataLoader,
    dataset: Dataset,
    model_class: ImageTextModel,
    hook_return_function: Callable,
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Tuple[List[Dict[str, Any]], List[bool]]:

    num_iterations = len(loader)
    hook_data = {}
    model = model_class.get_model()
    start_time = time.time()
    for i, item in enumerate(loader):

        text = item["text"][0]  # for now we support batch size = 1
        image_path = item["image"][0]
        response = item["response"][0]
        scenario = item["scenario"][0]

        instruction_, response_, continue_final_message_ = dataset.construct_input(text=text,
                                                                                   response=response,
                                                                                   force_answer=args.force_answer,
                                                                                   forced_answer_true=args.forced_answer_true,
                                                                                   descriptive_answer=args.descriptive_answer,
                                                                                   scenario=scenario,)
        
        args.generation_mode = False
        inputs = model_class.preprocessor(
            instruction=instruction_,
            image_file=image_path,
            response=response_,
            generation_mode=args.generation_mode,
            continue_final_message=continue_final_message_,
        )

        #logger.info (f"Last 30 input token ids: Iteration: {i}, {inputs['input_ids'][:, -30:]}")
        #logger.info (f"Logging message for Memory available: Iteration: {i}, Memory: {psutil.virtual_memory().available}")

        if args.generation_mode:
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
            item["model_output"] = out # Don't use this for inference mode, consumes lot more memory then
            item["model_generated_output"] = out[:, input_len:]
            item["model_predictions"] = model_class.get_tokenizer().batch_decode(
                out[:, input_len:], skip_special_tokens=True
            )

        else:
            out = model(**inputs).logits

        input_len = (
            inputs["input_ids"].shape[1]
            if inputs["input_ids"].ndim > 1
            else inputs["input_ids"].shape[0]
        )
        
        encoded_response = model_class.get_tokenizer()(response_, add_special_tokens=False)
        item["end_of_raw_input_index"] = input_len-len(encoded_response["input_ids"])-1
        item["end_of_input_index"] = input_len-1


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
    loader, dataset = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args
    )

    hook_data = inference(
        loader=loader,
        dataset=dataset,
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
