import argparse
import os
import random
import re
import time
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch

__all__ = [
    "register_hooks",
    "clear_forward_hooks",
    "clear_hooks_variables",
    "hooks_postprocessing",
    "set_seed",
    "setup_hooks",
]


# Dictionary to store hidden states
HIDDEN_STATES = {}


def set_seed(seed_value=42):
    # Python random seed
    random.seed(seed_value)

    # NumPy random seed
    np.random.seed(seed_value)

    # PyTorch random seed
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_dict_of_list(item: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in item.items():
        if k in data:
            data[k].append(v)
        else:
            data[k] = [v]
    return data


def fmatch(name: str, patterns: List[str], exact_match: bool = False) -> bool:
    if exact_match:
        return name in patterns
    else:
        # Convert patterns with '*' to proper regex expressions (where * means "any sequence of characters")
        regex_patterns = [
            re.compile(re.sub(r"\*", ".*", pattern)) for pattern in patterns
        ]
        return any([regex.search(name) for regex in regex_patterns])


def compute_time_left(start_time, iteration: int, num_iterations: int):
    elapsed_time = time.time() - start_time  # Time spent so far
    avg_time_per_iter = elapsed_time / iteration  # Average time per iteration
    remaining_iters = num_iterations - iteration
    time_left = avg_time_per_iter * remaining_iters  # Estimated time left
    return time_left / 60


def save_hidden_states(module_name: str = "", **kwargs: Any):
    """
    Save module output hidden states. In case of autoregressive, make sure the kv caching is enabled.
    """
    global HIDDEN_STATES

    def hook(module, input, output):
        if isinstance(output, tuple):  # e.g residual streams output is a tuple
            output = output[0]
        output = output.detach().cpu()
        if module_name in HIDDEN_STATES:
            HIDDEN_STATES[module_name].append(output)
        else:
            HIDDEN_STATES[module_name] = [output]

    return hook


def extract_token_of_interest_states(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    token_of_interest: str,
    tokenizer: Callable,
    token_of_interest_idx: Union[int, torch.Tensor] = None,
    token_of_interest_start_token: int = 0,
) -> Tuple[torch.Tensor]:

    if pred_tokens.shape[1] > tokens.shape[1]:
        pred_tokens = pred_tokens[
            :, -tokens.shape[1] :
        ]  # e.g. in case of language_model.lm_head only the hidden states for generated tokens are saved
    elif pred_tokens.shape[1] < tokens.shape[1]:
        tokens = tokens[:, -pred_tokens.shape[1] :]

    assert (
        token_of_interest is not None
    ), f"Please provide the token_of_interest, got {token_of_interest}"

    if token_of_interest_start_token != 0:
        # e.g. consider only te answers
        tokens = tokens[:, token_of_interest_start_token:]
        pred_tokens = pred_tokens[:, token_of_interest_start_token:]

    if token_of_interest_idx is None:
        # If the token_of_interest splits into different ids, we consider the first one (while skipping eos/bos tokens)
        tokens_of_interest = set(
            [
                token_of_interest,
                token_of_interest.capitalize(),
                token_of_interest.lower(),
            ]
        )
        token_of_interest_idx = torch.tensor(
            [
                tokenizer.encode(tok, add_special_tokens=False)[0]
                for tok in tokens_of_interest
            ]
        ).to(pred_tokens.device)
    # Step 1: Find where the tokens of interest exist in the batch (B, L)
    token_of_interest_batch_presence = torch.isin(
        pred_tokens, token_of_interest_idx
    )  # (B, L)
    # Step 2: Get the first occurrence index for each sequence
    token_of_interest_batch_first_pos = torch.argmax(
        token_of_interest_batch_presence.long(), dim=1
    )  # (B,)

    # Step 3: Mask for sequences with no token of interest
    no_token_found_mask = ~token_of_interest_batch_presence.any(dim=1)

    # Set the position to -1 if no token of interest is found
    token_of_interest_batch_first_pos[no_token_found_mask] = -1

    # Step 4: Now handle indexing into `v` based on the first position
    # Extract v at the first position for each batch (B,)
    # Select only valid positions in `v`
    v_selected = tokens[
        range(tokens.shape[0]),
        token_of_interest_batch_first_pos.clamp(min=0).to(tokens.device),
    ].unsqueeze(1)
    return v_selected, ~no_token_found_mask


def get_hidden_states(
    token_idx: int = None,
    token_start_end_idx: List[List[int]] = None,
    extract_token_of_interest: bool = False,
    token_of_interest_start_token: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    hidden_states = {}
    output = {}
    for k, v in HIDDEN_STATES.items():
        if isinstance(v, list) and len(v) > 1:
            v = torch.cat(v, dim=1)
        else:
            v = v[0]
        if token_idx is not None:
            v = v[:, token_idx, :].unsqueeze(1)
        elif token_start_end_idx is not None:
            v = v[:, int(token_start_end_idx[0]) : int(token_start_end_idx[1]), :]
        elif extract_token_of_interest:
            v, token_of_interest_mask = extract_token_of_interest_states(
                tokens=v,
                pred_tokens=kwargs["model_output"],
                token_of_interest=kwargs["token_of_interest"],
                tokenizer=kwargs["tokenizer"],
                token_of_interest_idx=kwargs.get("token_of_interest_idx", None),
                token_of_interest_start_token=token_of_interest_start_token,
            )
            output["token_of_interest_mask"] = token_of_interest_mask
            output["image_paths"] = kwargs["image"]
        else:
            pass
        hidden_states[k] = v
    output["hidden_states"] = hidden_states
    return output


def save_hidden_states_to_file(
    data: Dict[str, Any],
    data_keys: List[str] = ["hidden_states"],
    hook_name: str = "",
    args: argparse.Namespace = None,
    logger: Callable = None,
) -> None:
    saved_data = {}
    for data_key in data.keys():
        if data_key in data_keys:
            assert (
                data_key in data
            ), f"{data_key} not found in data, there is only: {data.keys()}"

            saved_data[data_key] = data[data_key]  # List[Any]
    file_name = os.path.join(args.save_dir, f"{hook_name}_{args.save_filename}.pth")
    torch.save(saved_data, file_name)
    if logger is not None:
        logger.info(f"Saving data to: {file_name}")


def register_hooks(
    model: Callable,
    modules_to_hook: List[str],
    hook_name: str = "save_hidden_states",
    tokenizer: Callable = None,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Callable:
    hook_function, hook_return_function = None, None
    if "save_hidden_states" == hook_name:
        # Save the hidden states of all tokens in the sequence
        hook_function = save_hidden_states
        hook_return_function = get_hidden_states
    elif "save_hidden_states_given_token_idx" == hook_name:
        # Save the hidden states at given token index
        hook_function = save_hidden_states
        hook_return_function = partial(get_hidden_states, token_idx=args.token_idx)
    elif "save_hidden_states_given_token_start_end_idx" == hook_name:
        # Save the hidden states of tokens between start and end index
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states, token_start_end_idx=args.token_start_end_idx
        )
    elif "save_hidden_states_for_token_of_interest" == hook_name:
        # Save the hidden states of tokens between start and end index
        hook_function = save_hidden_states

        hook_return_function = partial(
            get_hidden_states,
            extract_token_of_interest=True,
            token_of_interest=args.token_of_interest,
            token_of_interest_idx=args.token_of_interest_idx,
            token_of_interest_start_token=args.token_of_interest_start_token,
            tokenizer=tokenizer,
        )
    else:
        warnings.warn(f"{hook_name} is not supported. No hooks attached to model.")
    if hook_function is not None:
        hooked_modules = []
        for name, module in model.named_modules():
            if fmatch(
                name, modules_to_hook, exact_match=args.exact_match_modules_to_hook
            ):
                module.register_forward_hook(hook_function(module_name=name))
                hooked_modules.append(name)
        if logger is not None:
            logger.info(f"hooked_modules: {hooked_modules}")

    return hook_return_function


def hooks_postprocessing(
    hook_name: str = "save_hidden_states", args: argparse.Namespace = None
) -> Callable:
    hook_postprocessing_function = None
    if "save_hidden_states" in hook_name:
        data_keys = ["hidden_states", "image_paths", "model_predictions", "targets"]
        if "token_of_interest" in hook_name:
            data_keys.append("token_of_interest_mask")
        hook_postprocessing_function = partial(
            save_hidden_states_to_file,
            args=args,
            data_keys=data_keys,
            hook_name=hook_name,
        )
    else:
        warnings.warn(f"{hook_name} is not supported. No hooks attached to model.")

    return hook_postprocessing_function


def clear_forward_hooks(model: Callable) -> None:
    for module in model.modules():
        module._forward_hooks.clear()


def clear_hooks_variables():
    global HIDDEN_STATES
    HIDDEN_STATES = {}


def setup_hooks(
    model: Callable,
    modules_to_hook: List[str],
    hook_names: str,
    tokenizer: Callable = None,
    logger: Callable = None,
    args: argparse.Namespace = None,
):
    hook_return_functions, hook_postprocessing_functions = [], []
    for i, hook_name in enumerate(hook_names):
        if modules_to_hook is not None and i < len(modules_to_hook):
            modules_to_hook_ = modules_to_hook[i]
            assert isinstance(
                modules_to_hook_, list
            ), f"modules_to_hook_ must be of type list. modules_to_hook_: {modules_to_hook_}"
            hook_return_function = register_hooks(
                model=model,
                modules_to_hook=modules_to_hook_,
                hook_name=hook_name,
                tokenizer=tokenizer,
                logger=logger,
                args=args,
            )
        else:
            hook_return_function = None
        hook_postprocessing_function = hooks_postprocessing(
            hook_name=hook_name, args=args
        )

        hook_return_functions.append(hook_return_function)
        hook_postprocessing_functions.append(hook_postprocessing_function)

    return hook_return_functions, hook_postprocessing_functions
