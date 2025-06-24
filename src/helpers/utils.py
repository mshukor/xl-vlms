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
import torch.nn as nn
from tqdm import tqdm

import metrics
from datasets.constants import WORDS

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


def append_item_to_dict_of_list(key: str, value: Any, dictionary: Dict[str, Any]):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]
    return dictionary


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


def get_start_idx_generated_tokens(tokens: List[torch.Tensor]) -> int:
    if isinstance(tokens, list) and len(tokens) > 1:
        total_len = torch.cat(tokens, dim=1).shape[1]
        idx = tokens[0].shape[1] - total_len
    else:
        # teacher forcing mode
        v = v[0]
        idx = 0
    return idx  # generated tokens start after the prompt, count from last


class SteeringNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(input_size, hidden_size, bias=True)
        self.decoder = nn.Linear(hidden_size, output_size, bias=False)
        self.activ = nn.Tanh()
        
    def forward(self, inp):
        embed = self.activ(self.encoder(inp))
        shift = self.decoder(embed)
        return shift, embed

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


global SAMPLE_COUNTER
def apply_steering_vector(
    x: torch.Tensor,
    vector: torch.Tensor,
    alpha: float = 1,
    only_generated_tokens: bool = False,
    include_last_prompt_token: bool = False,
    start_prompt_token_idx: int = 0,
    individual_shift: bool = False,
) -> torch.Tensor:
    global SAMPLE_COUNTER

    if x.shape[1] > 1:
        if individual_shift: # when the vector is composed of #samples vectors, and for each samples a different steering vector is to be picked
            SAMPLE_COUNTER += 1
            vector = vector[SAMPLE_COUNTER-1]
        if only_generated_tokens:
            return x
        if include_last_prompt_token:
            start_prompt_token_idx = -1
        if start_prompt_token_idx > 0 or start_prompt_token_idx == -1:
            x_ = x[:, start_prompt_token_idx:, :]
            x_ = x_ + alpha * vector.to(x_.device).to(x_.dtype)
            x[:, start_prompt_token_idx:, :] = x_
            return x
        
    if individual_shift:
        if vector.shape[0] > 1:
            vector = vector[SAMPLE_COUNTER-1]
    x = x + alpha * vector.to(x.device).to(x.dtype)
    # import ipdb; ipdb.set_trace()
    return x



global PREDICTED_STEER
def apply_learned_steering_vector_steer(
    x: torch.Tensor,
    model: Any,
    alpha: float = 1,
    only_generated_tokens: bool = False,
    include_last_prompt_token: bool = False,
    start_prompt_token_idx: int = 0,
) -> torch.Tensor:
    global PREDICTED_STEER
    
    
    if x.shape[1] > 1:
        
        last_input_tokens = x[:,-1,:]
        last_input_tokens = last_input_tokens.to(dtype=torch.float16)

        PREDICTED_STEER = model(last_input_tokens)[0]
        vector = PREDICTED_STEER

        if only_generated_tokens:
            return x
        if include_last_prompt_token:
            start_prompt_token_idx = -1
        if start_prompt_token_idx > 0 or start_prompt_token_idx == -1:
            x_ = x[:, start_prompt_token_idx:, :]
            x_ = x_ + alpha * vector.to(x_.device).to(x_.dtype)
            x[:, start_prompt_token_idx:, :] = x_
            return x
        
    vector = PREDICTED_STEER
    x = x + alpha * vector.to(x.device).to(x.dtype)

    return x


def shift_hidden_states(
    vector: torch.Tensor = None,
    operation: str = "add",
    alpha: float = 1,
    only_generated_tokens: bool = False,
    include_last_prompt_token: bool = False,
    start_prompt_token_idx: int = 0,
    individual_shift: bool = False,
    **kwargs: Any,
):
    """
    Shift features in the vector's direction.
    """
    if "add" in operation:

        def hook(module, input, output):
            if isinstance(output, tuple):  # e.g. in the residual stream
                output_ = apply_steering_vector(
                    output[0],
                    vector,
                    alpha=alpha,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                    individual_shift=individual_shift,
                )
                return (output_,) + output[1:]
            else:
                output = apply_steering_vector(
                    output,
                    vector,
                    alpha=alpha,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                    individual_shift=individual_shift,
                )
                return output
            
    elif "learned_steer" in operation:

        def hook(module, input, output):
            if isinstance(output, tuple):  # e.g. in the residual stream
                output_ = apply_learned_steering_vector_steer(
                    output[0],
                    model=vector,
                    alpha=alpha,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                )
                return (output_,) + output[1:]
            else:
                output = apply_learned_steering_vector_steer(
                    output,
                    model=vector,
                    alpha=alpha,
                    only_generated_tokens=only_generated_tokens,
                    include_last_prompt_token=include_last_prompt_token,
                    start_prompt_token_idx=start_prompt_token_idx,
                )
                return output
            

    else:
        raise NotImplementedError(
            f"Only the following steering operation are supported: add, got {operation}"
        )

    return hook


def extract_token_of_interest_states(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    token_of_interest_idx: Union[int, torch.Tensor] = None,
    token_of_interest_start_token: int = 0,
) -> Tuple[torch.Tensor]:

    if token_of_interest_start_token != 0:
        # e.g. consider only te answers
        tokens = tokens[:, token_of_interest_start_token:]
        pred_tokens = pred_tokens[:, token_of_interest_start_token:]

    # Concider only text, no preds tokens for image tokens
    if pred_tokens.shape[1] > tokens.shape[1]:
        pred_tokens = pred_tokens[
            :, -tokens.shape[1] :
        ]  # e.g. in case of language_model.lm_head only the hidden states for generated tokens are saved
    elif pred_tokens.shape[1] < tokens.shape[1]:
        tokens = tokens[:, -pred_tokens.shape[1] :]

    assert (
        token_of_interest_idx is not None
    ), f"Please provide the token_of_interest_idx, got {token_of_interest_idx}"

    # If the token_of_interest splits into different ids, we consider the first one (while skipping eos/bos tokens)
    if not isinstance(token_of_interest_idx, torch.Tensor):
        token_of_interest_idx = torch.tensor([token_of_interest_idx])
    token_of_interest_idx = token_of_interest_idx.to(pred_tokens.device)

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


def extract_states_before_special_tokens(
    tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    end_special_tokens: List[str],
    tokenizer: Callable,
    token_of_interest_start_token: int = 0,
) -> Tuple[torch.Tensor]:
    if token_of_interest_start_token != 0:
        # e.g. consider only te answers
        tokens = tokens[:, token_of_interest_start_token:]
        pred_tokens = pred_tokens[:, token_of_interest_start_token:]

    # Concider only text, no preds tokens for image tokens
    if pred_tokens.shape[1] > tokens.shape[1]:
        pred_tokens = pred_tokens[
            :, -tokens.shape[1] :
        ]  # e.g. in case of language_model.lm_head only the hidden states for generated tokens are saved
    elif pred_tokens.shape[1] < tokens.shape[1]:
        tokens = tokens[:, -pred_tokens.shape[1] :]

    assert end_special_tokens is not None and isinstance(
        end_special_tokens, list
    ), f"Please provide the list of token_of_interest, got {end_special_tokens}"

    # If the token_of_interest splits into different ids, we consider the first one (while skipping eos/bos tokens)
    end_special_tokens_idx = torch.tensor(
        [
            tokenizer.encode(tok, add_special_tokens=False)[0]
            for tok in end_special_tokens
        ]
    ).to(pred_tokens.device)

    # Step 1: Find where the tokens of interest exist in the batch (B, L)
    token_of_interest_batch_presence = torch.isin(
        pred_tokens, end_special_tokens_idx
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
    v_selected = (
        tokens[
            range(tokens.shape[0]),
            : token_of_interest_batch_first_pos.to(tokens.device),
        ]
        .mean(1)
        .unsqueeze(1)
    )
    return v_selected, no_token_found_mask


def get_hidden_states(
    token_idx: int = None,
    token_start_end_idx: List[List[int]] = None,
    extract_token_of_interest: bool = False,
    token_of_interest_start_token: int = 0,
    extract_before_special_tokens: bool = False,
    extract_l2s_input_output: bool = False,
    save_only_generated_tokens: bool = False,
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

            if save_only_generated_tokens:
                start_idx_generated_tokens = -kwargs["model_generated_output"].shape[1]
                token_of_interest_start_token = start_idx_generated_tokens

            v, token_of_interest_mask = extract_token_of_interest_states(
                tokens=v,
                pred_tokens=kwargs["model_output"],
                token_of_interest_idx=kwargs.get("token_of_interest_idx", None),
                token_of_interest_start_token=token_of_interest_start_token,
            )
            output["token_of_interest_mask"] = token_of_interest_mask
            output["image"] = kwargs["image"]
        elif extract_before_special_tokens:

            if save_only_generated_tokens:
                start_idx_generated_tokens = -kwargs["model_generated_output"].shape[1]
                token_of_interest_start_token = start_idx_generated_tokens

            v, token_of_interest_mask = extract_states_before_special_tokens(
                tokens=v,
                pred_tokens=kwargs["model_output"],
                end_special_tokens=kwargs["end_special_tokens"],
                tokenizer=kwargs["tokenizer"],
                token_of_interest_start_token=token_of_interest_start_token,
            )
            output["token_of_interest_mask"] = torch.ones_like(
                token_of_interest_mask
            ).bool()
            output["image"] = kwargs["image"]

        elif extract_l2s_input_output:
            # end_of_raw_input_index corresponds to ":" after "ASSISTANT" token
            # it is used to extract the input and output representations from the right tokens, which does not include the forced answer
            end_of_raw_input_index = kwargs["end_of_raw_input_index"]
            end_of_input_index = kwargs["end_of_input_index"]

            # extracting the l2s inputs
            inputs = {"last_raw_input": v[:, end_of_raw_input_index, :].clone()}

            # extracting the l2s outputs
            average_tokens = torch.mean(v[:, end_of_raw_input_index+1:, :].clone(), dim=1).clone()
            
            last_input_tokens = v[:, end_of_input_index, :].clone()
            outputs = {"average" : average_tokens, "last_input": last_input_tokens}

            v = {"inputs": inputs, "outputs": outputs}
        else:
            pass

        if isinstance(v, dict):
            hidden_states[k] = v
        else:
            # v must be a torch tensor based on the above code. 
            # Clone to avoid memory leakage by keeping any reference to full original tensor
            hidden_states[k] = v.clone()

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
    file_name = os.path.join(
        args.save_dir, "features", f"{hook_name}_{args.save_filename}.pth"
    )
    torch.save(saved_data, file_name)
    if logger is not None:
        logger.info(f"Saving data to: {file_name}")


def save_analysis_to_file(
    data: Dict[str, Any],
    analysis_saving_path: str,
    data_keys: List[str] = ["text_grounding"],
    logger: Callable = None,
) -> None:
    saved_data = {}

    for data_key in data_keys:
        assert (
            data_key in data
        ), f"{data_key} not found in data, there is only: {data.keys()}"

        saved_data[data_key] = data[data_key]  # List[Any]
    file_name = f"{analysis_saving_path}.pth"
    torch.save(saved_data, file_name)
    if logger is not None:
        logger.info(f"Saving analysis data to: {file_name}")


def register_hooks(
    model: Callable,
    modules_to_hook: List[str],
    hook_name: str = "save_hidden_states",
    tokenizer: Callable = None,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Callable:
    global SAMPLE_COUNTER
    SAMPLE_COUNTER = 0
    
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
        token_of_interest = args.token_of_interest

        # Get index in tokenizer vocabulary for token of interest
        # Some tokenizers encode/decode space along with token, so include index of whitespace + token_of_interest
        tokens_of_interest = set(
            [
                token_of_interest,
                token_of_interest.capitalize(),
                token_of_interest.lower(),
                " " + token_of_interest,
            ]
        )
        token_of_interest_idx = args.token_of_interest_idx
        if token_of_interest_idx is None:
            token_of_interest_idx = torch.tensor(
                [
                    tokenizer.encode(tok, add_special_tokens=False)[0]
                    for tok in tokens_of_interest
                ]
            )
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_token_of_interest=True,
            token_of_interest_idx=token_of_interest_idx,
            token_of_interest_start_token=args.token_of_interest_start_token,
            save_only_generated_tokens=args.save_only_generated_tokens,
        )
    elif "save_hidden_states_for_token_of_interest_class" == hook_name:
        # Save the hidden states of tokens between start and end index
        token_of_interest = []
        tokens = list(WORDS[args.token_of_interest_class])
        for tok in tqdm(tokens):
            toks = [
                tok,
                tok.capitalize(),
                tok.lower(),
            ]
            token_of_interest.extend(toks)
        tokens_of_interest = list(set(token_of_interest))

        token_of_interest_idx = args.token_of_interest_idx
        if token_of_interest_idx is None:
            token_of_interest_idx = torch.tensor(
                [
                    tokenizer.encode(tok, add_special_tokens=False)[0]
                    for tok in tokens_of_interest
                ]
            )
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_token_of_interest=True,
            token_of_interest_idx=token_of_interest_idx,
            token_of_interest_start_token=args.token_of_interest_start_token,
            save_only_generated_tokens=args.save_only_generated_tokens,
        )
    elif "save_hidden_states_before_special_tokens" == hook_name:
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_before_special_tokens=True,
            end_special_tokens=args.end_special_tokens,
            tokenizer=tokenizer,
            save_only_generated_tokens=args.save_only_generated_tokens,
        )


    elif "save_hidden_states_for_l2s" == hook_name:
        hook_function = save_hidden_states
        hook_return_function = partial(
            get_hidden_states,
            extract_l2s_input_output=True,
        )
    elif "shift_hidden_states" in hook_name:
        operation = ""
        if "add" in hook_name:
            operation = "add"
        elif "learned_steer" in hook_name:
            operation = "learned_steer"
        else:
            raise NotImplementedError(
                f"Please provide a valid operation. Got {hook_name}"
            )

        only_generated_tokens = "only_generated" in hook_name
        include_last_prompt_token = "last_prompt_token" in hook_name


        if "learned_steer" in hook_name:

            # Re-create the model architecture (same input/output/hidden sizes)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_size, output_size, hidden_size = model.config.text_config.max_position_embeddings, model.config.text_config.max_position_embeddings, args.hidden_size
            model_steering = SteeringNet(input_size=input_size, output_size=output_size, hidden_size=hidden_size).to(device)

            model_steering.load_state_dict(torch.load(args.shift_vector_path[0]))
            dtype = next(model.parameters()).dtype
            model_steering = model_steering.to(dtype)

            model_steering.eval()
            vector = model_steering

        else:
            vector = torch.load(args.shift_vector_path[0])[args.shift_vector_key]

        hook_function = partial(
            shift_hidden_states,
            vector=vector,
            operation=operation,
            alpha=args.steering_alpha,
            only_generated_tokens=only_generated_tokens,
            include_last_prompt_token=include_last_prompt_token,
            start_prompt_token_idx=args.start_prompt_token_idx_steering,
            individual_shift=args.individual_shift
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
            logger.info(f"Apply {hook_name} to hooked_modules: {hooked_modules}")

    return hook_return_function


def hooks_postprocessing(
    hook_name: str = "save_hidden_states", args: argparse.Namespace = None
) -> Callable:
    hook_postprocessing_function = None
    if "save_hidden_states" in hook_name:

        data_keys = ["hidden_states", "image"]
        # temp change
        data_keys = ["hidden_states", "image", "model_predictions"]

        if "token_of_interest" in hook_name:
            data_keys.append("token_of_interest_mask")
        hook_postprocessing_function = partial(
            save_hidden_states_to_file,
            args=args,
            data_keys=data_keys,
            hook_name=hook_name,
        )
    elif "vqav2_accuracy" in hook_name:
        hook_postprocessing_function = metrics.get_metric(
            metric_name="vqav2_accuracy", args=args
        )

    elif "captioning_metrics" in hook_name:
        hook_postprocessing_function = metrics.get_metric(
            metric_name="captioning_metrics", args=args
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




