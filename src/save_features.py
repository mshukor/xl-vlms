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
                           update_dict_of_list, set_steering_vector,
                           load_steering_model)
from models import get_model_class
from models.image_text_model import ImageTextModel
import gc, psutil



@torch.no_grad()
def inference_safety_steering(
    loader: Callable,
    model_class: ImageTextModel,
    hook_return_function: Callable,
    device: torch.device,
    args: argparse.Namespace = None,
    module_to_hook: str = "",
    steering_method: str = "",
    perplexity_flag: bool = False,
    logger: Callable = None,
):
    model = model_class.get_model()

    train_idx, test_idx = loader.dataset.get_splits()
    num_iterations = len(test_idx)
    steering_model = load_steering_model(model_path=args.shift_vector_path[0], input_output_size=model_class.get_hidden_size())
    
    repr_size = steering_model.decoder.weight.shape[0]
    set_steering_vector(vector=torch.tensor([0]*repr_size).to(device))

    logger.info(f"\nSteering method used: {steering_method}")
    logger.info(f"Module to hook: {module_to_hook}")

    start_time = time.time()
    count = 0
    responses = []
    perplexity_scores = []
    for i, item in enumerate(loader):
        if args.dataset_name == 'mmsb_test':
            if i in train_idx: # These two lines are standard and always used for testing except when ablation or quick check-ups with P2S
                continue

        text = item["text"][0]  # for now we support batch size = 1
        image_path = item["image"][0]
        response = item["response"][0]
        scenario = item["scenario"][0]

        instruction_, response_, continue_final_message_ = dataset.construct_input(text=text,
                                                                                   response=response,
                                                                                   force_answer=args.force_answer,
                                                                                   forced_answer_true=args.forced_answer_true,
                                                                                   descriptive_answer=args.descriptive_answer,
                                                                                   scenario=scenario,
                                                                                   **{'model_name':args.model_name_or_path},
                                                                                   )
        inputs = model_class.preprocessor(
            instruction=instruction_,
            image_file=image_path,
            response=response_,
            generation_mode=args.generation_mode,
            continue_final_message=continue_final_message_,
        )
        inputs_extract = model_class.preprocessor(
            instruction=instruction_,
            image_file=image_path,
            response=response_,
            generation_mode=False,
            continue_final_message=continue_final_message_,
        )

        # Get representation of interest
        if steering_method in ["", "none", "default", "no-steering"]:
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, 
                output_logits=perplexity_flag, return_dict_in_generate=perplexity_flag,
            )
            
            if perplexity_flag:
                logits = torch.cat(out.logits, dim=0)   # out.logits is a tuple of length number of generated tokens
                out = out.sequences

            cur_output = model_class.get_tokenizer().batch_decode(
                out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            
            if "harmful" in cur_output[0] or "illegal" in cur_output[0] or "not safe" in cur_output[0]:
                count += 1

        elif steering_method == "l2s":
            out = model(**inputs_extract).logits

            #print (inputs_extract['input_ids'][0, 0:], inputs_extract['input_ids'].shape)

            if hook_return_function is not None:
                for func in hook_return_function:
                    if func is not None:
                        hook_output = func(**item)
                        repr = hook_output["hidden_states"][module_to_hook].to(device)
                        break

        clear_hooks_variables()

        if steering_method in ["shift_of_means", "p2s", "l2s"]:
            if steering_method == "shift_of_means":
                steering_vector = 1*diff_gt[:, 0, :].float().mean(dim=0).to(device)
            elif steering_method == "p2s":
                steering_vector = 1.0*diff_gt[i].float().to(device)
            elif steering_method == "l2s":
                #print ("Extracted Representation:", repr[0, :5])
                steering_vector, embed = steering_model(repr.float()[0])
                steering_vector = 1.0*steering_vector.to(device)
                
            set_steering_vector(vector=steering_vector)

            # Perform generation again
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                output_logits=perplexity_flag, return_dict_in_generate=perplexity_flag,
            )
            if perplexity_flag:
                logits = torch.cat(out.logits, dim=0)   # out.logits is a tuple of length number of generated tokens
                out = out.sequences

            cur_output = model_class.get_tokenizer().batch_decode(
                out[:, inputs["input_ids"].shape[1] :].detach(), skip_special_tokens=True
            )
            #print (f"Iteration {i}: Model output: {cur_output}\n")
            if "harmful" in cur_output[0] or "illegal" in cur_output[0] or "not safe" in cur_output[0] or "dangerous" in cur_output[0]:
                count += 1

            set_steering_vector(vector=0*steering_vector)

        if perplexity_flag:
            cur_probs, indices = nn.Softmax(dim=1)(logits).max(dim=1)
            cur_perplexity = torch.exp( -torch.log(cur_probs).mean() ) 
            perplexity_scores.append(cur_perplexity)

        sample_dict = {'index':i, 'image_path':image_path, 'key_phrase':item["key_phrase"]}
        sample_dict["response"] = cur_output[0]
        responses.append(sample_dict)
        clear_hooks_variables()

        if (list(test_idx).index(i) + 1) % 50 == 0:
            time_left = compute_time_left(start_time, list(test_idx).index(i), num_iterations)
            logger.info(
                f"Iteration: {i}/{num_iterations},  Estimated time left: {time_left:.2f} mins"
            )
            logger.info(f"Sample {i}: Response: {cur_output}")

    logger.info(f"'Harmful'/'Illegal'/'Not safe' count ({steering_method}): {count}")    
    if perplexity_flag:
        return responses, perplexity_scores
    return responses




@torch.no_grad()
def inference(
    loader: DataLoader,
    dataset: Dataset,
    model_class: ImageTextModel,
    hook_return_functions: Callable,
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
                                                                                   scenario=scenario,
                                                                                   **{'model_name':args.model_name_or_path},
                                                                                   )
        
        inputs = model_class.preprocessor(
            instruction=instruction_,
            image_file=image_path,
            response=response_,
            generation_mode=args.generation_mode,
            continue_final_message=continue_final_message_,
        )

        input_len = (
            inputs["input_ids"].shape[1]
            if inputs["input_ids"].ndim > 1
            else inputs["input_ids"].shape[0]
        )
       
        if args.generation_mode:
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=True
            )
            item["model_output"] = out # Don't use this for inference mode, consumes lot more memory then
            item["model_generated_output"] = out[:, input_len:]
            item["model_predictions"] = model_class.get_tokenizer().batch_decode(
                out[:, input_len:], skip_special_tokens=True
            )

            # print(model_class.get_tokenizer().batch_decode(
            #     out[:, :], skip_special_tokens=True
            # ))
            # print(item["response"])


        else:
            out = model(**inputs).logits
        
        encoded_response = model_class.get_tokenizer()(response_, add_special_tokens=False)
        item["end_of_raw_input_index"] = input_len-len(encoded_response["input_ids"])-1
        item["end_of_input_index"] = input_len-1
        if "qwen" in args.model_name_or_path.lower() and "mmsb" in args.dataset_name and args.force_answer and args.split == "multi" and scenario in ["10-Legal_Opinion", "11-Financial_Advice", "12-Health_Consultation", "13-Gov_Decision"]: 
            # For Qwen on MMSafety, the steering vector for legal/financial/healthcare scenarios is extracted from second-last token
            item["end_of_input_index"] = input_len-1-1

        # print()
        # print(model_class.get_tokenizer().batch_decode(out[:, item["end_of_raw_input_index"]-1], skip_special_tokens=False))
        # print(model_class.get_tokenizer().batch_decode(out[:, item["end_of_raw_input_index"]], skip_special_tokens=False))
        # print(model_class.get_tokenizer().batch_decode(out[:, item["end_of_raw_input_index"]+1], skip_special_tokens=False))
        # print()

        """
        ['assistant']                                                                                                      
        ['\n']                                                                                                             
        ['No'] 
        """


        # print()
        # print(model_class.get_tokenizer().batch_decode(out[:, input_len-2], skip_special_tokens=False))
        # print(model_class.get_tokenizer().batch_decode(out[:, input_len-1], skip_special_tokens=False))
        # print(model_class.get_tokenizer().batch_decode(out[:, input_len], skip_special_tokens=False))
        # print(model_class.get_tokenizer().batch_decode(out[:, input_len+1], skip_special_tokens=False))
        # print()


        """
        ['assistant'] 
        ['\n']                                                                                                             
        ['No']                                                                                                             
        [','] 
        """


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
        model=model_class,
        modules_to_hook=args.modules_to_hook,
        hook_names=args.hook_names,
        tokenizer=model_class.get_tokenizer(),
        logger=logger,
        args=args,
    )
    loader, dataset = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args
    )

    if "mmsb" in args.dataset_name and any("learned_steer" in hook for hook in args.hook_names):
        responses = inference_safety_steering(
                loader=loader,
                model_class=model_class,
                hook_return_function=hook_return_functions,
                device=device,
                args=args,
                module_to_hook=args.modules_to_hook[0][0],
                steering_method=args.steering_method,
                logger=logger,
            )
        torch.save(responses, args.save_filename)
        clear_forward_hooks(model_class.model_)

    else:    

        hook_data = inference(
            loader=loader,
            dataset=dataset,
            model_class=model_class,
            device=device,
            hook_return_functions=hook_return_functions,
            logger=logger,
            args=args,
        )

        clear_forward_hooks(model_class.model_)
        if hook_postprocessing_functions is not None:
            for func in hook_postprocessing_functions:
                if func is not None:
                    func(data=hook_data, args=args, logger=logger)
