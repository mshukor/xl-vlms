import argparse

__all__ = ["get_arguments"]


def parse_list_of_lists(arg):
    list_of_lists = [lst.split(",") for lst in arg.split(";")]
    return list_of_lists


def get_arguments():
    parser = argparse.ArgumentParser(description="XL-VLMs arguments parser")

    # General
    parser.add_argument("--seed", type=int, default=0, help="Global seed.")

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Name of the model as defined in the transformers library",
    )
    parser.add_argument(
        "--local_files_only",
        default=False,
        action="store_true",
        help="Load HF models from local.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="llava",
        help="Task prompts to be applied to each input instruction.",
    )

    # Data
    parser.add_argument(
        "--data_dir", type=str, default="data/", help="Path to data directory."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="annotations.json",
        help="Path to the annotation file.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=-1,
        help="Number of examples to process. -1 means the full dataset.",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="coco", help="Dataset name."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for data loading"
    )

    # Hooks
    parser.add_argument(
        "--hook_names",
        nargs="+",
        help="List of hook names.",
        default=None,
    )
    parser.add_argument(
        "--modules_to_hook",
        type=parse_list_of_lists,
        default=None,
        help='A list of lists. Example format: "1,2,3;4,5,6". Contains the lists of modules to hook for each hook.',
    )
    parser.add_argument(
        "--exact_match_modules_to_hook",
        default=False,
        action="store_true",
        help="Exact match when searching for modules to hook.",
    )
    parser.add_argument(
        "--hook_postprocessing_name",
        type=str,
        default="save_hidden_states",
        help="Save hook output",
    )

    # Save hidden states
    parser.add_argument(
        "--token_idx", type=int, default=None, help="Save only the token at this index."
    )
    parser.add_argument(
        "--token_start_end_idx",
        nargs="+",
        help="[start_idx, end_idx] to select the beginning and end index of tokens to be saved.",
        default=None,
    )
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--save_filename", type=str, default="results")
    parser.add_argument(
        "--features_path",
        nargs="+",
        help="list of paths to features.",
        default=None,
    )
    # Feature decomposition
    parser.add_argument(
        "--analysis_name",
        type=str,
        help="Analysis method.",
        default="decompose_activations",
    )
    parser.add_argument(
        "--feature_key",
        type=str,
        help="Key for features to extract.",
        default="hidden_states",
    )
    parser.add_argument(
        "--module_to_decompose",
        type=str,
        help="Module name to prepare representation matrix for and decompose",
        default=None,
    )
    parser.add_argument(
        "--decomposition_extract_pos",
        type=int,
        default=None,
        help="Local token position for representation extraction for decomposition. None means representions will be averaged over all stored positions",
    )
    parser.add_argument(
        "--decomposition_method",
        type=str,
        help="Method for dictionary learning or decomposition (eg. pca, kmeans, snmf)",
        default=None,
    )
    parser.add_argument(
        "--num_concepts",
        type=int,
        default=None,
        help="Number of concepts for dictionary learning.",
    )
    parser.add_argument(
        "--dl_max_iter",
        type=int,
        default=5000,
        help="Max number of iterations for dictionary learning optimization.",
    )
    parser.add_argument(
        "--num_grounded_text_tokens",
        type=int,
        default=10,
        help="Number of concepts for dictionary learning.",
    )
    parser.add_argument(
        "--num_most_activating_samples",
        type=int,
        default=5,
        help="Number of most activated samples.",
    )

    # Generation mode
    parser.add_argument(
        "--generation_mode",
        default=False,
        action="store_true",
        help="True for autoregressive generation mode. False for teacher forcing.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50, help="Max numberof generated tokens."
    )

    # Saving hidden states for token of interest
    parser.add_argument(
        "--select_token_of_interest_samples",
        default=False,
        action="store_true",
        help="Load HF models from local.",
    )
    parser.add_argument(
        "--token_of_interest",
        type=str,
        default=None,
        help="The entity that you want to extract the representation of from hidden states.",
    )
    parser.add_argument(
        "--token_of_interest_idx",
        type=int,
        default=None,
        help="The tokenization of the entity that you want to extract the representation of from hidden states.",
    )
    parser.add_argument(
        "--token_of_interest_start_token",
        type=int,
        default=0,
        help="Start looking for token of interest from this index.",
    )

    return parser.parse_args()
