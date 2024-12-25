import argparse
import os
from typing import Any, Callable, Dict, List

import torch
from nltk.corpus import words

from helpers.utils import save_analysis_to_file
from metrics.utils import GIST_FILE_PATH, get_stopwords, valid_word

__all__ = [
    "get_multimodal_grounding",
    "concept_text_grounding",
    "concept_image_grounding",
    "visualize_multimodal_grounding",
]

@torch.no_grad()
def concept_text_grounding(
    concepts: torch.Tensor,
    lm_head: Callable = None,
    tokenizer: Callable = None,
    num_top_tokens: int = 15,
    gist_file_path: str = GIST_FILE_PATH,
    pre_num_top_tokens: int = 50,
    keep_unique_words: bool = False,
) -> List[List[str]]:
    # components are of shape n_comp x feature_dim
    eng_corpus = words.words()
    stopwords = get_stopwords(gist_file_path=gist_file_path)
    num_concepts = concepts.shape[0]

    token_logits = lm_head(concepts.float())
    assert (
        pre_num_top_tokens > num_top_tokens
    ), f"pre_num_top_tokens {pre_num_top_tokens} <= num_top_tokens {num_top_tokens}"
    top_token_idx = token_logits.argsort(dim=-1, descending=True)[
        :, :pre_num_top_tokens
    ]
    grounded_words_list = []
    for k in range(num_concepts):
        comp_words = tokenizer.batch_decode(top_token_idx[k], skip_special_tokens=True)
        comp_words = [
            word.lower().strip()
            for word in comp_words
            if valid_word(word, eng_corpus=eng_corpus, stopwords=stopwords)
        ]
        if keep_unique_words:
            comp_words = list(set(comp_words))
        grounded_words_list.append(comp_words[:num_top_tokens])
    return grounded_words_list


def concept_image_grounding(
    activations: torch.Tensor,
    num_images_per_concept: int = 5,
) -> torch.Tensor:

    local_image_indices = torch.argsort(activations, dim=0, descending=True)[
        :num_images_per_concept
    ].T  # After transpose shape (num_concepts, num_images_per_concept)

    return local_image_indices


def get_multimodal_grounding(
    concepts: torch.Tensor,
    activations: torch.Tensor,
    model_class: Callable,
    text_grounding: bool = True,
    image_grounding: bool = True,
    module_to_decompose: str = "",
    save_dir: str = "",
    save_name: str = "",
    num_grounded_text_tokens: int = 10,
    num_most_activating_samples: int = 5,
    metadata: Dict[str, Any] = {},
    save_analysis: bool = False,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> None:
    lm_head = model_class.get_lm_head().float()
    tokenizer = model_class.get_tokenizer()
    grounding_dict = {}

    activations = torch.Tensor(activations)
    concepts = torch.Tensor(concepts)
    grounding_dict["concepts"] = concepts
    grounding_dict["activations"] = activations
    grounding_dict["decomposition_method"] = args.decomposition_method

    grounded_words = []

    if "lm_head" in module_to_decompose:
        top_tokens = concepts.argmax(axis=1)
        top_words = tokenizer.decode(top_tokens)
        if logger is not None:
            logger.info(f"Lm head top_tokens: {top_tokens}, top words: {top_words}")
            logger.info("Lm head only for analysis. Function returning")
        return

    if text_grounding:
        grounded_words = concept_text_grounding(
            concepts,
            lm_head=lm_head,
            tokenizer=tokenizer,
            num_top_tokens=num_grounded_text_tokens,
            pre_num_top_tokens=args.pre_num_top_tokens,
        )
        if logger is not None:
            for i in range(len(grounded_words)):
                logger.info(f"Concept {i} grounded words: {grounded_words[i]}")
        grounding_dict["text_grounding"] = grounded_words

    if image_grounding:
        logger.info("Activations size: ", activations.shape)

        image_indices = concept_image_grounding(
            activations=activations,
            num_images_per_concept=num_most_activating_samples,
        )
        image_paths = metadata.get("image", [])
        logger.info("Image paths length: ", len(image_paths))
        # Only keep image paths for samples with token_of_interest_mask True

        token_of_interest_mask = metadata.get("token_of_interest_mask", None)
        if token_of_interest_mask is not None:
            image_paths = [
                image_paths[i]
                for i in range(len(image_paths))
                if token_of_interest_mask[i]
            ]

        all_concept_image_paths = []
        for i, concept_indices in enumerate(image_indices):
            concept_image_paths = [
                image_paths[concept_indices[k]][0] for k in range(len(concept_indices))
            ]
            all_concept_image_paths.append(concept_image_paths)
            if logger is not None:
                logger.info(f"Concept {i} image paths: {concept_image_paths}")

        grounding_dict["image_grounding_paths"] = all_concept_image_paths

    grounding_dict["concepts"] = concepts
    grounding_dict["activations"] = activations

    if save_analysis:

        assert (
            save_name is not None
        ), "save_name should not be None when save_analysis is set to True."
        analysis_saving_name = (
            f"decompose_activations_{args.decomposition_method}_{save_name}"
        )
        analysis_saving_path = os.path.join(
            save_dir, "analysis", f"{analysis_saving_name}"
        )

        if logger is not None:
            logger.info(
                f"Saving decomposition results dictionary to: {analysis_saving_path}"
            )

        save_analysis_to_file(
            grounding_dict, analysis_saving_path, grounding_dict.keys(), logger=logger
        )

    return grounding_dict
