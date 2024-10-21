from typing import Any, Callable, Dict, List

import torch
from nltk.corpus import words

GIST_FILE_PATH = "src/assets/gist_stopwords.txt"


__all__ = [
    "get_multimodal_grounding",
    "concept_text_grounding",
    "concept_image_grounding",
    "visualize_multimodal_grounding",
]


def get_stopwords(gist_file_path: str = "gist_stopwords.txt") -> List[str]:
    gist_file = open(gist_file_path, "r")
    content = gist_file.read()
    stopwords = content.split(",")
    gist_file.close()
    return stopwords


def valid_word(word, eng_corpus: List[str], stopwords: List[str]) -> bool:
    word = word.lower().strip()
    return word in eng_corpus and len(word) > 2 and word not in stopwords


@torch.no_grad()
def concept_text_grounding(
    concepts: torch.Tensor,
    lm_head: Callable = None,
    tokenizer: Callable = None,
    num_top_tokens: int = 15,
) -> List[List[str]]:
    # components are of shape n_comp x feature_dim
    eng_corpus = words.words()
    stopwords = get_stopwords(gist_file_path=GIST_FILE_PATH)
    num_concepts = concepts.shape[0]

    token_logits = lm_head(concepts.float())
    top_token_idx = token_logits.argsort(dim=-1, descending=True)[:, :num_top_tokens]
    grounded_words_list = []
    for k in range(num_concepts):
        comp_words = tokenizer.batch_decode(
            top_token_idx[k, :num_top_tokens], skip_special_tokens=True
        )
        comp_words = [
            word.lower().strip()
            for word in comp_words
            if valid_word(word, eng_corpus=eng_corpus, stopwords=stopwords)
        ]
        grounded_words_list.append(comp_words)
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
    save_mas_dir: str = "",
    num_grounded_text_tokens: int = 10,
    num_most_activating_samples: int = 5,
    metadata: Dict[str, Any] = {},
    logger: Callable = None,
) -> None:

    lm_head = model_class.get_lm_head().float()
    tokenizer = model_class.get_tokenizer()

    activations = torch.Tensor(activations)
    concepts = torch.Tensor(concepts)

    grounded_words = []
    grounding_dict = {}
    if "lm_head" in module_to_decompose:
        top_tokens = concepts.argmax(axis=1)
        top_words = tokenizer.decode(top_tokens)
        logger.info(f"Lm head top_tokens: {top_tokens}, top words: {top_words}")
    elif text_grounding:
        grounded_words = concept_text_grounding(
            concepts,
            lm_head=lm_head,
            tokenizer=tokenizer,
            num_top_tokens=num_grounded_text_tokens,
        )
        for i in range(len(grounded_words)):
            logger.info(f"Concept {i} grounded words: {grounded_words[i]}")
        grounding_dict["text_grounding"] = grounded_words

    if image_grounding:
        image_indices = concept_image_grounding(
            activations=activations,
            num_images_per_concept=num_most_activating_samples,
        )
        image_paths = metadata.get("image_paths", [])
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
            logger.info(f"Concept {i} image paths: {concept_image_paths}")

        grounding_dict["image_grounding_paths"] = all_concept_image_paths

    return grounding_dict
