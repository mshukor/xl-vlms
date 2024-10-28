from typing import Dict, Any, Callable
import argparse
import clip
import numpy as np
import torch
from nltk.corpus import words

from analysis.multimodal_grounding import get_stopwords, valid_word
from metrics.clipscore import extract_image_features, img_clipscore
from analysis.feature_decomposition import project_test_samples



def get_clip_score(
    features: Dict[str, torch.Tensor] = None,
    metadata: Dict[str, Any] = {},
    concepts_dict: Dict[str, Any] = {},
    model_class: Callable = None,
    device: torch.device = torch.device("cpu"),
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Dict[str, Any]:

    features = list(features.values())[0]
    metadata = list(metadata.values())[0]
    analysis_model = concepts_dict["analysis_model"]
    grounding_words = concepts_dict["text_grounding"]
    projections = project_test_samples(
        sample=features,
        analysis_model=analysis_model,
        decomposition_type=concepts_dict["decomposition_method"],
    )
    if args.use_random_words:
        lm_head = model_class.get_lm_head().float()
        tokenizer = model_class.get_tokenizer()
        grounding_words = get_random_words(
            lm_head=lm_head,
            tokenizer=tokenizer,
            grounding_words=grounding_words,
        )
        logger.info(f"Random words usage is True. Only for CLIPScore evaluation")

    clipscore_dict = compute_test_clipscore(
        projections=projections,
        grounding_words=grounding_words,
        device=device,
        metadata=metadata,
    )
    logger.info(
        f"top-1 test CLIPScore (mean, std) {clipscore_dict['top_1_mean']: .3f} +/- {clipscore_dict['top_1_std']: .3f}"
    )

    return clipscore_dict


def get_random_words(lm_head, tokenizer, grounding_words):
    """
    This function replaces grounding words of each concept by a set of random words, possibly of same length
    Random words obtained by:
    (i) Sampling a random direction to decode with lm_head
    (ii) Decode top tokens which satisfy same valid word filters as grounding words
    """
    eng_corpus = words.words()
    stopwords = get_stopwords()
    all_random_words = []
    for k, concept_words in enumerate(grounding_words):
        # k is concept idx, words is grounded words for concept k
        desired_length = len(concept_words)
        num_top_tokens = min(
            10 * desired_length, lm_head.out_features
        )  # Should be more than enough
        random_direction = torch.rand(1, lm_head.in_features).float()
        token_logits = lm_head(random_direction)
        top_token_idx = token_logits.argsort(dim=-1, descending=True)[
            :, :num_top_tokens
        ]
        candidate_words = tokenizer.batch_decode(
            top_token_idx[0], skip_special_tokens=True
        )
        candidate_words = [
            word.lower().strip()
            for word in candidate_words
            if valid_word(word, eng_corpus=eng_corpus, stopwords=stopwords)
        ]
        if len(candidate_words) > desired_length:
            candidate_words = candidate_words[:desired_length]
        all_random_words.append(candidate_words)
    return all_random_words


def compute_overlap(grounding_words, logger: Callable = None) -> Dict[str, Any] :
    """
    Function to compute overlap metric given the grounded words of a concept dictionary
    Input: List of grounded words for concepts: List[List]
    """
    num_concepts = len(grounding_words)
    overlap_matrix = np.zeros([num_concepts, num_concepts])
    for i in range(num_concepts):
        words_i = grounding_words[i]
        if len(words_i) == 0:
            continue
        for j in range(num_concepts):
            words_j = grounding_words[j]
            overlap_ij = len([w for w in words_i if w in words_j])
            overlap_matrix[i, j] = overlap_ij * 1.0 / len(words_i)

    overlap_metric = overlap_matrix.sum() - np.diag(overlap_matrix).sum()
    overlap_metric = overlap_metric / (num_concepts * (num_concepts - 1))

    if logger is not None:
        logger.info(f"Overlap metric (lower is better): {overlap_metric: .3f}")
    
    scores = {}
    scores["overlap_metric"] = overlap_metric
    scores["overlap_matrix"] = overlap_matrix
    return scores


def compute_test_clipscore(projections, grounding_words, metadata, device):
    scores = []
    scores_gt = []
    image_paths = []
    target_captions = []
    num_samples = projections.shape[0]
    clip_model, transform = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    # Format of grounding words in original evaluation was reversed
    grounding_words = [words[::-1] for words in grounding_words]

    image_paths = metadata.get("image_paths", [])
    if "token_of_interest_mask" in metadata.keys():
        token_of_interest_mask = metadata.get("token_of_interest_mask", None)
        image_paths = [
            image_paths[i][0]
            for i in range(len(image_paths))
            if token_of_interest_mask[i]
        ]

    image_features = extract_image_features(
        image_paths, clip_model, device, batch_size=8
    )  # image_features of shape (num_images, 512)

    for idx in range(num_samples):
        img_activations = projections[idx]
        img_feat = image_features[idx]
        img_score = img_clipscore(
            clip_model, img_feat, img_activations, grounding_words, device, top_k=5
        )
        scores.append(img_score)
    scores = np.array(scores)

    # Return dictionary containing all test sample scores, their mean, std
    scores_dict = {}
    for k in [1, 3]:
        key = f"top_{k}_all"
        key_mean = f"top_{k}_mean"
        key_std = f"top_{k}_std"
        all_test_scores = scores[:, -k:].mean(axis=1)
        mean_topk_score, std_topk = all_test_scores.mean(), all_test_scores.std()
        scores_dict[key] = all_test_scores
        scores_dict[key_mean] = mean_topk_score
        scores_dict[key_std] = std_topk

    return scores_dict
