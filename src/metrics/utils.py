import json
from typing import Dict, List, Tuple, Union

import torch
from nltk.corpus import words

# from helpers.utils import (GIST_FILE_PATH, get_stopwords,
#                                            valid_word)

__all__ = ["get_words_frequency", "get_number_predictions_with_token_of_interest"]


# multimodal grounding utils

GIST_FILE_PATH = "src/assets/gist_stopwords.txt"


def is_number(string: str) -> bool:
    try:
        float(string)  # Attempt to convert to float
        return True
    except ValueError:
        return False


def get_stopwords(gist_file_path: str = "gist_stopwords.txt") -> List[str]:
    gist_file = open(gist_file_path, "r")
    content = gist_file.read()
    stopwords = content.split(",")
    gist_file.close()
    return stopwords


def valid_word(word, eng_corpus: List[str], stopwords: List[str]) -> bool:
    word = word.lower().strip()
    return (
        word in eng_corpus and len(word) > 2 and word not in stopwords
    ) or is_number(word)


def get_words_frequency(
    predictions: List[List[str]], gist_file_path: str = GIST_FILE_PATH
) -> Dict[str, int]:

    eng_corpus = words.words()
    stopwords = get_stopwords(gist_file_path=gist_file_path)

    words_count = {}
    for preds in predictions:
        for pred in preds:
            words_ = pred.split(" ")
            for w in words_:
                w = w.lower().strip()
                if valid_word(w, eng_corpus=eng_corpus, stopwords=stopwords):
                    if w in words_count:
                        words_count[w] += 1
                    else:
                        words_count[w] = 1
    return words_count


def get_number_predictions_with_token_of_interest(
    predictions: List[List[str]],
    targets: List[List[str]],
    ids: List[List[Union[str, int, torch.tensor]]] = [],
    preds_token_of_interests: List[str] = [],
    targets_token_of_interests: List[str] = [],
    predictions_path: str = None,
) -> Tuple[int]:
    num_preds_with_toi = 0
    num_preds_and_targets_with_toi = 0
    num_preds_and_baseline_preds_with_toi = 0
    num_preds_changed = 0
    baseline_predictions = {}
    if predictions_path is not None:
        baseline_predictions = json.load(open(predictions_path))
    targets_token_of_interests = [w.lower().strip() for w in targets_token_of_interests]
    for idx, (preds, gts) in enumerate(zip(predictions, targets)):
        for i in range(len(preds)):
            pred = preds[i]
            gt = gts[i]
            if ids:
                sample_id = ids[idx][i]
                if isinstance(sample_id, torch.Tensor):
                    sample_id = sample_id.item()
            else:
                sample_id = -1
            baseline_pred = baseline_predictions.get(str(sample_id), "").lower().strip()
            pred = pred.lower().strip()
            gt = [g.lower().strip() for g in gt]
            if any([k in pred for k in preds_token_of_interests]):
                num_preds_with_toi += 1
                for gt_ in gt:
                    if any([k in gt_ for k in targets_token_of_interests]):
                        num_preds_and_targets_with_toi += 1
                        break
                if any([k in baseline_pred for k in targets_token_of_interests]):
                    num_preds_and_baseline_preds_with_toi += 1
            if pred != baseline_pred:
                num_preds_changed += 1

    return (
        num_preds_with_toi,
        num_preds_and_targets_with_toi,
        num_preds_and_baseline_preds_with_toi,
        num_preds_changed,
    )
