# Adapted from https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/_task_utils/vqa_eval_metric.py#L4
import json
import re
import statistics
from typing import Any, Callable, Dict, List

import torch

from metrics.utils import get_number_predictions_with_token_of_interest

__all__ = ["compute_vqav2_accuracy", "EvalAIAnswerProcessor"]


class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


def vqav2_process_results(result, answers):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert isinstance(result, str), f"The result should be text, but got {result}."
    resAns = eval_ai_processor(result)
    accuracy = 0
    if answers is not None:
        gtAcc = []
        gtAnswers = [
            ans.replace("\n", " ").replace("\t", " ").strip() for ans in answers
        ]
        gtAnswers = [eval_ai_processor.process_punctuation(ans) for ans in answers]
        gtAnswers = [eval_ai_processor.process_digit_article(ans) for ans in answers]

        resAns = eval_ai_processor.process_punctuation(resAns)
        resAns = eval_ai_processor.process_digit_article(resAns)

        matchingAns = [item for item in gtAnswers if item == resAns]
        acc = min(1, float(len(matchingAns)) / 3)
        accuracy = acc
    return accuracy * 100.0, resAns


def update_count_dict(key: str, dictionary: Dict[str, Any]):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1
    return dictionary


def get_word_to_type_dict(type_to_answer_dict: Dict[str, Any]):

    words_to_type_count = {}
    for type_, words in type_to_answer_dict.items():
        for word in words:
            if word not in words_to_type_count:
                words_to_type_count[word] = {}
            words_to_type_count[word] = update_count_dict(
                type_, words_to_type_count[word]
            )
    answer_to_answer_type = {}
    for word in words_to_type_count:
        max_type = max(words_to_type_count[word], key=words_to_type_count[word].get)
        answer_to_answer_type[word] = max_type
    return answer_to_answer_type


def compute_vqav2_accuracy(
    data: Dict[str, Any],
    token_of_interest: str = None,
    category_of_interest: str = "",
    logger: Callable = None,
    answer_type_to_answer: str = "",
    preds_token_of_interests: List[str] = None,
    targets_token_of_interests: List[str] = None,
    save_filename: str = None,
    save_predictions: bool = False,
    predictions_path: str = None,
    **kwargs: Any,
):
    all_results = {}
    accuracy = {}
    token_of_interest_accuracy = {}
    token_of_interest_acc = 0
    all_answers = {}
    answer_types_count = {}
    question_types_count = {}
    num_samples = len(data["model_output"])

    for idx in range(len(data["targets"])):
        for i in range(len(data["targets"][idx])):
            data["targets"][idx][i] = data["targets"][idx][i].split("$$")

    answer_to_answer_type = {}
    answer_to_question_type = {}
    if answer_type_to_answer:
        type_to_answer_dict = json.load(open(answer_type_to_answer))
        answer_to_answer_type = get_word_to_type_dict(
            type_to_answer_dict["answer_type"]
        )
        answer_to_question_type = get_word_to_type_dict(
            type_to_answer_dict["question_type"]
        )

    for idx in range(num_samples):
        for i, answer in enumerate(data["model_predictions"][idx]):
            answers = data["targets"][idx][i]
            answer_type = data["answer_type"][idx][i]
            acc, ans = vqav2_process_results(answer, answers)
            if token_of_interest is not None:
                token_of_interest_acc, _ = vqav2_process_results(
                    answer, [token_of_interest]
                )
            if answer_type in accuracy:
                accuracy[answer_type].append(acc)
                all_answers[answer_type].append(answer)
                token_of_interest_accuracy[answer_type].append(token_of_interest_acc)
            else:
                accuracy[answer_type] = [acc]
                all_answers[answer_type] = [answer]
                token_of_interest_accuracy[answer_type] = [token_of_interest_acc]
            if ans in answer_to_answer_type:
                pred_answer_type = answer_to_answer_type[ans]
            else:
                try:
                    float(ans)
                    pred_answer_type = "number"
                except:
                    pred_answer_type = "unrecognized"
            # pred_answer_type = answer_to_answer_type.get(ans, "unrecognized")
            answer_types_count = update_count_dict(pred_answer_type, answer_types_count)
            pred_question_type = answer_to_question_type.get(ans, "unrecognized")
            question_types_count = update_count_dict(
                pred_question_type, question_types_count
            )
    global_accuracy = {k: statistics.mean(v) for k, v in accuracy.items()}
    global_accuracy["overall"] = statistics.mean(list(global_accuracy.values()))
    overall_without_category_of_interest = [
        v
        for k, v in global_accuracy.items()
        if category_of_interest not in k and "overall" not in k
    ]
    global_accuracy["overall_without_category_of_interest"] = (
        statistics.mean(overall_without_category_of_interest)
        if overall_without_category_of_interest
        else 0
    )

    answer_counts = {k: {} for k in all_answers}
    answer_counts["overall"] = {}
    answer_counts["overall_without_category_of_interest"] = {}
    for answer_type in all_answers:
        for answer in all_answers[answer_type]:
            if answer in answer_counts[answer_type]:
                answer_counts[answer_type][answer] += 1
            else:
                answer_counts[answer_type][answer] = 1
        answer_counts["overall"].update(answer_counts[answer_type])
        if category_of_interest not in answer_type:
            answer_counts["overall_without_category_of_interest"].update(
                answer_counts[answer_type]
            )

    token_of_interest_accuracy = {
        k: statistics.mean(v) for k, v in token_of_interest_accuracy.items()
    }
    token_of_interest_accuracy["overall"] = statistics.mean(
        list(token_of_interest_accuracy.values())
    )
    overall_without_category_of_interest = [
        v
        for k, v in token_of_interest_accuracy.items()
        if category_of_interest not in k and "overall" not in k
    ]
    token_of_interest_accuracy["overall_without_category_of_interest"] = (
        statistics.mean(overall_without_category_of_interest)
        if overall_without_category_of_interest
        else 0
    )

    (
        num_preds_with_toi,
        num_preds_and_targets_with_toi,
        num_preds_and_baseline_preds_with_toi,
        num_preds_changed,
    ) = (0, 0, 0, 0)
    if preds_token_of_interests is not None and targets_token_of_interests is not None:
        (
            num_preds_with_toi,
            num_preds_and_targets_with_toi,
            num_preds_and_baseline_preds_with_toi,
            num_preds_changed,
        ) = get_number_predictions_with_token_of_interest(
            data["model_predictions"],
            data["targets"],
            ids=data["question_id"],
            preds_token_of_interests=preds_token_of_interests,
            targets_token_of_interests=targets_token_of_interests,
            predictions_path=predictions_path,
        )
    all_results["num_preds_with_toi"] = num_preds_with_toi
    all_results["num_preds_and_targets_with_toi"] = num_preds_and_targets_with_toi
    all_results["num_preds_and_baseline_preds_with_toi"] = (
        num_preds_and_baseline_preds_with_toi
    )
    all_results["num_preds_changed"] = num_preds_changed

    if logger:
        for k, v in answer_counts.items():
            logger.info(f"Answers stats {k}: \n{v}")
        logger.info(
            f"Predicted answer types:\n{json.dumps(answer_types_count, indent=4)}\n"
        )
        logger.info(
            f"Predicted question types:\n{json.dumps(question_types_count, indent=4)}\n"
        )
        logger.info(f"Accuracy:\n{json.dumps(global_accuracy, indent=4)}\n")
        logger.info(
            f"Token of interest accuracy: \n{json.dumps(token_of_interest_accuracy, indent=4)}"
        )
        logger.info(
            f"num_preds_with_toi {preds_token_of_interests}: {num_preds_with_toi}"
        )
        logger.info(
            f"num_preds_and_targets_with_toi {targets_token_of_interests}: {num_preds_and_targets_with_toi}"
        )
        logger.info(
            f"num_preds_and_baseline_preds_with_toi {targets_token_of_interests}: {num_preds_and_baseline_preds_with_toi}"
        )
        logger.info(f"num_preds_changed: {num_preds_changed}")
    all_results["answer_counts"] = answer_counts
    all_results["answer_types_count"] = answer_types_count
    all_results["question_types_count"] = question_types_count
    all_results["global_accuracy"] = global_accuracy
    all_results["token_of_interest_accuracy"] = token_of_interest_accuracy

    if save_filename:
        with open(save_filename, "w") as json_file:
            json.dump(all_results, json_file, indent=4)
        if logger is not None:
            logger.info(f"Saving data to: {save_filename}")
        if save_predictions:
            id_to_answer = {}
            for i in range(len(data["model_predictions"])):
                preds = data["model_predictions"][i]
                ids = data["question_id"][i]
                for id, pred in zip(ids, preds):
                    if isinstance(id, torch.Tensor):
                        id = id.item()
                    id_to_answer[id] = pred
            save_filename = save_filename.split(".json")[0] + "_model_prediction.json"
            with open(save_filename, "w") as json_file:
                json.dump(id_to_answer, json_file, indent=4)
            if logger is not None:
                logger.info(
                    f"Saving {len(id_to_answer)} predictions to: {save_filename}"
                )
    return all_results
