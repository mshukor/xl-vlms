import json
from typing import Any, Dict


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




answer_type_to_answer="/data/mshukor/data/coco/type_to_answer_dict.json"
type_to_answer_dict = json.load(open(answer_type_to_answer))
answer_to_answer_type = get_word_to_type_dict(
    type_to_answer_dict["answer_type"]
)
answer_to_question_type = get_word_to_type_dict(
    type_to_answer_dict["question_type"]
)


print(answer_to_answer_type['yes'])
print(len(answer_to_answer_type))
