import json
import os
from typing import Any, Callable, Dict, List

import numpy as np
from torch.utils.data import Dataset

from datasets.constants import WORDS
from models.constants import TASK_PROMPTS

__all__ = ["COCODataset"]


class ImageTextDataset(Dataset):
    def __init__(
        self,
        annotation_file: str = "dataset_coco.json",
        data_dir: str = "/data/mshukor/data",
        split: str = "train",
        dataset_size: int = -1,
        seed: int = 0,
        dataset_name: str = "coco",
        mode: str = "train",
        questions_file: str = "",
        prompt_template: str = "llava",
        token_of_interest_num_samples: int = -1,
        **kwargs: Any,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.questions_file = questions_file
        self.dataset_size = dataset_size
        self.seed = seed
        self.dataset_name = dataset_name
        self.mode = mode
        self.prompt_template = prompt_template
        self.token_of_interest_num_samples = token_of_interest_num_samples

        self.rng = np.random.default_rng(seed)

        self.split = split

        self.create_dataset()

    def create_dataset(
        self,
    ) -> None:
        raise NotImplementedError(
            f"create_dataset() is not defined from dataset: {self.dataset_name}"
        )

    def apply_prompt(self, item: Dict[str, Any], mode: str = "train") -> str:
        if "train" in mode:
            text = f"{item['instruction']}{item['response']}"
        else:
            text = item["instruction"]
        return text

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        item = self.data[idx]
        item["text"] = self.apply_prompt(item, mode=self.mode)

        return item

    def __len__(
        self,
    ) -> None:
        return len(self.data)

    def get_difference_variations_of_token_of_interest(
        self, token_of_interest: str
    ) -> List[str]:

        tokens_of_interest = set(
            [
                token_of_interest,
                token_of_interest.capitalize(),
                token_of_interest.lower(),
            ]
        )
        return list(tokens_of_interest)

    def token_of_interest_presence(
        self,
        idx: int = 0,
        token_of_interests: List[str] = ["dog"],
        token_of_interest_key: str = "response",
    ) -> bool:
        """
        Checking a single sample's caption for presence of TOI (token of interest)
        """
        if isinstance(token_of_interests, str):
            token_of_interests = [token_of_interests]
        return any(
            [k in self.data[idx][token_of_interest_key] for k in token_of_interests]
        )

    def token_of_interest_idx_extractor(
        self,
        token_of_interest: str = "dog",
        token_of_interest_key: str = "response",
        allow_different_variations: bool = False,
        token_of_interest_class: str = None,
        logger: Callable = None,
    ) -> List[int]:
        """
        To extract the indices of samples where the TOI (token of interest) appears
        in their caption.
        """

        if token_of_interest_class is not None:
            token_of_interests = list(WORDS[token_of_interest_class])
        elif isinstance(token_of_interest, str):
            token_of_interests = [token_of_interest]
        else:
            token_of_interests = token_of_interest

        if allow_different_variations:
            for tok in token_of_interests:
                token_of_interests.extend(
                    self.get_difference_variations_of_token_of_interest(tok)
                )
            token_of_interests = list(set(token_of_interests))

        if logger is not None:
            logger.info(f"Start selecting samples containing {token_of_interests}")

        token_of_interest_indices = []
        num_samples = 0
        for idx in range(len(self)):
            if self.token_of_interest_presence(
                idx, token_of_interests, token_of_interest_key=token_of_interest_key
            ):
                token_of_interest_indices.append(idx)
                num_samples += 1
                if (
                    self.token_of_interest_num_samples > 0
                    and num_samples >= self.token_of_interest_num_samples
                ):
                    break
        if logger is not None:
            logger.info(
                f"Selecting {len(token_of_interest_indices)} indices for token_of_interests: {token_of_interests}"
            )
        return token_of_interest_indices

    def idx_in_ids(
        self,
        idx: int = 0,
        img_ids: List[str] = None,
    ) -> bool:

        return True if self.data[idx]["img_id"] in img_ids else False

    def indices_from_ids_extractor(self, img_ids: List[int] = None) -> List[int]:

        filtered_indices = []

        for idx in range(len(self)):
            if self.idx_in_ids(idx, img_ids):
                filtered_indices.append(idx)

        return filtered_indices


class COCODataset(ImageTextDataset):
    def create_dataset(
        self,
    ) -> None:
        annotation_path = os.path.join(self.data_dir, self.annotation_file)
        with open(annotation_path) as f:
            karpathy_data = json.load(f)

        data = []
        for datum in karpathy_data["images"]:
            split_ = datum["split"]
            if split_ != self.split:
                continue

            img_id = datum["filename"].split(".")[0]

            if "train" in img_id:
                source = "train2014"
            elif "val" in img_id:
                source = "val2014"
            else:
                raise NotImplementedError(
                    f"Please specify the image directory for the image: {img_id}"
                )

            image_path = os.path.join(self.data_dir, source, datum["filename"])
            instruction = TASK_PROMPTS.get(self.prompt_template, {}).get(
                "ShortCaptioning", "An image of "
            )
            targets = [d["raw"].strip() for d in datum["sentences"]]
            response = targets[0]  # take only the first caption

            item = {
                "img_id": img_id,
                "instruction": instruction,
                "response": response,
                "image": image_path,
                "targets": "$$".join(targets),
            }
            data.append(item)

        if self.dataset_size > 0:
            data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data


class VQAv2Dataset(ImageTextDataset):
    def create_dataset(
        self,
    ) -> None:

        annotations = json.load(
            open(os.path.join(self.data_dir, self.annotation_file))
        )["annotations"]
        questions = json.load(open(os.path.join(self.data_dir, self.questions_file)))[
            "questions"
        ]
        qid_2_question = {d["question_id"]: d["question"] for d in questions}

        assert self.split in [
            "val2014",
            "train2014",
        ], f"{self.split} split is not supported."

        data = []
        for datum in annotations:

            img_id = datum["image_id"]
            image_name = f"COCO_{self.split}_{img_id:012d}.jpg"
            image_path = os.path.join(self.data_dir, self.split, image_name)

            question_id = datum["question_id"]
            instruction = f"{qid_2_question[question_id].strip()}"
            instruction += TASK_PROMPTS.get(self.prompt_template, {}).get(
                "ShortVQA", " "
            )
            response = datum["multiple_choice_answer"].strip()
            item = {
                "img_id": img_id,
                "instruction": instruction,
                "response": response,
                "image": image_path,
                "answer_type": datum["answer_type"],
                "question_id": datum["question_id"],
                "targets": "$$".join([d["answer"] for d in datum["answers"]]),
            }
            data.append(item)

        if self.dataset_size > 0:
            data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data




class POPE_test_Dataset(ImageTextDataset):
    def create_dataset(
        self,
    ) -> None:
        

        annotations = json.load(
            open(os.path.join(self.data_dir, self.annotation_file))
        )
        splits = self.split.split(",")


        for split in splits:
            assert split in [
                "adversarial",
                "popular",
                "random",
            ], f"{self.split} split is not supported."


        assert len(splits)==1
        split = splits[0]     

        data = []
        for datum in annotations:

            image_name = datum["filename"]

            source = "images"
            
            image_path = os.path.join(self.data_dir, source, image_name)

            instruction = datum["instruction"].split("Answer with just one word.")[0].strip()
            response = datum["response"].strip()
            sample_subset = datum["subset"].strip()


            if response=="yes" or (response=="no" and sample_subset==split):


                item = {
                    "instruction": instruction,
                    "response": response,
                    "image": image_path,
                    "targets": "$$".join([datum["response"]]),
                }
                data.append(item)

        if self.dataset_size > 0:
            data = data[:self.dataset_size]

            # data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data





class POPE_train_Dataset(ImageTextDataset):
    def create_dataset(
        self,
    ) -> None:
        

        annotations = json.load(
            open(os.path.join(self.data_dir, self.annotation_file))
        )

        data = []

        dataset_size = 0
        if self.split=="all":
            if self.dataset_size > 0:
                dataset_size = self.dataset_size
                positive_answers = dataset_size//2
                remaining = dataset_size - positive_answers
                adversarial_num_samples_negative = remaining//3
                popular_num_samples_negative = remaining//3
                random_num_samples_negative = remaining - (adversarial_num_samples_negative+popular_num_samples_negative)



            positive_samples_picked = 0
            negative_adversarial_samples_picked = 0
            negative_popular_samples_picked = 0
            negative_random_samples_picked = 0


            for datum in annotations:

                image_name = datum["filename"]

                source = "images"
                
                image_path = os.path.join(self.data_dir, source, image_name)

                instruction = datum["instruction"].split("Answer with just one word.")[0].strip()
                response = datum["response"].strip()
                sample_subset = datum["subset"].strip()

                item = {
                    "instruction": instruction,
                    "response": response,
                    "image": image_path,
                    "targets": "$$".join([datum["response"]]),
                }

                if dataset_size>0:
                    if response=="yes" and positive_samples_picked<positive_answers:
                    
                        data.append(item)
                        positive_samples_picked += 1

                    elif sample_subset=="adversarial" and negative_adversarial_samples_picked<adversarial_num_samples_negative:
                        data.append(item)
                        negative_adversarial_samples_picked += 1
                    
                    elif sample_subset=="popular" and negative_popular_samples_picked<popular_num_samples_negative:
                        data.append(item)
                        negative_popular_samples_picked += 1

                    elif sample_subset=="random" and negative_random_samples_picked<random_num_samples_negative:
                        data.append(item)
                        negative_random_samples_picked += 1
                else:
                    data.append(item)


        elif self.split in ["adversarial", "popular", "random"]:

            split = self.split
            if self.dataset_size > 0:
                dataset_size = self.dataset_size
                positive_answers = dataset_size//2
                negative_answers = dataset_size - positive_answers
                

            positive_samples_picked = 0
            negative_samples_picked = 0

            for datum in annotations:

                image_name = datum["filename"]

                source = "images"
                
                image_path = os.path.join(self.data_dir, source, image_name)

                instruction = datum["instruction"].split("Answer with just one word.")[0].strip()
                response = datum["response"].strip()
                sample_subset = datum["subset"].strip()

                item = {
                    "instruction": instruction,
                    "response": response,
                    "image": image_path,
                    "targets": "$$".join([datum["response"]]),
                }


                if dataset_size>0:
                    if response=="yes" and positive_samples_picked<positive_answers:
                        data.append(item)
                        positive_samples_picked += 1

                    elif sample_subset==split and negative_samples_picked<negative_answers:
                        data.append(item)
                        negative_samples_picked += 1
     
                else:
                    data.append(item)

        else:
            NotImplementedError


        self.data = data



