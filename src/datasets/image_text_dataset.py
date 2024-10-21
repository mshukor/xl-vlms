import json
import os
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset

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
        prompt_template: str = "llava",
        **kwargs: Any,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.dataset_size = dataset_size
        self.seed = seed
        self.dataset_name = dataset_name
        self.mode = mode
        self.prompt_template = prompt_template

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

    def token_of_interest_presence(
        self,
        idx: int = 0,
        tokeof_interest: str = "dog",
    ) -> bool:
        """
        Checking a single sample's caption for presence of TOI (token of interest)
        """

        return True if tokeof_interest in self.data[idx]["response"] else False

    def token_of_interest_idx_extractor(
        self, tokeof_interest: str = "dog"
    ) -> List[int]:
        """
        To extract the indices of samples where the TOI (token of interest) appears
        in their caption.
        """

        token_of_interest_indices = []

        for idx in range(len(self)):
            if self.token_of_interest_presence(idx, tokeof_interest):
                token_of_interest_indices.append(idx)

        return token_of_interest_indices


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
            }
            data.append(item)

        if self.dataset_size > 0:
            data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data
