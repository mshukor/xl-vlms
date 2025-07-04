from typing import Any, Callable, Dict

import requests
from PIL import Image

__all__ = ["ImageTextModel"]


class ImageTextModel:
    def __init__(
        self,
        model_name_or_path: str = "llava-hf/llava-1.5-7b-hf",
        processor_name: str = "llava-hf/llava-1.5-7b-hf",
        local_files_only: bool = False,
        **kwargs: Any,
    ):

        self.model_name_or_path = model_name_or_path

        self.processor_name = processor_name
        self.local_files_only = local_files_only

        if processor_name is None:
            self.processor_name = model_name_or_path

        self.set_model()
        self.set_processor()
        self.set_preprocessor()

    def set_model(
        self,
    ) -> None:
        raise NotImplementedError(
            f"set_model() is not defined for the model: {self.model_name_or_path}"
        )

    def set_processor(
        self,
    ) -> None:
        raise NotImplementedError(
            f"set_processor() is not defined for the model: {self.processor_name}"
        )

    def set_preprocessor(
        self,
    ) -> None:
        raise NotImplementedError(
            f"set_preprocessor() is not defined for the model: {self.processor_name}"
        )

    def preprocess_text(
        self,
        instruction: str = "What are these?",
        response: str = "",
        generation_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError(
            f"preprocess_text() is not defined for the model: {self.model_name_or_path}"
        )

    def preprocess_images(
        self,
        image_file: str,
        **kwargs: Any,
    ):
        if "http" in image_file:
            image = Image.open(requests.get(image_file, stream=True).raw)
        else:
            image = Image.open(image_file)
        return image

    def preprocess_input(
        self,
        instruction: str = "What are these?",
        image_file: str = None,
        response: str = "",
        generation_mode: bool = False,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        image = self.preprocess_images(image_file)
        text = self.preprocess_text(
            instruction=instruction,
            response=response,
            generation_mode=generation_mode,
        )

        inputs = self.processor_(images=image, text=text, return_tensors="pt")

        return inputs

    def preprocessor(
        self,
        instruction: str = "What are these?",
        image_file: str = "",
        response: str = "",
        generation_mode: bool = False,
        continue_final_message: bool = False,
        **kwargs: Any,
    ):
        preprocessor = self.get_preprocessor()
        inputs = (
            preprocessor(
                instruction=instruction,
                image_file=image_file,
                response=response,
                generation_mode=generation_mode,
                continue_final_message=continue_final_message,
            )
            .to(self.get_model().device)
            .to(self.get_model().dtype)
        )

        return inputs

    def get_model(
        self,
    ) -> Callable:

        return self.model_

    def get_language_model(
        self,
    ) -> Callable:

        return self.model_.language_model

    def get_lm_head(
        self,
    ) -> Callable:

        return self.model_.language_model.lm_head

    def get_processor(
        self,
    ) -> Callable:

        return self.processor_

    def get_preprocessor(
        self,
    ) -> Callable:

        return self.preprocessor_

    def get_tokenizer(
        self,
    ) -> Callable:

        return self.tokenizer_
