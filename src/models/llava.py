from typing import Any, Callable, Dict, List

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .image_text_model import ImageTextModel

__all__ = ["LLaVA"]


class LLaVA(ImageTextModel):

    def set_model(
        self,
    ) -> None:

        self.model_ = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
        )

    def get_language_model(
        self,
    ) -> Callable:

        return self.model_.language_model

    def get_lm_head(
        self,
    ) -> Callable:

        return self.model_.language_model.lm_head

    def set_processor(
        self,
    ) -> None:

        self.processor_ = AutoProcessor.from_pretrained(
            self.processor_name, local_files_only=self.local_files_only
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(
        self,
    ) -> None:

        self.preprocessor_ = self.preprocess_input

    def get_conversation_round(
        self, instruction: str = "What are these?", response: str = ""
    ) -> List[Dict[str, Any]]:

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image"},
                ],
            },
        ]
        if response:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response},
                    ],
                },
            )

        return conversation

    def preprocess_text(
        self,
        instruction: str = "What are these?",
        response: str = "",
        generation_mode: bool = False,
    ) -> str:

        conversation = self.get_conversation_round(
            instruction=instruction, response=response
        )
        prompt = self.processor_.apply_chat_template(
            conversation, add_generation_prompt=generation_mode
        )

        return prompt
