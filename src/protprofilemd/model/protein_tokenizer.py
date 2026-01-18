import re
import logging
from abc import ABC, abstractmethod
from typing import List, Union

from transformers import T5Tokenizer, AutoTokenizer
import torch


class ProteinTokenizer(T5Tokenizer, ABC):
    @abstractmethod
    def _preprocess_text(self, text: str) -> str:
        pass

    @abstractmethod
    def _postprocess_text(self, text: str) -> str:
        pass

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = None,
        do_lower_case=False,
        use_fast=True,
        legacy=False,
        **kwargs,
    ) -> T5Tokenizer:
        """
        Load pretrained tokenizer, suppressing the warning about class name mismatch.
        This warning is safe to ignore since ProteinTokenizer subclasses only add
        preprocessing/postprocessing and don't change the core tokenization logic.

        Example:
        ```python
        tokenizer = ProstT5Tokenizer.from_pretrained()
        # or
        tokenizer = ProstT5Tokenizer.from_pretrained("Rostlab/ProstT5")

        tokenizer.protein_encode(["A", "asdfasffad", "asdfasdfasf".upper()])
        tokenizer.protein_decode([[3, 1], [1], [3, 7, 10, 15, 3, 7, 10, 15, 3, 7, 15, 1]])
        ```
        """
        if pretrained_model_name_or_path is None:
            if cls.__name__ == "ProstT5Tokenizer":
                pretrained_model_name_or_path = "Rostlab/ProstT5"
            elif cls.__name__ == "ProtT5Tokenizer":
                pretrained_model_name_or_path = "Rostlab/prot_t5_xl_uniref50"
            else:
                raise ValueError(f"Unknown tokenizer class: {cls.__name__}")

        transformers_logger = logging.getLogger("transformers.tokenization_utils_base")
        original_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)
        try:
            tokenizer = super().from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                do_lower_case=do_lower_case,
                use_fast=use_fast,
                legacy=legacy,
                **kwargs,
            )
        finally:
            transformers_logger.setLevel(original_level)

        return tokenizer

    def protein_encode(
        self,
        text: Union[str, List[str]],
        padding: bool = False,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> Union[torch.Tensor, List[int], List[List[int]]]:
        if isinstance(text, str):
            text = [text]

        text = [self._preprocess_text(t) for t in text]

        encoded_text = self.batch_encode_plus(
            text,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
        )

        return encoded_text

    def protein_decode(
        self,
        tokens: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ):
        if isinstance(tokens, list) and all(isinstance(x, int) for x in tokens):
            tokens = [tokens]

        decoded_text = self.batch_decode(
            tokens,
            skip_special_tokens=skip_special_tokens,
        )

        decoded_text = [self._postprocess_text(t) for t in decoded_text]

        return decoded_text


class ProstT5Tokenizer(ProteinTokenizer):
    def _preprocess_text(self, text: str) -> str:
        if text.isupper():
            text = "<AA2fold> " + " ".join(list(re.sub(r"[UZOB]", "X", text)))
        else:
            text = "<fold2AA> " + " ".join(list(re.sub(r"[uzob]", "X", text)))
        return text

    def _postprocess_text(self, text: str) -> str:
        return text.replace(" ", "")[9:]


class ProtT5Tokenizer(ProteinTokenizer):
    def _preprocess_text(self, text: str) -> str:
        return " ".join(list(re.sub(r"[UZOB]", "X", text.upper())))

    def _postprocess_text(self, text: str) -> str:
        return text.replace(" ", "")
