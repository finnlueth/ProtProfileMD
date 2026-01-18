from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch
from torch import nn
from transformers import T5EncoderModel


class ProteinEncoder(nn.Module, ABC):
    def __init__(self, pretrained_model: str, **kwargs) -> None:
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model,
            **kwargs,
        )

    @abstractmethod
    def _transform_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        pass

    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]],
        attention_mask: Optional[Union[torch.Tensor, List[List[int]]]],
    ) -> dict[str, Any]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        updated_mask = self._transform_attention_mask(attention_mask)
        outputs = torch.mul(outputs["last_hidden_state"], updated_mask.unsqueeze(-1))
        return {
            "last_hidden_state": outputs,
            "masks": updated_mask,
        }


class ProstT5(ProteinEncoder):
    def __init__(self, pretrained_model: str, **kwargs) -> None:
        super().__init__(pretrained_model, **kwargs)

    def forward(
        self,
        input_ids: Union[torch.Tensor, List[List[int]]],
        attention_mask: Optional[Union[torch.Tensor, List[List[int]]]],
    ) -> dict[str, Any]:
        outputs = super().forward(input_ids, attention_mask)
        outputs["last_hidden_state"] = outputs["last_hidden_state"][:, 1:]
        outputs["masks"] = outputs["masks"][:, 1:]
        return outputs

    def _transform_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.clone()
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
        mask[batch_indices, seq_lengths] = 0
        mask[:, 0] = 0
        return mask


class ProtT5(ProteinEncoder):
    def __init__(self, pretrained_model: str, pretrained_model_kwargs: dict) -> None:
        super().__init__(pretrained_model, pretrained_model_kwargs)

    def _transform_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.clone()
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(attention_mask.size(0), device=attention_mask.device)
        mask[batch_indices, seq_lengths] = 0
        return mask
