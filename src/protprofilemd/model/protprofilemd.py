from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
)

from .utils import LOSS_FUNCTION_MAP, PREDICTION_HEAD_MAP, PROTEIN_ENCODER_MAP

from ..utils.definitions import STRUCTURE_BG


# @PreTrainedModel.register_for_auto_class()
class ProtProfileMDConfig(PretrainedConfig):
    model_type = "ProtProfileMD"

    def __init__(
        self,
        base_model: str = None,
        base_model_kwargs: dict = None,
        profile_head: str = None,
        profile_head_kwargs: dict = None,
        loss_function: str = None,
        loss_function_kwargs: dict = None,
    ) -> None:
        self.base_model = base_model
        self.base_model_kwargs = base_model_kwargs
        self.profile_head = profile_head
        self.profile_head_kwargs = profile_head_kwargs
        self.loss_function = loss_function
        self.loss_function_kwargs = loss_function_kwargs

        super().__init__()


# @PreTrainedModel.register_for_auto_class()
class ProtProfileMD(PreTrainedModel):
    config_class = ProtProfileMDConfig
    base_model_prefix = "ProtProfileMD"
    supports_gradient_checkpointing = True

    def __init__(self, config: ProtProfileMDConfig) -> None:
        super().__init__(config)
        self.protein_encoder = PROTEIN_ENCODER_MAP[config.base_model](
            pretrained_model=config.base_model,
            **config.base_model_kwargs,
        )
        self.profile_head = PREDICTION_HEAD_MAP[config.profile_head](
            **config.profile_head_kwargs,
        )
        self.loss_fct = LOSS_FUNCTION_MAP[config.loss_function](
            **config.loss_function_kwargs,
        )

        self.post_init()
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            # module.weight.data.zero_()
            
            if module.bias is not None:
                # module.bias.data.zero_()
                module.bias.data.normal_(mean=0.0, std=0.02)
                
                # structure_bg_tensor = torch.tensor(
                #     STRUCTURE_BG, 
                #     dtype=module.bias.dtype, 
                #     device=module.bias.device
                # )
                # module.bias.data = structure_bg_tensor
                # module.bias.data = torch.log(structure_bg_tensor)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        profiles: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        encoder_outputs = self.protein_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # [batch_size, seq_len, 20]
        profiles_pred = self.profile_head(encoder_outputs["last_hidden_state"])

        loss = None
        if profiles is not None:
            # [batch_size * seq_len, 20]
            y_pred = profiles_pred.flatten(end_dim=1)
            y_true = profiles.flatten(end_dim=1)

            y_pred_mask = encoder_outputs["masks"].flatten(end_dim=1)
            y_true_mask = ~torch.any(y_true == -100, dim=1)

            y_pred = y_pred[y_pred_mask.bool()]
            y_true = y_true[y_true_mask.bool()]

            loss = self.loss_fct(torch.log(y_pred), y_true)

        return {
            "loss": loss,
            "profiles": profiles_pred,
            "masks": encoder_outputs["masks"],
            # "logits": encoder_outputs["last_hidden_state"],
        }


AutoConfig.register(ProtProfileMDConfig.model_type, ProtProfileMDConfig)
AutoModel.register(ProtProfileMDConfig, ProtProfileMD)
