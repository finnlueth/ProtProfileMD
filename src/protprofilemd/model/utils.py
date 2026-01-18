from torch import nn

from protprofilemd.model.protein_tokenizer import ProstT5Tokenizer, ProtT5Tokenizer

from .protprofilemd_heads import ProfileHeadLinear, ProfileHeadConv1D
from .protein_encoder import ProstT5, ProtT5

PREDICTION_HEAD_MAP = {
    'linear': ProfileHeadLinear,
    # TODO: 'linear_deep': ProfileHeadLinearDeep,
    'conv1d': ProfileHeadConv1D,
}

CONFIG_MAP = {
    'MD_train' : 'configs/protprofile_train.yaml',
    'MD_inference' : 'configs/protprofile_inference.yaml',
}

PROTEIN_ENCODER_MAP = {
    'Rostlab/ProstT5': ProstT5,
    'Rostlab/prot_t5_xl_uniref50': ProtT5,
}

LOSS_FUNCTION_MAP = {
    'kldiv': nn.KLDivLoss,
    'mse': nn.MSELoss,
}

TOKENIZER_MAP = {
    'Rostlab/ProstT5': ProstT5Tokenizer,
    'Rostlab/prot_t5_xl_uniref50': ProtT5Tokenizer,
}