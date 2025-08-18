from .expert import UpstreamExpert as _UpstreamExpert
from .expert import UpstreamExpertCondition as _UpstreamExpertCondition


def hf_wav2vec2_custom(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)

def hf_wav2vec2_condition(ckpt, *args, **kwargs):
    return _UpstreamExpertCondition(ckpt, *args, **kwargs)
