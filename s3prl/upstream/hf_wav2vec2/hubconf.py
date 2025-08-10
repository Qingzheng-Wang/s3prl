from .expert import UpstreamExpert as _UpstreamExpert
from .expert import UpstreamExpertLang2VecCondition as _UpstreamExpertLang2VecCondition
from .expert import UpstreamExpertCondition as _UpstreamExpertCondition


def hf_wav2vec2_custom(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)

def hf_wav2vec2_lang2vec_condition(ckpt, *args, **kwargs):
    return _UpstreamExpertLang2VecCondition(ckpt, *args, **kwargs)

def hf_wav2vec2_condition(ckpt, *args, **kwargs):
    return _UpstreamExpertCondition(ckpt, *args, **kwargs)
