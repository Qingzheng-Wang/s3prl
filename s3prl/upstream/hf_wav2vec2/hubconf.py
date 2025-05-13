from .expert import UpstreamExpert as _UpstreamExpert
from .expert import UpstreamExpertLang2VecCondition as _UpstreamExpertLang2VecCondition


def hf_wav2vec2_custom(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)

def hf_wav2vec2_lang2vec_condition(ckpt, *args, **kwargs):
    return _UpstreamExpertLang2VecCondition(ckpt, *args, **kwargs)
