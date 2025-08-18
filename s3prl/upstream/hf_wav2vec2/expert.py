import logging

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        try:
            self.extracter = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
        except:
            alter_extractor = "facebook/wav2vec2-base"
            logger.info(
                f"The model {ckpt} on huggingface does not have a correspoinding feature extractor. "
                f"Using {alter_extractor}'s feature extractor as the alternative."
            )
            self.extracter = Wav2Vec2FeatureExtractor.from_pretrained(alter_extractor)
        self.model = Wav2Vec2Model.from_pretrained(ckpt)

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]
        input_values = self.extracter(
            wavs,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
            sampling_rate=SAMPLE_RATE,
        ).to(device)
        output_values = self.model(**input_values, output_hidden_states=True)

        return {"hidden_states": output_values.hidden_states}


class UpstreamExpertCondition(UpstreamExpert):
    def __init__(self, ckpt, **kwds):
        super().__init__(ckpt, **kwds)
        try:
            from transformers import Wav2Vec2ModelCondition
        except Exception as e:
            raise ImportError(
                "Error: Wav2Vec2ModelCondition is not found.\n"
                "Please install the modified transformers version:\n"
                "  If you have already installed transformers, please uninstall it first.\n"
                "  (optional) pip uninstall transformers\n"
                "  git clone -b v4.51.3-qingzheng https://github.com/Qingzheng-Wang/transformers.git\n"
                "  cd transformers\n"
                "  pip install -e ."
            )
        self.model = Wav2Vec2ModelCondition.from_pretrained(ckpt)
    
    def forward(self, wavs, labels=None):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]
        input_values = self.extracter(
            wavs,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
            sampling_rate=SAMPLE_RATE,
        ).to(device)
        output_values = self.model(**input_values, output_hidden_states=True, labels=labels)

        return {
            "hidden_states": output_values.hidden_states, 
            "intermediate_lang2vec_preds": output_values.intermediate_lang2vec_preds,
        }
