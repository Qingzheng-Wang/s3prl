from collections import OrderedDict
from typing import Dict, List, Union
import torch
import torch.nn as nn
import logging
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

HIDDEN_DIM = 8

logger = logging.getLogger(__name__)

try:
    from espnet2.tasks.ssl import SSLTask
except ModuleNotFoundError:
    SSLTask = None
    logger.warning("ESPnet is not installed, cannot use espnet_hubert upstream")

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "XEUS"

        print(
            f"{self.name} - You can use model_config to construct your customized model: {model_config}"
        )
        print(f"{self.name} - You can use ckpt to load your pretrained weights: {ckpt}")
        print(
            f"{self.name} - If you store the pretrained weights and model config in a single file, "
            "you can just choose one argument (ckpt or model_config) to pass. It's up to you!"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.xeus_train_args = SSLTask.build_model_from_file(
            model_config,
            ckpt,
            self.device
        )

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 320

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        wavs = pad_sequence(wavs, batch_first=True)
        print(f"Shape of wavs: {wavs.shape}")
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(self.device)
        # wavs: (batch_size, max_len, 1)

        feats, _, _, feats_lens = self.model.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)

        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        return {
            "hidden_states": feats,
        }
