import logging
from functools import cache

import torch

from ..inference import inference
from .train import Denoiser, HParams

logger = logging.getLogger(__name__)


@cache
def load_denoiser(run_dir, device):
    if run_dir is None:
        return Denoiser(HParams())
    hp = HParams.load(run_dir)
    denoiser = Denoiser(hp)
    path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    state_dict = torch.load(path, map_location="cpu")["module"]
    denoiser.load_state_dict(state_dict)
    denoiser.eval()
    denoiser.to(device)
    return denoiser


@torch.inference_mode()
def denoise(dwav, sr, run_dir, device):
    denoiser = load_denoiser(run_dir, device)
    return inference(model=denoiser, dwav=dwav, sr=sr, device=device)
