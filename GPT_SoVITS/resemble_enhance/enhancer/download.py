import logging
from pathlib import Path

import torch

RUN_NAME = "enhancer_stage2"

logger = logging.getLogger(__name__)


def get_url(relpath):
    return f"https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/{RUN_NAME}/{relpath}?download=true"


def get_path(relpath):
    return Path(__file__).parent.parent / "model_repo" / RUN_NAME / relpath


def download():
    relpaths = ["hparams.yaml", "ds/G/latest", "ds/G/default/mp_rank_00_model_states.pt"]
    for relpath in relpaths:
        path = get_path(relpath)
        if path.exists():
            continue
        url = get_url(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path))
    return get_path("")
