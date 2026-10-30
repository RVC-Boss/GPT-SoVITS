"""Text-to-semantic inference wrapper preserving the checkpoint's ``model`` prefix."""

from torch import nn

from AR.models.t2s_model import Text2SemanticDecoder


class Text2SemanticInference(nn.Module):
    """Checkpoint-compatible text-to-semantic model used by the TTS service."""

    def __init__(self, config):
        super().__init__()
        self.model = Text2SemanticDecoder(config=config)
