from ...hparams import HParams
from .base import Chain, Choice, Permutation
from .custom import RandomGaussianNoise, RandomRIR


class Distorter(Chain):
    def __init__(self, hp: HParams, training: bool = False, mode: str = "enhancer"):
        # Lazy import
        from .sox import RandomBandpassDistorter, RandomEqualizer, RandomLowpassDistorter, RandomOverdrive, RandomReverb

        if training:
            permutation = Permutation(
                RandomRIR(hp.rir_dir),
                RandomReverb(),
                RandomGaussianNoise(),
                RandomOverdrive(),
                RandomEqualizer(),
                Choice(
                    RandomLowpassDistorter(),
                    RandomBandpassDistorter(),
                ),
            )
            if mode == "denoiser":
                super().__init__(permutation)
            else:
                # 80%: distortion, 20%: clean
                super().__init__(Choice(permutation, Chain(), p=[0.8, 0.2]))
        else:
            super().__init__(
                RandomRIR(hp.rir_dir, deterministic=True),
                RandomReverb(deterministic=True),
            )
