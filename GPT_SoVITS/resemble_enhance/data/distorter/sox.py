import logging
import os
import random
import warnings
from functools import partial

import numpy as np
import torch

try:
    import augment
except ImportError:
    raise ImportError(
        "augment is not installed, please install it first using:"
        "\npip install git+https://github.com/facebookresearch/WavAugment@54afcdb00ccc852c2f030f239f8532c9562b550e"
    )

from .base import Effect

_logger = logging.getLogger(__name__)
_DEBUG = bool(os.environ.get("DEBUG", False))


class AttachableEffect(Effect):
    def attach(self, chain: augment.EffectChain) -> augment.EffectChain:
        raise NotImplementedError

    def apply(self, wav: np.ndarray, sr: int):
        chain = augment.EffectChain()
        chain = self.attach(chain)
        tensor = torch.from_numpy(wav)[None].float()  # (1, T)
        tensor = chain.apply(tensor, src_info={"rate": sr}, target_info={"channels": 1, "rate": sr})
        wav = tensor.numpy()[0]  # (T,)
        return wav


class SoxEffect(AttachableEffect):
    def __init__(self, effect_name: str, *args, **kwargs):
        self.effect_name = effect_name
        self.args = args
        self.kwargs = kwargs

    def attach(self, chain: augment.EffectChain) -> augment.EffectChain:
        _logger.debug(f"Attaching {self.effect_name} with {self.args} and {self.kwargs}")
        if not hasattr(chain, self.effect_name):
            raise ValueError(f"EffectChain has no attribute {self.effect_name}")
        return getattr(chain, self.effect_name)(*self.args, **self.kwargs)


class Maybe(AttachableEffect):
    """
    Attach an effect with a probability.
    """

    def __init__(self, prob: float, effect: AttachableEffect):
        self.prob = prob
        self.effect = effect
        if _DEBUG:
            warnings.warn("DEBUG mode is on. Maybe -> Must.")
            self.prob = 1

    def attach(self, chain: augment.EffectChain) -> augment.EffectChain:
        if random.random() > self.prob:
            return chain
        return self.effect.attach(chain)


class Chain(AttachableEffect):
    """
    Attach a chain of effects.
    """

    def __init__(self, *effects: AttachableEffect):
        self.effects = effects

    def attach(self, chain: augment.EffectChain) -> augment.EffectChain:
        for effect in self.effects:
            chain = effect.attach(chain)
        return chain


class Choice(AttachableEffect):
    """
    Attach one of the effects randomly.
    """

    def __init__(self, *effects: AttachableEffect):
        self.effects = effects

    def attach(self, chain: augment.EffectChain) -> augment.EffectChain:
        return random.choice(self.effects).attach(chain)


class Generator:
    def __call__(self) -> str:
        raise NotImplementedError


class Uniform(Generator):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self) -> str:
        return str(random.uniform(self.low, self.high))


class Randint(Generator):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self) -> str:
        return str(random.randint(self.low, self.high))


class Concat(Generator):
    def __init__(self, *parts: Generator | str):
        self.parts = parts

    def __call__(self):
        return "".join([part if isinstance(part, str) else part() for part in self.parts])


class RandomLowpassDistorter(SoxEffect):
    def __init__(self, low=2000, high=16000):
        super().__init__("sinc", "-n", Randint(50, 200), Concat("-", Uniform(low, high)))


class RandomBandpassDistorter(SoxEffect):
    def __init__(self, low=100, high=1000, min_width=2000, max_width=4000):
        super().__init__("sinc", "-n", Randint(50, 200), partial(self._fn, low, high, min_width, max_width))

    @staticmethod
    def _fn(low, high, min_width, max_width):
        start = random.randint(low, high)
        stop = start + random.randint(min_width, max_width)
        return f"{start}-{stop}"


class RandomEqualizer(SoxEffect):
    def __init__(self, low=100, high=4000, q_low=1, q_high=5, db_low: int = -30, db_high: int = 30):
        super().__init__(
            "equalizer",
            Uniform(low, high),
            lambda: f"{random.randint(q_low, q_high)}q",
            lambda: random.randint(db_low, db_high),
        )


class RandomOverdrive(SoxEffect):
    def __init__(self, gain_low=5, gain_high=40, colour_low=20, colour_high=80):
        super().__init__("overdrive", Uniform(gain_low, gain_high), Uniform(colour_low, colour_high))


class RandomReverb(Chain):
    def __init__(self, deterministic=False):
        super().__init__(
            SoxEffect(
                "reverb",
                Uniform(50, 50) if deterministic else Uniform(0, 100),
                Uniform(50, 50) if deterministic else Uniform(0, 100),
                Uniform(50, 50) if deterministic else Uniform(0, 100),
            ),
            SoxEffect("channels", 1),
        )


class Flanger(SoxEffect):
    def __init__(self):
        super().__init__("flanger")


class Phaser(SoxEffect):
    def __init__(self):
        super().__init__("phaser")
