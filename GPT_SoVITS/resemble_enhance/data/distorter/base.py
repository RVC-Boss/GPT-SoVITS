import itertools
import os
import random
import time
import warnings

import numpy as np

_DEBUG = bool(os.environ.get("DEBUG", False))


class Effect:
    def apply(self, wav: np.ndarray, sr: int):
        """
        Args:
            wav: (T)
            sr: sample rate
        Returns:
            wav: (T) with the same sample rate of `sr`
        """
        raise NotImplementedError

    def __call__(self, wav: np.ndarray, sr: int):
        """
        Args:
            wav: (T)
            sr: sample rate
        Returns:
            wav: (T) with the same sample rate of `sr`
        """
        assert len(wav.shape) == 1, wav.shape

        if _DEBUG:
            start = time.time()
        else:
            start = None

        shape = wav.shape
        assert wav.ndim == 1, f"{self}: Expected wav.ndim == 1, got {wav.ndim}."
        wav = self.apply(wav, sr)
        assert shape == wav.shape, f"{self}: {shape} != {wav.shape}."

        if start is not None:
            end = time.time()
            print(f"{self.__class__.__name__}: {end - start:.3f} sec")

        return wav


class Chain(Effect):
    def __init__(self, *effects):
        super().__init__()

        self.effects = effects

    def apply(self, wav, sr):
        for effect in self.effects:
            wav = effect(wav, sr)
        return wav


class Maybe(Effect):
    def __init__(self, prob, effect):
        super().__init__()

        self.prob = prob
        self.effect = effect

        if _DEBUG:
            warnings.warn("DEBUG mode is on. Maybe -> Must.")
            self.prob = 1

    def apply(self, wav, sr):
        if random.random() > self.prob:
            return wav
        return self.effect(wav, sr)


class Choice(Effect):
    def __init__(self, *effects, **kwargs):
        super().__init__()
        self.effects = effects
        self.kwargs = kwargs

    def apply(self, wav, sr):
        return np.random.choice(self.effects, **self.kwargs)(wav, sr)


class Permutation(Effect):
    def __init__(self, *effects, n: int | None = None):
        super().__init__()
        self.effects = effects
        self.n = n

    def apply(self, wav, sr):
        if self.n is None:
            n = np.random.binomial(len(self.effects), 0.5)
        else:
            n = self.n
        if n == 0:
            return wav
        perms = itertools.permutations(self.effects, n)
        effects = random.choice(list(perms))
        return Chain(*effects)(wav, sr)
