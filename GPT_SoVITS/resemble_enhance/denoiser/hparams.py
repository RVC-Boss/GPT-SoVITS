from dataclasses import dataclass

from ..hparams import HParams as HParamsBase


@dataclass(frozen=True)
class HParams(HParamsBase):
    batch_size_per_gpu: int = 128
    distort_prob: float = 0.5
