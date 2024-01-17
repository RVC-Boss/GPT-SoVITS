# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/bucketsampler.py
import itertools
import math
import random
from random import shuffle
from typing import Iterator
from typing import Optional
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import Sampler

__all__ = [
    "DistributedBucketSampler",
]

T_co = TypeVar("T_co", covariant=True)


class DistributedBucketSampler(Sampler[T_co]):
    r"""
    sort the dataset wrt. input length
    divide samples into buckets
    sort within buckets
    divide buckets into batches
    sort batches
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 32,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            torch.cuda.set_device(rank)
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if (
            self.drop_last and len(self.dataset) % self.num_replicas != 0
        ):  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas)
                / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas
            )  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.id_with_length = self._get_sample_lengths()
        self.id_buckets = self.make_buckets(bucket_width=2.0)

    def _get_sample_lengths(self):
        id_with_lengths = []
        for i in range(len(self.dataset)):
            id_with_lengths.append((i, self.dataset.get_sample_length(i)))
        id_with_lengths.sort(key=lambda x: x[1])
        return id_with_lengths

    def make_buckets(self, bucket_width: float = 2.0):
        buckets = []
        cur = []
        max_sec = bucket_width
        for id, sec in self.id_with_length:
            if sec < max_sec:
                cur.append(id)
            else:
                buckets.append(cur)
                cur = [id]
                max_sec += bucket_width
        if len(cur) > 0:
            buckets.append(cur)
        return buckets

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            random.seed(self.epoch + self.seed)
            shuffled_bucket = []
            for buc in self.id_buckets:
                buc_copy = buc.copy()
                shuffle(buc_copy)
                shuffled_bucket.append(buc_copy)
            grouped_batch_size = self.batch_size * self.num_replicas
            shuffled_bucket = list(itertools.chain(*shuffled_bucket))
            n_batch = int(math.ceil(len(shuffled_bucket) / grouped_batch_size))
            batches = [
                shuffled_bucket[b * grouped_batch_size : (b + 1) * grouped_batch_size]
                for b in range(n_batch)
            ]
            shuffle(batches)
            indices = list(itertools.chain(*batches))
        else:
            # type: ignore[arg-type]
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
