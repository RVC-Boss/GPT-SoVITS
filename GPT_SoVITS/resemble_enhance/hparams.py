import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()


def _make_stft_cfg(hop_length, win_length=None):
    if win_length is None:
        win_length = 4 * hop_length
    n_fft = 2 ** (win_length - 1).bit_length()
    return dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _build_rich_table(rows, columns, title=None):
    table = Table(title=title, header_style=None)
    for column in columns:
        table.add_column(column.capitalize(), justify="left")
    for row in rows:
        table.add_row(*map(str, row))
    return Panel(table, expand=False)


def _rich_print_dict(d, title="Config", key="Key", value="Value"):
    console.print(_build_rich_table(d.items(), [key, value], title))


@dataclass(frozen=True)
class HParams:
    # Dataset
    fg_dir: Path = Path("data/fg")
    bg_dir: Path = Path("data/bg")
    rir_dir: Path = Path("data/rir")
    load_fg_only: bool = False
    praat_augment_prob: float = 0

    # Audio settings
    wav_rate: int = 44_100
    n_fft: int = 2048
    win_size: int = 2048
    hop_size: int = 420  # 9.5ms
    num_mels: int = 128
    stft_magnitude_min: float = 1e-4
    preemphasis: float = 0.97
    mix_alpha_range: tuple[float, float] = (0.2, 0.8)

    # Training
    nj: int = 64
    training_seconds: float = 1.0
    batch_size_per_gpu: int = 16
    min_lr: float = 1e-5
    max_lr: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 1_000_000
    gradient_clipping: float = 1.0

    @property
    def deepspeed_config(self):
        return {
            "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
            "optimizer": {
                "type": "Adam",
                "params": {"lr": float(self.min_lr)},
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": float(self.min_lr),
                    "warmup_max_lr": float(self.max_lr),
                    "warmup_num_steps": self.warmup_steps,
                    "total_num_steps": self.max_steps,
                    "warmup_type": "linear",
                },
            },
            "gradient_clipping": self.gradient_clipping,
        }

    @property
    def stft_cfgs(self):
        assert self.wav_rate == 44_100, f"wav_rate must be 44_100, got {self.wav_rate}"
        return [_make_stft_cfg(h) for h in (100, 256, 512)]

    @classmethod
    def from_yaml(cls, path: Path) -> "HParams":
        logger.info(f"Reading hparams from {path}")
        # First merge to fix types (e.g., str -> Path)
        return cls(**dict(OmegaConf.merge(cls(), OmegaConf.load(path))))

    def save_if_not_exists(self, run_dir: Path):
        path = run_dir / "hparams.yaml"
        if path.exists():
            logger.info(f"{path} already exists, not saving")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(asdict(self), str(path))

    @classmethod
    def load(cls, run_dir, yaml: Path | None = None):
        hps = []

        if (run_dir / "hparams.yaml").exists():
            hps.append(cls.from_yaml(run_dir / "hparams.yaml"))

        if yaml is not None:
            hps.append(cls.from_yaml(yaml))

        if len(hps) == 0:
            hps.append(cls())

        for hp in hps[1:]:
            if hp != hps[0]:
                errors = {}
                for k, v in asdict(hp).items():
                    if getattr(hps[0], k) != v:
                        errors[k] = f"{getattr(hps[0], k)} != {v}"
                raise ValueError(f"Found inconsistent hparams: {errors}, consider deleting {run_dir}")

        return hps[0]

    def print(self):
        _rich_print_dict(asdict(self), title="HParams")
