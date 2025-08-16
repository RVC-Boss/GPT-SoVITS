# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import argparse
import logging
import os
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  # WandbLogger
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy
from pytorch_lightning.strategies.strategy import Strategy

from GPT_SoVITS.AR.data.data_module import Text2SemanticDataModule
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.AR.utils import get_newest_ckpt
from GPT_SoVITS.AR.utils.io import load_yaml_config
from GPT_SoVITS.process_ckpt import save

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")


os.environ["MASTER_ADDR"] = "localhost"
if platform.system() == "Windows":
    os.environ["USE_LIBUV"] = "0"

torch.set_grad_enabled(True)


class ARModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, monitor_candidates)
            if self.if_save_latest is True:  # 如果设置只保存最后一个ckpt，在保存下一个ckpt后要清理掉之前的所有ckpt
                to_clean = list(os.listdir(self.dirpath))
                for name in to_clean:
                    try:
                        os.remove(f"{self.dirpath}/{name}")
                    except Exception as _:
                        pass
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if self.if_save_every_weights is True:
                    to_save_od: OrderedDict[str, Any] = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = dictt[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = f"GPT-e{trainer.current_epoch + 1}"
                    # torch.save(
                    # print(os.environ)
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        save(
                            to_save_od,
                            f"{self.half_weights_save_dir}/{self.exp_name}-e{trainer.current_epoch + 1}.ckpt",
                        )
            self._save_last_checkpoint(trainer, monitor_candidates)


def main(args):
    config = load_yaml_config(args.config_file)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            strategy: Strategy = DDPStrategy(
                process_group_backend="nccl" if platform.system() != "Windows" else "gloo", find_unused_parameters=False
            )
        else:
            strategy = SingleDeviceStrategy("cuda:0")
    else:
        strategy = SingleDeviceStrategy("cpu")

    seed_everything(config["train"]["seed"], workers=True)

    ckpt_callback: ModelCheckpoint = ARModelCheckpoint(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)

    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        limit_val_batches=0,
        devices=-1 if torch.cuda.is_available() else 1,
        benchmark=False,
        fast_dev_run=False,
        strategy=strategy,
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        use_distributed_sampler=False,  # 非常简单的修改，但解决了采用自定义的 bucket_sampler 下训练步数不一致的问题！
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(config, output_dir)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
    )

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="configs/s1longer.yaml",
        help="path of config file",
    )
    # args for dataset
    # parser.add_argument('--train_semantic_path',type=str,default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/6-name2semantic.tsv')
    # parser.add_argument('--train_phoneme_path', type=str, default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/2-name2text.txt')

    # parser.add_argument('--dev_semantic_path', type=str, default='dump_mix/semantic_dev.tsv')
    # parser.add_argument('--dev_phoneme_path', type=str, default='dump_mix/phoneme_dev.npy')
    # parser.add_argument('--output_dir',type=str,default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/logs_s1',help='directory to save the results')
    # parser.add_argument('--output_dir',type=str,default='/liujing04/gpt_logs/s1/xuangou_ft',help='directory to save the results')

    args = parser.parse_args()
    logging.info(str(args))
    main(args)
