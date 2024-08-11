# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/data/dataset.py
# reference: https://github.com/lifeiteng/vall-e
import pdb
import sys

# sys.path.append("/data/docker/liujing04/gpt-vits/mq-vits-s1bert_no_bert")
import traceback, os
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import torch, json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer

version = os.environ.get('version',None)

from text import cleaned_text_to_sequence

# from config import exp_dir


def batch_sequences(sequences: List[np.array], axis: int = 0, pad_value: int = 0):
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_value = dtype.type(pad_value)
    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = np.max(seq_lengths)

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = (
            [(0, 0)] * axis + [(0, max_length - length)] + [(0, 0)] * (ndim - axis - 1)
        )
        padded_seq = np.pad(seq, padding, mode="constant", constant_values=pad_value)
        padded_sequences.append(padded_seq)
    batch = np.stack(padded_sequences)
    return batch


class Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(
        self,
        phoneme_path: str,
        semantic_path: str,
        max_sample: int = None,
        max_sec: int = 100,
        pad_val: int = 1024,
        # min value of phoneme/sec
        min_ps_ratio: int = 3,
        # max value of phoneme/sec
        max_ps_ratio: int = 25,
    ) -> None:
        super().__init__()

        self.semantic_data = pd.read_csv(
            semantic_path, delimiter="\t", encoding="utf-8"
        )
        # get dict
        self.path2 = phoneme_path  # "%s/2-name2text.txt"%exp_dir#phoneme_path
        self.path3 = "%s/3-bert" % (
            os.path.dirname(phoneme_path)
        )  # "%s/3-bert"%exp_dir#bert_dir
        self.path6 = semantic_path  # "%s/6-name2semantic.tsv"%exp_dir#semantic_path
        assert os.path.exists(self.path2)
        assert os.path.exists(self.path6)
        self.phoneme_data = {}
        with open(self.path2, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        for line in lines:
            tmp = line.split("\t")
            if len(tmp) != 4:
                continue
            self.phoneme_data[tmp[0]] = [tmp[1], tmp[2], tmp[3]]

        # self.phoneme_data = np.load(phoneme_path, allow_pickle=True).item()
        # pad for semantic tokens
        self.PAD: int = pad_val
        # self.hz = 25
        # with open("/data/docker/liujing04/gpt-vits/mq-vits-s1bert_no_bert/configs/s2.json", "r") as f:data = f.read()
        # data=json.loads(data)["model"]["semantic_frame_rate"]#50hz
        # self.hz=int(data[:-2])#
        self.hz = int(os.environ.get("hz", "25hz")[:-2])

        # max seconds of semantic token
        self.max_sec = max_sec
        self.min_ps_ratio = min_ps_ratio
        self.max_ps_ratio = max_ps_ratio

        if max_sample is not None:
            self.semantic_data = self.semantic_data[:max_sample]

        # {idx: (semantic, phoneme)}
        # semantic list, phoneme list
        self.semantic_phoneme = []
        self.item_names = []

        self.inited = False

        if not self.inited:
            # 调用初始化函数
            self.init_batch()
            self.inited = True
            del self.semantic_data
            del self.phoneme_data
        # self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        # self.tokenizer = AutoTokenizer.from_pretrained("/data/docker/liujing04/bert-vits2/Bert-VITS2-master20231106/bert/chinese-roberta-wwm-ext-large")

    def init_batch(self):
        semantic_data_len = len(self.semantic_data)
        phoneme_data_len = len(self.phoneme_data.keys())
        print("semantic_data_len:", semantic_data_len)
        print("phoneme_data_len:", phoneme_data_len)
        print(self.semantic_data)
        idx = 0
        num_not_in = 0
        num_deleted_bigger = 0
        num_deleted_ps = 0
        for i in range(semantic_data_len):
            # 先依次遍历
            # get str
            item_name = self.semantic_data.iloc[i,0]
            # print(self.phoneme_data)
            try:
                phoneme, word2ph, text = self.phoneme_data[item_name]
            except Exception:
                traceback.print_exc()
                # print(f"{item_name} not in self.phoneme_data !")
                num_not_in += 1
                continue

            semantic_str = self.semantic_data.iloc[i,1]
            # get token list
            semantic_ids = [int(idx) for idx in semantic_str.split(" ")]
            # (T), 是否需要变成 (1, T) -> 不需要，因为需要求 len
            # 过滤掉太长的样本
            if (
                len(semantic_ids) > self.max_sec * self.hz
            ):  #########1###根据token个数推测总时长过滤时长60s（config里）#40*25=1k
                num_deleted_bigger += 1
                continue
            # (T, ), 这个速度不会很慢，所以可以在一开始就处理，无需在 __getitem__ 里面单个处理####
            phoneme = phoneme.split(" ")

            try:
                phoneme_ids = cleaned_text_to_sequence(phoneme, version)
            except:
                traceback.print_exc()
                # print(f"{item_name} not in self.phoneme_data !")
                num_not_in += 1
                continue
            # if len(phoneme_ids) >400:###########2：改为恒定限制为semantic/2.5就行
            if (
                len(phoneme_ids) > self.max_sec * self.hz / 2.5
            ):  ###########2：改为恒定限制为semantic/2.5就行
                num_deleted_ps += 1
                continue
            # if len(semantic_ids) > 1000:###########3
            #     num_deleted_bigger += 1
            #     continue

            ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self.hz)

            if (
                ps_ratio > self.max_ps_ratio or ps_ratio < self.min_ps_ratio
            ):  ##########4#3~25#每秒多少个phone
                num_deleted_ps += 1
                # print(item_name)
                continue

            self.semantic_phoneme.append((semantic_ids, phoneme_ids))
            idx += 1
            self.item_names.append(item_name)

        min_num = 100  # 20直接不补#30补了也不存ckpt
        leng = len(self.semantic_phoneme)
        if leng < min_num:
            tmp1 = self.semantic_phoneme
            tmp2 = self.item_names
            self.semantic_phoneme = []
            self.item_names = []
            for _ in range(max(2, int(min_num / leng))):
                self.semantic_phoneme += tmp1
                self.item_names += tmp2
        if num_not_in > 0:
            print(f"there are {num_not_in} semantic datas not in phoneme datas")
        if num_deleted_bigger > 0:
            print(
                f"deleted {num_deleted_bigger} audios who's duration are bigger than {self.max_sec} seconds"
            )
        if num_deleted_ps > 0:
            # 4702 for LibriTTS, LirbriTTS 是标注数据, 是否需要筛？=> 需要，有值为 100 的极端值
            print(
                f"deleted {num_deleted_ps} audios who's phoneme/sec are bigger than {self.max_ps_ratio} or smaller than {self.min_ps_ratio}"
            )
        """
        there are 31 semantic datas not in phoneme datas
        deleted 34 audios who's duration are bigger than 54 seconds
        deleted 3190 audios who's phoneme/sec are bigger than 25 or smaller than 3
        dataset.__len__(): 366463

        """
        # 345410 for LibriTTS
        print("dataset.__len__():", self.__len__())

    def __get_item_names__(self) -> List[str]:
        return self.item_names

    def __len__(self) -> int:
        return len(self.semantic_phoneme)

    def __getitem__(self, idx: int) -> Dict:
        semantic_ids, phoneme_ids = self.semantic_phoneme[idx]
        item_name = self.item_names[idx]
        phoneme_ids_len = len(phoneme_ids)
        # semantic tokens target
        semantic_ids_len = len(semantic_ids)

        flag = 0
        path_bert = "%s/%s.pt" % (self.path3, item_name)
        if os.path.exists(path_bert) == True:
            bert_feature = torch.load(path_bert, map_location="cpu")
        else:
            flag = 1
        if flag == 1:
            # bert_feature=torch.zeros_like(phoneme_ids,dtype=torch.float32)
            bert_feature = None
        else:
            assert bert_feature.shape[-1] == len(phoneme_ids)
        return {
            "idx": idx,
            "phoneme_ids": phoneme_ids,
            "phoneme_ids_len": phoneme_ids_len,
            "semantic_ids": semantic_ids,
            "semantic_ids_len": semantic_ids_len,
            "bert_feature": bert_feature,
        }

    def get_sample_length(self, idx: int):
        semantic_ids = self.semantic_phoneme[idx][0]
        sec = 1.0 * len(semantic_ids) / self.hz
        return sec

    def collate(self, examples: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phoneme_ids: List[torch.Tensor] = []
        phoneme_ids_lens: List[int] = []
        semantic_ids: List[torch.Tensor] = []
        semantic_ids_lens: List[int] = []
        # return

        for item in examples:
            sample_index.append(item["idx"])
            phoneme_ids.append(np.array(item["phoneme_ids"], dtype=np.int64))
            semantic_ids.append(np.array(item["semantic_ids"], dtype=np.int64))
            phoneme_ids_lens.append(item["phoneme_ids_len"])
            semantic_ids_lens.append(item["semantic_ids_len"])

        # pad 0
        phoneme_ids = batch_sequences(phoneme_ids)
        semantic_ids = batch_sequences(semantic_ids, pad_value=self.PAD)

        # # convert each batch to torch.tensor
        phoneme_ids = torch.tensor(phoneme_ids)
        semantic_ids = torch.tensor(semantic_ids)
        phoneme_ids_lens = torch.tensor(phoneme_ids_lens)
        semantic_ids_lens = torch.tensor(semantic_ids_lens)
        bert_padded = torch.FloatTensor(len(examples), 1024, max(phoneme_ids_lens))
        bert_padded.zero_()

        for idx, item in enumerate(examples):
            bert = item["bert_feature"]
            if bert != None:
                bert_padded[idx, :, : bert.shape[-1]] = bert

        return {
            # List[int]
            "ids": sample_index,
            # torch.Tensor (B, max_phoneme_length)
            "phoneme_ids": phoneme_ids,
            # torch.Tensor (B)
            "phoneme_ids_len": phoneme_ids_lens,
            # torch.Tensor (B, max_semantic_ids_length)
            "semantic_ids": semantic_ids,
            # torch.Tensor (B)
            "semantic_ids_len": semantic_ids_lens,
            # torch.Tensor (B, 1024, max_phoneme_length)
            "bert_feature": bert_padded,
        }


if __name__ == "__main__":
    root_dir = "/data/docker/liujing04/gpt-vits/prepare/dump_mix/"
    dataset = Text2SemanticDataset(
        phoneme_path=root_dir + "phoneme_train.npy",
        semantic_path=root_dir + "semantic_train.tsv",
    )

    batch_size = 12
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=dataset.collate, shuffle=False
    )
    for i, batch in enumerate(dataloader):
        if i % 1000 == 0:
            print(i)
        # if i == 0:
        #     print('batch["ids"]:', batch["ids"])
        # print('batch["phoneme_ids"]:', batch["phoneme_ids"],
        #       batch["phoneme_ids"].shape)
        # print('batch["phoneme_ids_len"]:', batch["phoneme_ids_len"],
        #       batch["phoneme_ids_len"].shape)
        # print('batch["semantic_ids"]:', batch["semantic_ids"],
        #       batch["semantic_ids"].shape)
        # print('batch["semantic_ids_len"]:', batch["semantic_ids_len"],
        #       batch["semantic_ids_len"].shape)
