"""Text bucket construction, padding, and output-order restoration."""

from typing import List

import numpy as np
import torch


class BatchingMixin:
    def batch_sequences(self, sequences: List[torch.Tensor], axis: int = 0, pad_value: int = 0, max_length: int = None):
        seq = sequences[0]
        ndim = seq.dim()
        if axis < 0:
            axis += ndim
        dtype: torch.dtype = seq.dtype
        pad_value = torch.tensor(pad_value, dtype=dtype)
        seq_lengths = [seq.shape[axis] for seq in sequences]
        if max_length is None:
            max_length = max(seq_lengths)
        else:
            max_length = max(seq_lengths) if max_length < max(seq_lengths) else max_length

        padded_sequences = []
        for seq, length in zip(sequences, seq_lengths):
            padding = [0] * axis + [0, max_length - length] + [0] * (ndim - axis - 1)
            padded_seq = torch.nn.functional.pad(seq, padding, value=pad_value)
            padded_sequences.append(padded_seq)
        batch = torch.stack(padded_sequences)
        return batch

    def to_batch(
        self,
        data: list,
        prompt_data: dict = None,
        batch_size: int = 5,
        threshold: float = 0.75,
        split_bucket: bool = True,
        device: torch.device = torch.device("cpu"),
        precision: torch.dtype = torch.float32,
    ):
        _data: list = []
        index_and_len_list = []
        for idx, item in enumerate(data):
            norm_text_len = len(item["norm_text"])
            index_and_len_list.append([idx, norm_text_len])

        batch_index_list = []
        if split_bucket:
            index_and_len_list.sort(key=lambda x: x[1])
            index_and_len_list = np.array(index_and_len_list, dtype=np.int64)

            batch_index_list_len = 0
            pos = 0
            while pos < index_and_len_list.shape[0]:
                # batch_index_list.append(index_and_len_list[pos:min(pos+batch_size,len(index_and_len_list))])
                pos_end = min(pos + batch_size, index_and_len_list.shape[0])
                while pos < pos_end:
                    batch = index_and_len_list[pos:pos_end, 1].astype(np.float32)
                    score = batch[(pos_end - pos) // 2] / (batch.mean() + 1e-8)
                    if (score >= threshold) or (pos_end - pos == 1):
                        batch_index = index_and_len_list[pos:pos_end, 0].tolist()
                        batch_index_list_len += len(batch_index)
                        batch_index_list.append(batch_index)
                        pos = pos_end
                        break
                    pos_end = pos_end - 1

            assert batch_index_list_len == len(data)

        else:
            for i in range(len(data)):
                if i % batch_size == 0:
                    batch_index_list.append([])
                batch_index_list[-1].append(i)

        for batch_idx, index_list in enumerate(batch_index_list):
            item_list = [data[idx] for idx in index_list]
            phones_list = []
            phones_len_list = []
            # bert_features_list = []
            all_phones_list = []
            all_phones_len_list = []
            all_bert_features_list = []
            norm_text_batch = []
            all_bert_max_len = 0
            all_phones_max_len = 0
            for item in item_list:
                if prompt_data is not None:
                    all_bert_features = torch.cat([prompt_data["bert_features"], item["bert_features"]], 1).to(
                        dtype=precision, device=device
                    )
                    all_phones = torch.LongTensor(prompt_data["phones"] + item["phones"]).to(device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                    # norm_text = prompt_data["norm_text"]+item["norm_text"]
                else:
                    all_bert_features = item["bert_features"].to(dtype=precision, device=device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                    all_phones = phones
                    # norm_text = item["norm_text"]

                all_bert_max_len = max(all_bert_max_len, all_bert_features.shape[-1])
                all_phones_max_len = max(all_phones_max_len, all_phones.shape[-1])

                phones_list.append(phones)
                phones_len_list.append(phones.shape[-1])
                all_phones_list.append(all_phones)
                all_phones_len_list.append(all_phones.shape[-1])
                all_bert_features_list.append(all_bert_features)
                norm_text_batch.append(item["norm_text"])

            phones_batch = phones_list
            all_phones_batch = all_phones_list
            all_bert_features_batch = all_bert_features_list

            max_len = max(all_bert_max_len, all_phones_max_len)
            # phones_batch = self.batch_sequences(phones_list, axis=0, pad_value=0, max_length=max_len)
            #### 直接对phones和bert_features进行pad。（padding策略会影响T2S模型生成的结果，但不直接影响复读概率。影响复读概率的主要因素是mask的策略）
            # all_phones_batch = self.batch_sequences(all_phones_list, axis=0, pad_value=0, max_length=max_len)
            # all_bert_features_batch = all_bert_features_list
            # all_bert_features_batch = torch.zeros((len(all_bert_features_list), 1024, max_len), dtype=precision, device=device)
            # for idx, item in enumerate(all_bert_features_list):
            #     all_bert_features_batch[idx, :, : item.shape[-1]] = item

            # #### 先对phones进行embedding、对bert_features进行project，再pad到相同长度，（padding策略会影响T2S模型生成的结果，但不直接影响复读概率。影响复读概率的主要因素是mask的策略）
            # all_phones_list = [self.t2s_model.model.ar_text_embedding(item.to(self.t2s_model.device)) for item in all_phones_list]
            # all_phones_list = [F.pad(item,(0,0,0,max_len-item.shape[0]),value=0) for item in all_phones_list]
            # all_phones_batch = torch.stack(all_phones_list, dim=0)

            # all_bert_features_list = [self.t2s_model.model.bert_proj(item.to(self.t2s_model.device).transpose(0, 1)) for item in all_bert_features_list]
            # all_bert_features_list = [F.pad(item,(0,0,0,max_len-item.shape[0]), value=0) for item in all_bert_features_list]
            # all_bert_features_batch = torch.stack(all_bert_features_list, dim=0)

            batch = {
                "phones": phones_batch,
                "phones_len": torch.LongTensor(phones_len_list).to(device),
                "all_phones": all_phones_batch,
                "all_phones_len": torch.LongTensor(all_phones_len_list).to(device),
                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch,
                "max_len": max_len,
            }
            _data.append(batch)

        return _data, batch_index_list

    def recovery_order(self, data: list, batch_index_list: list) -> list:
        """
        Recovery the order of the audio according to the batch_index_list.

        Args:
            data (List[list(torch.Tensor)]): the out of order audio .
            batch_index_list (List[list[int]]): the batch index list.

        Returns:
            list (List[torch.Tensor]): the data in the original order.
        """
        length = len(sum(batch_index_list, []))
        _data = [None] * length
        for i, index_list in enumerate(batch_index_list):
            for j, index in enumerate(index_list):
                _data[index] = data[i][j]
        return _data
