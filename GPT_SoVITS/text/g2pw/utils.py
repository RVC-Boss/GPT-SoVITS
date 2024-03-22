# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Credits
    This code is modified from https://github.com/GitYCC/g2pW
"""
import os
import re


def wordize_and_map(text: str):
    words = []
    index_map_from_text_to_word = []
    index_map_from_word_to_text = []
    while len(text) > 0:
        match_space = re.match(r'^ +', text)
        if match_space:
            space_str = match_space.group(0)
            index_map_from_text_to_word += [None] * len(space_str)
            text = text[len(space_str):]
            continue

        match_en = re.match(r'^[a-zA-Z0-9]+', text)
        if match_en:
            en_word = match_en.group(0)

            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + len(en_word)
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))

            index_map_from_text_to_word += [len(words)] * len(en_word)

            words.append(en_word)
            text = text[len(en_word):]
        else:
            word_start_pos = len(index_map_from_text_to_word)
            word_end_pos = word_start_pos + 1
            index_map_from_word_to_text.append((word_start_pos, word_end_pos))

            index_map_from_text_to_word += [len(words)]

            words.append(text[0])
            text = text[1:]
    return words, index_map_from_text_to_word, index_map_from_word_to_text


def tokenize_and_map(tokenizer, text: str):
    words, text2word, word2text = wordize_and_map(text=text)

    tokens = []
    index_map_from_token_to_text = []
    for word, (word_start, word_end) in zip(words, word2text):
        word_tokens = tokenizer.tokenize(word)

        if len(word_tokens) == 0 or word_tokens == ['[UNK]']:
            index_map_from_token_to_text.append((word_start, word_end))
            tokens.append('[UNK]')
        else:
            current_word_start = word_start
            for word_token in word_tokens:
                word_token_len = len(re.sub(r'^##', '', word_token))
                index_map_from_token_to_text.append(
                    (current_word_start, current_word_start + word_token_len))
                current_word_start = current_word_start + word_token_len
                tokens.append(word_token)

    index_map_from_text_to_token = text2word
    for i, (token_start, token_end) in enumerate(index_map_from_token_to_text):
        for token_pos in range(token_start, token_end):
            index_map_from_text_to_token[token_pos] = i

    return tokens, index_map_from_text_to_token, index_map_from_token_to_text


def _load_config(config_path: os.PathLike):
    import importlib.util
    spec = importlib.util.spec_from_file_location('__init__', config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


default_config_dict = {
    'manual_seed': 1313,
    'model_source': 'bert-base-chinese',
    'window_size': 32,
    'num_workers': 2,
    'use_mask': True,
    'use_char_phoneme': False,
    'use_conditional': True,
    'param_conditional': {
        'affect_location': 'softmax',
        'bias': True,
        'char-linear': True,
        'pos-linear': False,
        'char+pos-second': True,
        'char+pos-second_lowrank': False,
        'lowrank_size': 0,
        'char+pos-second_fm': False,
        'fm_size': 0,
        'fix_mode': None,
        'count_json': 'train.count.json'
    },
    'lr': 5e-5,
    'val_interval': 200,
    'num_iter': 10000,
    'use_focal': False,
    'param_focal': {
        'alpha': 0.0,
        'gamma': 0.7
    },
    'use_pos': True,
    'param_pos ': {
        'weight': 0.1,
        'pos_joint_training': True,
        'train_pos_path': 'train.pos',
        'valid_pos_path': 'dev.pos',
        'test_pos_path': 'test.pos'
    }
}


def load_config(config_path: os.PathLike, use_default: bool=False):
    config = _load_config(config_path)
    if use_default:
        for attr, val in default_config_dict.items():
            if not hasattr(config, attr):
                setattr(config, attr, val)
            elif isinstance(val, dict):
                d = getattr(config, attr)
                for dict_k, dict_v in val.items():
                    if dict_k not in d:
                        d[dict_k] = dict_v
    return config
