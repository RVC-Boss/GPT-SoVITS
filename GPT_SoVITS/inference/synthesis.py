"""Text-to-audio synthesis behavior shared by the public inference pipeline."""

from __future__ import annotations

import gc
import math
import os
import time
import traceback
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from GPT_SoVITS.Accel.factory import create_acceleration
from GPT_SoVITS.inference.audio import denorm_spec, mel_fn, mel_fn_v4, norm_spec, resample
from GPT_SoVITS.inference.request import SynthesisRequest, set_seed
from GPT_SoVITS.module.models_v5 import V5_VERSIONS, synthesize_v5_mel
from GPT_SoVITS.runtime.i18n import I18nAuto
from .segmentation import splits


language = os.environ.get("language", "Auto")
i18n = I18nAuto(language=language)


class NO_PROMPT_ERROR(Exception):
    pass


class SynthesisMixin:
    @torch.no_grad()
    def run(self, inputs: dict):
        """
        Text to speech inference.

        Args:
            inputs (dict):
                {
                    "text": "",                   # str.(required) text to be synthesized
                    "text_lang: "",               # str.(required) language of the text to be synthesized
                    "ref_audio_path": "",         # str.(required) reference audio path
                    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                    "prompt_text": "",            # str.(optional) prompt text for the reference audio
                    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                    "top_k": 15,                  # int. top k sampling
                    "top_p": 1,                   # float. top p sampling
                    "temperature": 1,             # float. temperature for sampling
                    "text_split_method": "cut1",  # str. text split method, see text_segmentation_method.py for details.
                    "batch_size": 1,              # int. batch size for inference
                    "batch_threshold": 0.75,      # float. threshold for batch splitting.
                    "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
                    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                    "seed": -1,                   # int. random seed for reproducibility.
                    "parallel_infer": True,       # bool. whether to use parallel inference.
                    "repetition_penalty": 1.35,   # float. repetition penalty for T2S model.
                    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                    "super_sampling": False,      # bool. whether to use super-sampling for audio when using VITS model V3.
                    "return_fragment": False,     # bool. step by step return the audio fragment. (Best Quality, Slowest response speed. old version of streaming mode)
                    "streaming_mode": False,      # bool. return audio chunk by chunk. (Medium quality, Slow response speed)
                    "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
                    "min_chunk_length": 16,        # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
                    "fixed_length_chunk": False,  # bool. When turned on, it can achieve faster streaming response, but with lower quality. (lower quality, faster response speed)
                }
        returns:
            Tuple[int, np.ndarray]: sampling rate and audio data.
        """
        request = SynthesisRequest.from_mapping(inputs, self.configs.version)
        inputs = {**inputs, **request.as_dict()}
        ########## variables initialization ###########
        self.stop_flag: bool = False
        text: str = inputs.get("text", "")
        text_lang: str = inputs.get("text_lang", "")
        ref_audio_path: str = inputs.get("ref_audio_path", "")
        aux_ref_audio_paths: list = inputs.get("aux_ref_audio_paths", [])
        prompt_text: str = inputs.get("prompt_text", "")
        prompt_lang: str = inputs.get("prompt_lang", "")
        top_k: int = inputs.get("top_k", 15)
        top_p: float = inputs.get("top_p", 1)
        temperature: float = inputs.get("temperature", 1)
        text_split_method: str = inputs.get("text_split_method", "cut1")
        batch_size = inputs.get("batch_size", 1)
        batch_threshold = inputs.get("batch_threshold", 0.75)
        speed_factor = inputs.get("speed_factor", 1.0)
        split_bucket = inputs.get("split_bucket", True)
        return_fragment = inputs.get("return_fragment", False)
        fragment_interval = inputs.get("fragment_interval", 0.3)
        seed = inputs.get("seed", -1)
        seed = -1 if seed in ["", None] else seed
        actual_seed = set_seed(seed)
        parallel_infer = inputs.get("parallel_infer", True)
        use_cuda_graph = inputs.get("use_cuda_graph", True)
        use_flash_attention = inputs.get("use_flash_attention", True)
        repetition_penalty = inputs.get("repetition_penalty", 1.35)
        sample_steps = inputs.get("sample_steps")
        cfg_rate = inputs.get("cfg_rate")
        sample_steps = (4 if self.configs.version == "v5turbo" else 32) if sample_steps is None else int(sample_steps)
        cfg_rate = (1.30 if self.configs.version == "v5dev" else 0.0) if cfg_rate is None else float(cfg_rate)
        super_sampling = inputs.get("super_sampling", False)
        streaming_mode = inputs.get("streaming_mode", False)
        overlap_length = inputs.get("overlap_length", 2)
        min_chunk_length = inputs.get("min_chunk_length", 16)
        fixed_length_chunk = inputs.get("fixed_length_chunk", False)
        chunk_split_thershold = 0.0 # 该值代表语义token与mute token的余弦相似度阈值，若大于该阈值，则视为可切分点。

        gpt_streaming = inputs.get("gpt_streaming", False)
        if gpt_streaming:
            from .streaming import (validate_streaming_options, PipelineStop, semantic_updates,
                                    v2_token_chunks, v5_reference, stream_v5_windows,
                                    StreamingWindowSchedule)
            validate_streaming_options(inputs, self.configs.version)
            stream_stop = PipelineStop(self)
            stream_options = dict(inputs, sample_steps=sample_steps, cfg_rate=cfg_rate,
                speed_factor=speed_factor,
                streaming_chunk_seconds=inputs.get("streaming_chunk_seconds", 2.0),
                streaming_lookahead_seconds=inputs.get("streaming_lookahead_seconds", 0.12),
                streaming_overlap_seconds=inputs.get("streaming_overlap_seconds", 0.04),
                gpt_streaming_lookahead_seconds=inputs.get("gpt_streaming_lookahead_seconds", 0.5))
            acoustic_rng = torch.Generator(device=self.configs.device).manual_seed(actual_seed)
            # Reference semantic codes depend on the active SoVITS model.
            if getattr(self, "_streaming_ref_model", None) != id(self.vits_model):
                ref_audio_path = ref_audio_path or self.prompt_cache["ref_audio_path"]
                if not ref_audio_path:
                    raise ValueError("A reloadable reference audio is required after model changes")
                self.prompt_cache["ref_audio_path"] = None
                self._streaming_ref_model = id(self.vits_model)
            if self.prompt_cache["prompt_lang"] != prompt_lang:
                self.prompt_cache["prompt_text"] = None

        if streaming_mode and self.configs.use_vocoder and not gpt_streaming:
            print(i18n("SoVits V3/4模型不支持流式推理模式，已自动回退到分段返回模式"))
            streaming_mode = False
            return_fragment = True

        if streaming_mode:
            if parallel_infer:
                print(i18n("不支持同时开启并行推理和流式推理模式，已自动关闭并行推理模式"))
                parallel_infer = False
            else:
                print(i18n("流式推理模式已开启"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive
        elif parallel_infer:
            print(i18n("并行推理模式已开启"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_batch_infer
        else:
            print(i18n("串行推理模式已开启"))
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched

        if return_fragment and streaming_mode:
            print(i18n("流式推理模式不支持分段返回，已自动关闭分段返回"))
            return_fragment = False

        if (return_fragment or streaming_mode) and split_bucket:
            print(i18n("分段返回模式/流式推理模式不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False


        if split_bucket and speed_factor == 1.0 and not (self.configs.use_vocoder and parallel_infer):
            print(i18n("分桶处理模式已开启"))
        elif speed_factor != 1.0:
            print(i18n("语速调节不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        elif self.configs.use_vocoder and parallel_infer:
            print(i18n("当开启并行推理模式时，SoVits V3/4模型不支持分桶处理，已自动关闭分桶处理"))
            split_bucket = False
        else:
            print(i18n("分桶处理模式已关闭"))

        # if fragment_interval < 0.01:
        #     fragment_interval = 0.01
        #     print(i18n("分段间隔过小，已自动设置为0.01"))

        no_prompt_text = False
        if prompt_text in [None, ""]:
            no_prompt_text = True

        assert text_lang in self.configs.languages
        if not no_prompt_text:
            assert prompt_lang in self.configs.languages

        if no_prompt_text and self.configs.use_vocoder:
            raise NO_PROMPT_ERROR("prompt_text cannot be empty when using SoVITS_V3")

        if ref_audio_path in [None, ""] and (
            (self.prompt_cache["prompt_semantic"] is None) or (self.prompt_cache["refer_spec"] in [None, []])
        ):
            raise ValueError(
                "ref_audio_path cannot be empty, when the reference audio is not set using set_ref_audio()"
            )

        ###### setting reference audio and prompt text preprocessing ########
        t0 = time.perf_counter()
        if (ref_audio_path is not None) and (
            ref_audio_path != self.prompt_cache["ref_audio_path"]
            or (self.is_v2pro and self.prompt_cache["refer_spec"][0][1] is None)
        ):
            if not os.path.exists(ref_audio_path):
                raise ValueError(f"{ref_audio_path} not exists")
            self.set_ref_audio(ref_audio_path)

        aux_ref_audio_paths = aux_ref_audio_paths if aux_ref_audio_paths is not None else []
        paths = set(aux_ref_audio_paths) & set(self.prompt_cache["aux_ref_audio_paths"])
        if not (len(list(paths)) == len(aux_ref_audio_paths) == len(self.prompt_cache["aux_ref_audio_paths"])):
            self.prompt_cache["aux_ref_audio_paths"] = aux_ref_audio_paths
            self.prompt_cache["refer_spec"] = [self.prompt_cache["refer_spec"][0]]
            for path in aux_ref_audio_paths:
                if path in [None, ""]:
                    continue
                if not os.path.exists(path):
                    print(i18n("音频文件不存在，跳过："), path)
                    continue
                self.prompt_cache["refer_spec"].append(self._get_ref_spec(path))

        if not no_prompt_text:
            prompt_text = prompt_text.strip("\n")
            if prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_lang != "en" else "."
            print(i18n("实际输入的参考文本:"), prompt_text)
            if self.prompt_cache["prompt_text"] != prompt_text:
                phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(
                    prompt_text, prompt_lang, self.configs.version
                )
                self.prompt_cache["prompt_text"] = prompt_text
                self.prompt_cache["prompt_lang"] = prompt_lang
                self.prompt_cache["phones"] = phones
                self.prompt_cache["bert_features"] = bert_features
                self.prompt_cache["norm_text"] = norm_text

        ###### text preprocessing ########
        t1 = time.perf_counter()
        data: list = None
        if not (return_fragment or streaming_mode):
            data = self.text_preprocessor.preprocess(text, text_lang, text_split_method, self.configs.version)
            if len(data) == 0:
                yield 16000, np.zeros(int(16000), dtype=np.int16)
                return

            batch_index_list: list = None
            data, batch_index_list = self.to_batch(
                data,
                prompt_data=self.prompt_cache if not no_prompt_text else None,
                batch_size=batch_size,
                threshold=batch_threshold,
                split_bucket=split_bucket,
                device=self.configs.device,
                precision=self.precision,
            )
        else:
            print(f"############ {i18n('切分文本')} ############")
            # Preserve cut0's single-session behavior; other methods use the
            # frontend's normal punctuation/newline and long-text handling.
            texts = self.text_preprocessor.pre_seg_text(
                text.replace("\n", " ") if gpt_streaming and text_split_method == "cut0" else text,
                text_lang,
                text_split_method,
            )
            if gpt_streaming and text_split_method == "cut0" and len(texts) > 1:
                raise ValueError("Text exceeds the single-session frontend limit; shorten the input")
            data = []
            for i in range(len(texts)):
                if i % batch_size == 0:
                    data.append([])
                data[-1].append(texts[i])

            def make_batch(batch_texts):
                batch_data = []
                print(f"############ {i18n('提取文本Bert特征')} ############")
                for text in tqdm(batch_texts):
                    phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(
                        text, text_lang, self.configs.version
                    )
                    if phones is None:
                        continue
                    res = {
                        "phones": phones,
                        "bert_features": bert_features,
                        "norm_text": norm_text,
                    }
                    batch_data.append(res)
                if len(batch_data) == 0:
                    return None
                batch, _ = self.to_batch(
                    batch_data,
                    prompt_data=self.prompt_cache if not no_prompt_text else None,
                    batch_size=batch_size,
                    threshold=batch_threshold,
                    split_bucket=False,
                    device=self.configs.device,
                    precision=self.precision,
                )
                return batch[0]

        t2 = time.perf_counter()
        streaming_tokens = streaming_audio = semantic_token_generator = None
        streaming_reference = None
        try:
            print("############ 推理 ############")
            ###### inference ######
            t_34 = 0.0
            t_45 = 0.0
            audio = []
            is_first_package = True
            output_sr = self.configs.sampling_rate if not self.configs.use_vocoder else self.vocoder_configs["sr"]
            for item in data:
                if gpt_streaming and stream_stop.is_set():
                    break
                t3 = time.perf_counter()
                if return_fragment or streaming_mode:
                    item = make_batch(item)
                    if item is None:
                        continue

                batch_phones: List[torch.LongTensor] = item["phones"]
                # batch_phones:torch.LongTensor = item["phones"]
                batch_phones_len: torch.LongTensor = item["phones_len"]
                all_phoneme_ids: torch.LongTensor = item["all_phones"]
                all_phoneme_lens: torch.LongTensor = item["all_phones_len"]
                all_bert_features: torch.LongTensor = item["all_bert_features"]
                norm_text: str = item["norm_text"]
                max_len = item["max_len"]

                print(i18n("前端处理后的文本(每句):"), norm_text)
                if no_prompt_text:
                    prompt = None
                else:
                    prompt = (
                        self.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(self.configs.device)
                    )

                refer_audio_spec = []

                sv_emb = [] if self.is_v2pro else None
                for spec, audio_tensor in self.prompt_cache["refer_spec"]:
                    spec = spec.to(dtype=self.precision, device=self.configs.device)
                    refer_audio_spec.append(spec)
                    if self.is_v2pro:
                        sv_emb.append(self.sv_model.compute_embedding3(audio_tensor))

                if not streaming_mode:
                    print(f"############ {i18n('预测语义Token')} ############")
                    accel_batch_size = batch_size if parallel_infer else 1
                    if self.t2s_accel is not None and self.t2s_accel.max_batch_size != accel_batch_size:
                        self.t2s_accel.close()
                        self.t2s_accel = None
                    if self.t2s_accel is None:
                        self.t2s_accel = create_acceleration(
                            self.configs.t2s_weights_path,
                            device=torch.device(self.configs.device),
                            dtype=self.precision,
                            max_batch_size=accel_batch_size,
                            use_cuda_graph=use_cuda_graph,
                            use_flash_attention=use_flash_attention,
                        )
                    use_accel = self.t2s_accel is not None and self.t2s_accel.prepare(
                        use_cuda_graph, use_flash_attention
                    )
                    if use_accel:
                        pred_semantic_list, idx_list = [], []
                        for start in range(0, len(all_phoneme_ids), accel_batch_size):
                            end = start + accel_batch_size
                            tokens, token_lengths = self.t2s_accel.infer_batch(
                                all_phoneme_ids[start:end],
                                all_phoneme_lens[start:end],
                                prompt[start:end] if prompt is not None else None,
                                all_bert_features[start:end],
                                parallel_infer=parallel_infer,
                                use_cuda_graph=use_cuda_graph,
                                use_flash_attention=use_flash_attention,
                                top_k=top_k,
                                top_p=top_p,
                                temperature=temperature,
                                early_stop_num=self.configs.hz * self.configs.max_sec,
                                repetition_penalty=repetition_penalty,
                            )
                            pred_semantic_list.extend(tokens)
                            idx_list.extend(token_lengths)
                    else:
                        pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
                            all_phoneme_ids,
                            all_phoneme_lens,
                            prompt,
                            all_bert_features,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            early_stop_num=self.configs.hz * self.configs.max_sec,
                            max_len=max_len,
                            repetition_penalty=repetition_penalty,
                        )
                    t4 = time.perf_counter()
                    t_34 += t4 - t3


                    batch_audio_fragment = []

                    # ## vits并行推理 method 1
                    # pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                    # pred_semantic_len = torch.LongTensor([item.shape[0] for item in pred_semantic_list]).to(self.configs.device)
                    # pred_semantic = self.batch_sequences(pred_semantic_list, axis=0, pad_value=0).unsqueeze(0)
                    # max_len = 0
                    # for i in range(0, len(batch_phones)):
                    #     max_len = max(max_len, batch_phones[i].shape[-1])
                    # batch_phones = self.batch_sequences(batch_phones, axis=0, pad_value=0, max_length=max_len)
                    # batch_phones = batch_phones.to(self.configs.device)
                    # batch_audio_fragment = (self.vits_model.batched_decode(
                    #         pred_semantic, pred_semantic_len, batch_phones, batch_phones_len,refer_audio_spec
                    #     ))
                    print(f"############ {i18n('合成音频')} ############")
                    if not self.configs.use_vocoder:
                        if speed_factor == 1.0 and parallel_infer:
                            print(f"{i18n('并行合成中')}...")
                            # ## vits并行推理 method 2
                            pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]
                            upsample_rate = math.prod(self.vits_model.upsample_rates)
                            audio_frag_idx = [
                                pred_semantic_list[i].shape[0] * 2 * upsample_rate
                                for i in range(0, len(pred_semantic_list))
                            ]
                            audio_frag_end_idx = [sum(audio_frag_idx[: i + 1]) for i in range(0, len(audio_frag_idx))]
                            all_pred_semantic = (
                                torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(0).to(self.configs.device)
                            )
                            _batch_phones = torch.cat(batch_phones).unsqueeze(0).to(self.configs.device)

                            _batch_audio_fragment = self.vits_model.decode(
                                    all_pred_semantic, _batch_phones, refer_audio_spec, speed=speed_factor, sv_emb=sv_emb
                                ).detach()[0, 0, :]

                            audio_frag_end_idx.insert(0, 0)
                            batch_audio_fragment = [
                                _batch_audio_fragment[audio_frag_end_idx[i - 1] : audio_frag_end_idx[i]]
                                for i in range(1, len(audio_frag_end_idx))
                            ]
                        else:
                            # ## vits串行推理
                            for i, idx in enumerate(tqdm(idx_list)):
                                phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
                                _pred_semantic = (
                                    pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0)
                                )  # .unsqueeze(0)#mq要多unsqueeze一次
                                audio_fragment = self.vits_model.decode(
                                        _pred_semantic, phones, refer_audio_spec, speed=speed_factor, sv_emb=sv_emb
                                    ).detach()[0, 0, :]
                                batch_audio_fragment.append(audio_fragment)  ###试试重建不带上prompt部分
                    else:
                        if parallel_infer:
                            print(f"{i18n('并行合成中')}...")
                            audio_fragments = self.using_vocoder_synthesis_batched_infer(
                                idx_list, pred_semantic_list, batch_phones, speed=speed_factor,
                                sample_steps=sample_steps, cfg_rate=cfg_rate
                            )
                            batch_audio_fragment.extend(audio_fragments)
                        else:
                            for i, idx in enumerate(tqdm(idx_list)):
                                phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
                                _pred_semantic = (
                                    pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0)
                                )  # .unsqueeze(0)#mq要多unsqueeze一次
                                audio_fragment = self.using_vocoder_synthesis(
                                    _pred_semantic, phones, speed=speed_factor,
                                    sample_steps=sample_steps, cfg_rate=cfg_rate
                                )
                                batch_audio_fragment.append(audio_fragment)

                elif gpt_streaming and self.configs.version in V5_VERSIONS:
                    if streaming_reference is None:
                        streaming_reference = v5_reference(self)
                    reference = streaming_reference
                    # Validate the acoustic context before starting the AR owner thread.
                    frames = round(stream_options["streaming_chunk_seconds"] * 100)
                    future = round(stream_options["streaming_lookahead_seconds"] * 100)
                    if reference[6].shape[-1] + frames + future > 1000:
                        raise ValueError("Reference plus streaming window exceeds the V5 context")
                    streaming_tokens = semantic_updates(self, all_phoneme_ids[0], prompt,
                        all_bert_features[0], stream_options, stream_stop)
                    streaming_audio = stream_v5_windows(self.vits_model, self.vocoder, denorm_spec,
                        self.configs.device, streaming_tokens, batch_phones[0], reference,
                        stream_options, stream_stop)
                    for pcm in streaming_audio:
                        yield output_sr, pcm
                    streaming_audio.close()
                    streaming_tokens.close()
                    streaming_audio = streaming_tokens = None
                    if not stream_stop.is_set() and fragment_interval:
                        yield output_sr, np.zeros(int(output_sr * fragment_interval), dtype=np.int16)
                    t4 = time.perf_counter()
                else:
                    # refer_audio_spec: torch.Tensor = [
                    #     item.to(dtype=self.precision, device=self.configs.device)
                    #     for item in self.prompt_cache["refer_spec"]
                    # ]
                    window_schedule = None
                    if gpt_streaming:
                        if inputs.get("streaming_window_mode", "fixed") == "hybrid":
                            window_schedule = StreamingWindowSchedule(min_chunk_length, min_chunk_length * 4)
                        streaming_tokens = semantic_updates(self, all_phoneme_ids[0], prompt,
                            all_bert_features[0], stream_options, stream_stop)
                        semantic_token_generator = v2_token_chunks(streaming_tokens, min_chunk_length,
                            self.configs.mute_emb_sim_matrix if not fixed_length_chunk else None,
                            chunk_split_thershold, schedule=window_schedule)
                    else:
                        semantic_token_generator =self.t2s_model.model.infer_panel(
                            all_phoneme_ids[0].unsqueeze(0),
                            all_phoneme_lens,
                            prompt,
                            all_bert_features[0].unsqueeze(0),
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            early_stop_num=self.configs.hz * self.configs.max_sec,
                            max_len=max_len,
                            repetition_penalty=repetition_penalty,
                            streaming_mode=True,
                            chunk_length=min_chunk_length,
                            mute_emb_sim_matrix=self.configs.mute_emb_sim_matrix if not fixed_length_chunk else None,
                            chunk_split_thershold=chunk_split_thershold,
                        )
                    t4 = time.perf_counter()
                    t_34 += t4 - t3
                    phones = batch_phones[0].unsqueeze(0).to(self.configs.device)
                    is_first_chunk = True

                    if not self.configs.use_vocoder:
                        # if speed_factor == 1.0:
                        #     upsample_rate = math.prod(self.vits_model.upsample_rates)*(2 if self.vits_model.semantic_frame_rate == "25hz" else 1)
                        # else:
                        upsample_rate = math.prod(self.vits_model.upsample_rates)*((2 if self.vits_model.semantic_frame_rate == "25hz" else 1)/speed_factor)
                    else:
                        # if speed_factor == 1.0:
                        #     upsample_rate = self.vocoder_configs["upsample_rate"]*(3.875 if self.configs.version == "v3" else 4)
                        # else:
                        upsample_rate = self.vocoder_configs["upsample_rate"]*((3.875 if self.configs.version == "v3" else 4)/speed_factor)

                    last_audio_chunk = None
                    # last_tokens = None
                    last_latent = None
                    previous_tokens = []
                    overlap_len = overlap_length
                    overlap_size = math.ceil(overlap_length*upsample_rate)
                    for semantic_tokens, is_final in semantic_token_generator:
                        if semantic_tokens is None and last_audio_chunk is not None:
                            yield self.audio_postprocess(
                                    [[last_audio_chunk[-overlap_size:]]],
                                    output_sr,
                                    None,
                                    speed_factor,
                                    False,
                                    0.0,
                                    super_sampling if self.configs.use_vocoder and self.configs.version == "v3" else False,
                                )
                            break

                        _semantic_tokens = semantic_tokens
                        print(f"semantic_tokens shape:{semantic_tokens.shape}")

                        previous_tokens.append(semantic_tokens)

                        _semantic_tokens = torch.cat(previous_tokens, dim=-1)

                        if not is_first_chunk and semantic_tokens.shape[-1] < 10:
                            overlap_len = overlap_length+(10-semantic_tokens.shape[-1])
                        else:
                            overlap_len = overlap_length


                        if not self.configs.use_vocoder:
                            token_padding_length = 0
                            # token_padding_length = int(phones.shape[-1]*2)-_semantic_tokens.shape[-1]
                            # if token_padding_length>0:
                            #     _semantic_tokens = F.pad(_semantic_tokens, (0, token_padding_length), "constant", 486)
                            # else:
                            #     token_padding_length = 0

                            audio_chunk, latent, latent_mask = self.vits_model.decode_streaming(
                                                    _semantic_tokens.unsqueeze(0), 
                                                    phones, refer_audio_spec, 
                                                    speed=speed_factor,
                                                    sv_emb=sv_emb,
                                                    result_length=semantic_tokens.shape[-1]+overlap_len if not is_first_chunk else None,
                                                    overlap_frames=last_latent[:,:,-overlap_len*(2 if self.vits_model.semantic_frame_rate == "25hz" else 1):] \
                                                    if last_latent is not None else None,
                                                    padding_length=token_padding_length,
                                                    **({"generator": acoustic_rng} if gpt_streaming else {})
                                                )
                            audio_chunk=audio_chunk.detach()[0, 0, :]
                        else:
                            raise RuntimeError(i18n("SoVits V3/4模型不支持流式推理模式"))

                        if overlap_len>overlap_length:
                            audio_chunk=audio_chunk[-int((overlap_length+semantic_tokens.shape[-1])*upsample_rate):]

                        audio_chunk_ = audio_chunk
                        if is_first_chunk and not is_final:
                            is_first_chunk = False
                            audio_chunk_ = audio_chunk_[:-overlap_size]
                        elif is_first_chunk and is_final: 
                            is_first_chunk = False
                        elif not is_first_chunk and (not is_final or gpt_streaming):
                            audio_chunk_ = self.sola_algorithm([last_audio_chunk, audio_chunk_], overlap_size)
                            audio_chunk_ = (
                                audio_chunk_[last_audio_chunk.shape[0]-overlap_size:-overlap_size] if not is_final \
                                    else audio_chunk_[last_audio_chunk.shape[0]-overlap_size:]
                                    )

                        last_latent = latent
                        last_audio_chunk = audio_chunk
                        audio_result = self.audio_postprocess(
                                [[audio_chunk_]],
                                output_sr,
                                None,
                                speed_factor,
                                False,
                                0.0,
                                super_sampling if self.configs.use_vocoder and self.configs.version == "v3" else False,
                            )
                        if window_schedule is not None:
                            window_schedule.observe(semantic_tokens.numel(), audio_result[1].size, output_sr)
                        yield audio_result

                        if is_first_package: 
                            print(f"first_package_delay: {time.perf_counter()-t0:.3f}")
                            is_first_package = False


                    if gpt_streaming:
                        semantic_token_generator.close()
                        streaming_tokens.close()
                        semantic_token_generator = streaming_tokens = None
                    if not gpt_streaming or not stream_stop.is_set():
                        yield output_sr, np.zeros(int(output_sr*fragment_interval), dtype=np.int16)

                t5 = time.perf_counter()
                t_45 += t5 - t4
                if return_fragment:
                    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t4 - t3, t5 - t4))
                    yield self.audio_postprocess(
                        [batch_audio_fragment],
                        output_sr,
                        None,
                        speed_factor,
                        False,
                        fragment_interval,
                        super_sampling if self.configs.use_vocoder and self.configs.version == "v3" else False,
                    )
                elif streaming_mode:...
                else:
                    audio.append(batch_audio_fragment)

                if self.stop_flag:
                    yield output_sr, np.zeros(int(output_sr), dtype=np.int16)
                    return

            if not (return_fragment or streaming_mode):
                print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t_34, t_45))
                if len(audio) == 0:
                    yield output_sr, np.zeros(int(output_sr), dtype=np.int16)
                    return
                yield self.audio_postprocess(
                    audio,
                    output_sr,
                    batch_index_list,
                    speed_factor,
                    split_bucket,
                    fragment_interval,
                    super_sampling if self.configs.use_vocoder and self.configs.version == "v3" else False,
                )

        except Exception as e:
            if gpt_streaming:
                raise
            traceback.print_exc()
            # 必须返回一个空音频, 否则会导致显存不释放。
            yield 16000, np.zeros(int(16000), dtype=np.int16)
            # 重置模型, 否则会导致显存释放不完全。
            del self.t2s_model
            del self.vits_model
            self.t2s_model = None
            self.vits_model = None
            self.init_t2s_weights(self.configs.t2s_weights_path)
            self.init_vits_weights(self.configs.vits_weights_path)
            raise e
        finally:
            if gpt_streaming:
                stream_stop.set()
                for iterator in (streaming_audio, semantic_token_generator, streaming_tokens):
                    if iterator is not None:
                        iterator.close()
            self.empty_cache()
    def empty_cache(self):
        try:
            gc.collect()  # 触发gc的垃圾回收。避免内存一直增长。
            if "cuda" in str(self.configs.device):
                torch.cuda.empty_cache()
            elif str(self.configs.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass
    def audio_postprocess(
        self,
        audio: List[torch.Tensor],
        sr: int,
        batch_index_list: list = None,
        speed_factor: float = 1.0,
        split_bucket: bool = True,
        fragment_interval: float = 0.3,
        super_sampling: bool = False,
    ) -> Tuple[int, np.ndarray]:
        if fragment_interval>0:
            zero_wav = torch.zeros(
                int(self.configs.sampling_rate * fragment_interval), dtype=self.precision, device=self.configs.device
            )

        for i, batch in enumerate(audio):
            for j, audio_fragment in enumerate(batch):
                max_audio = torch.abs(audio_fragment).max()  # 简单防止16bit爆音
                if max_audio > 1:
                    audio_fragment /= max_audio
                audio_fragment: torch.Tensor = torch.cat([audio_fragment, zero_wav], dim=0) if fragment_interval>0 else audio_fragment
                audio[i][j] = audio_fragment

        if split_bucket:
            audio = self.recovery_order(audio, batch_index_list)
        else:
            # audio = [item for batch in audio for item in batch]
            audio = sum(audio, [])

        audio = torch.cat(audio, dim=0)

        if super_sampling:
            print(f"############ {i18n('音频超采样')} ############")
            t1 = time.perf_counter()
            self.init_sr_model()
            if not self.sr_model_not_exist:
                audio, sr = self.sr_model(audio.unsqueeze(0), sr)
                max_audio = np.abs(audio).max()
                if max_audio > 1:
                    audio /= max_audio
                audio = (audio * 32768).astype(np.int16)
            else:
                audio = audio.cpu().numpy()
                audio = (audio * 32768).astype(np.int16)
            t2 = time.perf_counter()
            print(f"超采样用时：{t2 - t1:.3f}s")
        else:
            audio = audio.cpu().numpy()
            audio = (audio * 32768).astype(np.int16)


        return sr, audio
    def using_vocoder_synthesis(
        self, semantic_tokens: torch.Tensor, phones: torch.Tensor, speed: float = 1.0,
        sample_steps: int = None, cfg_rate: float = None
    ):
        sample_steps = (4 if self.configs.version == "v5turbo" else 32) if sample_steps is None else int(sample_steps)
        cfg_rate = (1.30 if self.configs.version == "v5dev" else 0.0) if cfg_rate is None else float(cfg_rate)
        prompt_semantic_tokens = self.prompt_cache["prompt_semantic"].unsqueeze(0).unsqueeze(0).to(self.configs.device)
        prompt_phones = torch.LongTensor(self.prompt_cache["phones"]).unsqueeze(0).to(self.configs.device)
        raw_entry = self.prompt_cache["refer_spec"][0]
        if isinstance(raw_entry, tuple):
            raw_entry = raw_entry[0]
        refer_audio_spec = raw_entry.to(dtype=self.precision, device=self.configs.device)

        fea_ref, ge = self.vits_model.decode_encp(prompt_semantic_tokens, prompt_phones, refer_audio_spec)
        ref_audio: torch.Tensor = self.prompt_cache["raw_audio"]
        ref_sr = self.prompt_cache["raw_sr"]
        ref_audio = ref_audio.to(self.configs.device).float()
        if ref_audio.shape[0] == 2:
            ref_audio = ref_audio.mean(0).unsqueeze(0)

        # tgt_sr = self.vocoder_configs["sr"]
        tgt_sr = 24000 if self.configs.version == "v3" else 32000
        if ref_sr != tgt_sr:
            ref_audio = resample(ref_audio, ref_sr, tgt_sr, self.configs.device)

        mel2 = mel_fn(ref_audio) if self.configs.version == "v3" else mel_fn_v4(ref_audio)
        mel2 = norm_spec(mel2)
        if self.configs.version in V5_VERSIONS:
            fea_todo, _ = self.vits_model.decode_encp(semantic_tokens, phones, refer_audio_spec, ge, speed)
            mel = synthesize_v5_mel(
                self.vits_model, fea_ref, fea_todo, mel2.to(self.precision), sample_steps, cfg_rate,
            )
            with torch.inference_mode():
                return self.vocoder(denorm_spec(mel).to(self.precision))[0, 0]
        T_min = min(mel2.shape[2], fea_ref.shape[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        T_ref = self.vocoder_configs["T_ref"]
        T_chunk = self.vocoder_configs["T_chunk"]
        if T_min > T_ref:
            mel2 = mel2[:, :, -T_ref:]
            fea_ref = fea_ref[:, :, -T_ref:]
            T_min = T_ref
        chunk_len = T_chunk - T_min

        mel2 = mel2.to(self.precision)
        fea_todo, ge = self.vits_model.decode_encp(semantic_tokens, phones, refer_audio_spec, ge, speed)

        cfm_resss = []
        idx = 0
        while 1:
            fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
            if fea_todo_chunk.shape[-1] == 0:
                break
            idx += chunk_len
            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)

            cfm_res = self.vits_model.cfm.inference(
                fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
            )
            cfm_res = cfm_res[:, :, mel2.shape[2] :]

            mel2 = cfm_res[:, :, -T_min:]
            fea_ref = fea_todo_chunk[:, :, -T_min:]

            cfm_resss.append(cfm_res)
        cfm_res = torch.cat(cfm_resss, 2)
        cfm_res = denorm_spec(cfm_res)

        with torch.inference_mode():
            wav_gen = self.vocoder(cfm_res)
            audio = wav_gen[0][0]  # .cpu().detach().numpy()

        return audio
    def using_vocoder_synthesis_batched_infer(
        self,
        idx_list: List[int],
        semantic_tokens_list: List[torch.Tensor],
        batch_phones: List[torch.Tensor],
        speed: float = 1.0,
        sample_steps: int = None,
        cfg_rate: float = None,
    ) -> List[torch.Tensor]:
        sample_steps = (4 if self.configs.version == "v5turbo" else 32) if sample_steps is None else int(sample_steps)
        cfg_rate = (1.30 if self.configs.version == "v5dev" else 0.0) if cfg_rate is None else float(cfg_rate)
        if self.configs.version in V5_VERSIONS:
            return [
                self.using_vocoder_synthesis(
                    semantic_tokens_list[i][-idx:].unsqueeze(0).unsqueeze(0),
                    batch_phones[i].unsqueeze(0).to(self.configs.device),
                    speed=speed, sample_steps=sample_steps, cfg_rate=cfg_rate,
                )
                for i, idx in enumerate(idx_list)
            ]
        prompt_semantic_tokens = self.prompt_cache["prompt_semantic"].unsqueeze(0).unsqueeze(0).to(self.configs.device)
        prompt_phones = torch.LongTensor(self.prompt_cache["phones"]).unsqueeze(0).to(self.configs.device)
        raw_entry = self.prompt_cache["refer_spec"][0]
        if isinstance(raw_entry, tuple):
            raw_entry = raw_entry[0]
        refer_audio_spec = raw_entry.to(dtype=self.precision, device=self.configs.device)

        fea_ref, ge = self.vits_model.decode_encp(prompt_semantic_tokens, prompt_phones, refer_audio_spec)
        ref_audio: torch.Tensor = self.prompt_cache["raw_audio"]
        ref_sr = self.prompt_cache["raw_sr"]
        ref_audio = ref_audio.to(self.configs.device).float()
        if ref_audio.shape[0] == 2:
            ref_audio = ref_audio.mean(0).unsqueeze(0)

        # tgt_sr = self.vocoder_configs["sr"]
        tgt_sr = 24000 if self.configs.version == "v3" else 32000
        if ref_sr != tgt_sr:
            ref_audio = resample(ref_audio, ref_sr, tgt_sr, self.configs.device)

        mel2 = mel_fn(ref_audio) if self.configs.version == "v3" else mel_fn_v4(ref_audio)
        mel2 = norm_spec(mel2)
        T_min = min(mel2.shape[2], fea_ref.shape[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        T_ref = self.vocoder_configs["T_ref"]
        T_chunk = self.vocoder_configs["T_chunk"]
        if T_min > T_ref:
            mel2 = mel2[:, :, -T_ref:]
            fea_ref = fea_ref[:, :, -T_ref:]
            T_min = T_ref
        chunk_len = T_chunk - T_min

        mel2 = mel2.to(self.precision)

        # #### batched inference
        overlapped_len = self.vocoder_configs["overlapped_len"]
        feat_chunks = []
        feat_lens = []
        feat_list = []

        for i, idx in enumerate(idx_list):
            phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
            semantic_tokens = (
                semantic_tokens_list[i][-idx:].unsqueeze(0).unsqueeze(0)
            )  # .unsqueeze(0)#mq要多unsqueeze一次
            feat, _ = self.vits_model.decode_encp(semantic_tokens, phones, refer_audio_spec, ge, speed)
            feat_list.append(feat)
            feat_lens.append(feat.shape[2])

        feats = torch.cat(feat_list, 2)
        feats_padded = F.pad(feats, (overlapped_len, 0), "constant", 0)
        pos = 0
        padding_len = 0
        while True:
            if pos == 0:
                chunk = feats_padded[:, :, pos : pos + chunk_len]
            else:
                pos = pos - overlapped_len
                chunk = feats_padded[:, :, pos : pos + chunk_len]
            pos += chunk_len
            if chunk.shape[-1] == 0:
                break

            # padding for the last chunk
            padding_len = chunk_len - chunk.shape[2]
            if padding_len != 0:
                chunk = F.pad(chunk, (0, padding_len), "constant", 0)
            feat_chunks.append(chunk)

        feat_chunks = torch.cat(feat_chunks, 0)
        bs = feat_chunks.shape[0]
        fea_ref = fea_ref.repeat(bs, 1, 1)
        fea = torch.cat([fea_ref, feat_chunks], 2).transpose(2, 1)
        pred_spec = self.vits_model.cfm.inference(
            fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0
        )
        pred_spec = pred_spec[:, :, -chunk_len:]
        dd = pred_spec.shape[1]
        pred_spec = pred_spec.permute(1, 0, 2).contiguous().view(dd, -1).unsqueeze(0)
        # pred_spec = pred_spec[..., :-padding_len]

        pred_spec = denorm_spec(pred_spec)

        with torch.no_grad():
            wav_gen = self.vocoder(pred_spec)
            audio = wav_gen[0][0]  # .cpu().detach().numpy()

        audio_fragments = []
        upsample_rate = self.vocoder_configs["upsample_rate"]
        pos = 0

        while pos < audio.shape[-1]:
            audio_fragment = audio[pos : pos + chunk_len * upsample_rate]
            audio_fragments.append(audio_fragment)
            pos += chunk_len * upsample_rate

        audio = self.sola_algorithm(audio_fragments, overlapped_len * upsample_rate)
        if padding_len > 0:
            audio = audio[overlapped_len * upsample_rate : -padding_len * upsample_rate]
        else:
            audio = audio[overlapped_len * upsample_rate :]

        audio_fragments = []
        for feat_len in feat_lens:
            audio_fragment = audio[: feat_len * upsample_rate]
            audio_fragments.append(audio_fragment)
            audio = audio[feat_len * upsample_rate :]

        return audio_fragments
    def sola_algorithm(
        self,
        audio_fragments: List[torch.Tensor],
        overlap_len: int,
        search_len:int= 320
    ):
        # overlap_len-=search_len

        dtype = audio_fragments[0].dtype

        for i in range(len(audio_fragments) - 1):
            f1 = audio_fragments[i].float()
            f2 = audio_fragments[i + 1].float()
            w1 = f1[-overlap_len:]
            w2 = f2[:overlap_len+search_len]
            # w2 = w2[-w2.shape[-1]//2:]
            # assert w1.shape == w2.shape
            corr_norm = F.conv1d(w2.view(1, 1, -1), w1.view(1, 1, -1)).view(-1)

            corr_den = F.conv1d(w2.view(1, 1, -1)**2, torch.ones_like(w1).view(1, 1, -1)).view(-1)+ 1e-8
            idx = (corr_norm/corr_den.sqrt()).argmax()

            print(f"seg_idx: {idx}")

            # idx = corr.argmax()
            f1_ = f1[: -overlap_len]
            audio_fragments[i] = f1_

            f2_ = f2[idx:]
            window = torch.hann_window((overlap_len) * 2, device=f1.device, dtype=f1.dtype)
            f2_[: overlap_len] = (
                window[: overlap_len] * f2_[: overlap_len]
                + window[overlap_len :] * f1[-overlap_len :]
            )

            # window = torch.sin(torch.arange((overlap_len - idx), device=f1.device) * np.pi / (overlap_len - idx))
            # f2_[: (overlap_len - idx)] = (
            #     window * f2_[: (overlap_len - idx)]
            #     + (1-window) * f1[-(overlap_len - idx) :]
            # )

            audio_fragments[i + 1] = f2_

        return torch.cat(audio_fragments, 0).to(dtype)
