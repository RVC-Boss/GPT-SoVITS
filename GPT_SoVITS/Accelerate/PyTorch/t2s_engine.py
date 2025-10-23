import contextlib
import gc
import os
import time
import traceback
from importlib import import_module
from typing import Literal

import torch
from rich.progress import BarColumn, Progress, TextColumn

from ..logger import SpeedColumnToken, console, logger, timer
from .structs import T2SEngineProtocol, T2SRequest, T2SResult, T2SSession
from .t2s_model_abc import (
    CUDAGraphCacheABC,
    T2SDecoderABC,
    TorchProfiler,
)


class T2SEngine(T2SEngineProtocol):
    def __init__(
        self,
        decoder_model: T2SDecoderABC,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        assert device.type in {"cpu", "cuda", "mps", "xpu", "mtia"}
        assert dtype in {torch.float16, torch.bfloat16, torch.float32}

        self.device = device if device.type != "mps" else torch.device("cpu")
        self.dtype = dtype

        self.decoder_model: T2SDecoderABC = decoder_model.to(self.device, self.dtype)
        # self.decoder_model.compile()

        self.graphcache: CUDAGraphCacheABC = decoder_model.graph_cache_class(self.decoder_model)

    def _handle_request(self, request: T2SRequest):
        with self.device:
            decoder = self.decoder_model
            session = T2SSession(decoder, request, device=self.device, dtype=self.dtype)
            batch_idx = torch.arange(session.bsz)
            debug = request.debug

            t1 = 0.0
            infer_speed = 0.0
            infer_time = 0.0
            idx = 0
            graph_state = None

            torch_profiler = TorchProfiler(debug)
            with (
                torch_profiler.profiler(),
                Progress(
                    TextColumn("[cyan]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total} tokens"),
                    SpeedColumnToken(show_speed=True),
                    console=console,
                    transient=True,
                ) as progress,
            ):
                torch_profiler.start()
                max_token = min(int(1500 - session.input_pos.max()), 1000) * session.bsz
                task = progress.add_task("T2S Decoding", total=max_token)

                try:
                    for idx in range(max_token):
                        progress.update(task, advance=session.bsz)
                        if idx == 0:
                            with torch_profiler.record("Prefill"), timer("Torch.Prefill", debug=debug):
                                session.kv_cache = decoder.init_cache(session.bsz)
                                t1 = time.perf_counter()
                                xy_dec = decoder.h.prefill(session.xy_pos, session.kv_cache, session.attn_mask)
                                xy_dec = xy_dec[batch_idx, None, session.input_pos - 1]
                        else:
                            if (
                                idx == 1
                                and request.use_cuda_graph
                                and self.graphcache.is_applicable
                                and torch.cuda.is_available()
                                and torch.version.cuda is not None
                                and os.environ.get("CUDAGraph", "1") != "0"
                            ):
                                graph_state = self.graphcache[session.bsz].assign_graph(session)

                            with torch_profiler.record("Decode"), timer("Torch.Decode", debug=debug):
                                if session.graph:
                                    assert session.stream
                                    session.stream.wait_stream(torch.cuda.default_stream())
                                    with torch.cuda.stream(session.stream):
                                        session.xy_pos_.copy_(session.xy_pos)
                                        session.graph.replay()
                                        xy_dec = session.xy_dec_.clone()
                                else:
                                    args, kwds = decoder.pre_forward(session)
                                    xy_dec = decoder.h(
                                        session.input_pos,
                                        session.xy_pos,
                                        session.kv_cache,
                                        *args,
                                        **kwds,
                                    )

                        with (
                            torch.cuda.stream(session.stream)
                            if session.stream is not None
                            else contextlib.nullcontext()
                        ):
                            decoder.post_forward(idx, session)
                            logits = decoder.ar_predict_layer(xy_dec.squeeze(1))

                            if idx == 0:
                                logits[:, -1] = float("-inf")

                            with torch_profiler.record("Sampling"), timer("Torch.Sampling", debug=debug):
                                samples = session.sample(
                                    logits=logits,
                                    previous_tokens=session.y[:, : session.y_len + idx],
                                    top_k=request.top_k,
                                    top_p=request.top_p,
                                    repetition_penalty=request.repetition_penalty,
                                    temperature=request.temperature,
                                )
                                session.y[batch_idx.reshape(-1, 1), session.y_len + idx] = samples
                                session.input_pos.add_(1)

                            with torch_profiler.record("EOS"), timer("Torch.EOS", debug=debug):
                                argmax_token = torch.argmax(logits, dim=-1)
                                sample_token = samples.squeeze(1)
                                EOS_mask = (argmax_token == decoder.EOS) | (sample_token == decoder.EOS)

                                newly_done_mask = EOS_mask & (~session.completed)
                                newly_done_indices = newly_done_mask.nonzero()

                                if newly_done_indices.numel() > 0:
                                    for i in newly_done_indices:
                                        session.y_results[i] = session.y[
                                            i, session.y_len : session.y_len + idx
                                        ].squeeze(0)
                                        session.completed[newly_done_indices] = True

                            if torch.all(session.completed).item():
                                logger.info(
                                    f"T2S Decoding EOS {session.prefill_len.tolist().__str__().strip('[]')} -> {[i.size(-1) for i in session.y_results].__str__().strip('[]')}"
                                )
                                logger.info(
                                    f"Infer Speed: {(idx + 1) * session.bsz / (time.perf_counter() - t1):.2f} token/s"
                                )
                                infer_time = time.perf_counter() - t1
                                infer_speed = (idx + 1) * session.bsz / infer_time
                                break

                            if (request.early_stop_num != -1 and idx >= request.early_stop_num) or idx == max_token - 1:
                                for i in range(session.bsz):
                                    if not session.completed[i].item():
                                        session.y_results[i] = session.y[
                                            [i], session.y_len : session.y_len + idx
                                        ].squeeze(0)
                                        session.completed[i] = True
                                    logger.error("Bad Full Prediction")
                                    infer_time = time.perf_counter() - t1
                                    infer_speed = (idx + 1) * session.bsz / infer_time
                                break

                            with torch_profiler.record("NextPos"), timer("Torch.NextPos", debug=debug):
                                y_emb = decoder.ar_audio_embedding(samples)
                                session.xy_pos = decoder.ar_audio_position(session.input_pos - session.x_lens, y_emb)

                            if idx == 10:
                                torch_profiler.end()
                finally:
                    if (
                        request.use_cuda_graph
                        and self.graphcache.is_applicable
                        and torch.cuda.is_available()
                        and torch.version.cuda is not None
                        and os.environ.get("CUDAGraph", "1") != "0"
                    ):
                        self.graphcache.release_graph(graph_state)

                        match decoder.device.type:
                            case "cuda":
                                torch.cuda.empty_cache()
                            case "mps":
                                torch.mps.empty_cache()
                            case "xpu":
                                torch.xpu.empty_cache()
                            case "mtia":
                                torch.mtia.empty_cache()
                            case "cpu":
                                gc.collect(1)

            return session.y_results[: request.valid_length], infer_speed, infer_time, (idx + 1) * session.bsz

    def generate(self, request: T2SRequest):
        try:
            result, infer_speed, infer_time, total_tokens = self._handle_request(request)
            t2s_result = T2SResult(
                result=result,
                infer_speed=(infer_speed, infer_time),
                total_tokens=total_tokens,
                status="Success",
            )
        except Exception as e:
            t2s_result = T2SResult(status="Error", exception=e, traceback=traceback.format_exc())
        if self.decoder_model.compiled:
            self.decoder_model.save_compile_cache()
            self.compiled = None
        return t2s_result

    @staticmethod
    def load_decoder(
        weights_path: os.PathLike,
        max_batch_size: int = 1,
        backend: str = "Flash-Attn-Varlen-CUDAGraph",
        quantize_mode: Literal["Int8", "FP8", "FP8_E4M3FN"] | None = None,
    ) -> T2SDecoderABC:
        logger.info(f"Loading Text2Semantic Weights from {weights_path} with {backend} Backend")
        module_path = f".backends.{backend.lower().replace('-', '_').replace('cudagraph', 'cuda_graph')}"
        decoder_cls_name = "T2SDecoder"
        decoder_mod = import_module(module_path, package=__package__)
        decoder_cls: type[T2SDecoderABC] = getattr(decoder_mod, decoder_cls_name)
        dict_s1 = torch.load(weights_path, map_location="cpu", weights_only=False, mmap=True)
        config = dict_s1["config"]
        decoder: T2SDecoderABC = decoder_cls(config, max_batch_size=max_batch_size)
        state_dict = dict_s1["weight"]
        decoder.load_state_dict(state_dict)

        if quantize_mode is not None:
            decoder.quantize(quantize_mode)
            logger.info(f"Quantized by {quantize_mode} Quantization")

        return decoder.eval()
