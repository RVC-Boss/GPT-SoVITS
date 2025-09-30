import contextlib
import gc
import os
import sys
import time
import traceback
from importlib import import_module

import torch
from rich.progress import BarColumn, Progress, TextColumn

from ..logger import SpeedColumnToken, console, logger
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

        self.device = device
        self.dtype = dtype

        self.decoder_model: T2SDecoderABC = decoder_model.to(self.device, self.dtype)
        self.decoder_model.compile()

        self.graphcache: CUDAGraphCacheABC = self.init_cache()

    def _handle_request(self, request: T2SRequest):
        with self.device:
            decoder = self.decoder_model
            session = T2SSession(decoder, request, device=self.device, dtype=self.dtype)
            batch_idx = torch.arange(session.bsz)

            t1 = 0.0
            infer_speed = 0.0
            infer_time = 0.0

            torch_profiler = TorchProfiler(request.debug)
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
                max_token = int(min(2000 - session.input_pos.max(), 1500))
                task = progress.add_task("T2S Decoding", total=max_token)

                for idx in range(max_token):
                    progress.update(task, advance=1)
                    if idx == 0:
                        session.kv_cache = decoder.init_cache(session.bsz)
                        xy_dec = decoder.h.prefill(session.xy_pos, session.kv_cache, session.attn_mask)
                        xy_dec = xy_dec[None, batch_idx, session.input_pos - 1]
                    else:
                        if (
                            request.use_cuda_graph
                            and session.graph is None
                            and self.graphcache.is_applicable
                            and torch.cuda.is_available()
                        ):
                            self.graphcache.assign_graph(session)

                        with torch_profiler.record("AR"):
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

                    with torch.cuda.stream(session.stream) if session.stream is not None else contextlib.nullcontext():
                        decoder.post_forward(idx, session)
                        logits = decoder.ar_predict_layer(xy_dec[:, -1])

                        if idx == 0:
                            logits[:, -1] = float("-inf")

                        with torch_profiler.record("Sampling"):
                            samples = session.sample(
                                logits=logits,
                                previous_tokens=session.y[:, : session.y_len + idx],
                                top_k=request.top_k,
                                top_p=request.top_p,
                                repetition_penalty=request.repetition_penalty,
                                temperature=request.temperature,
                            )
                            session.y[batch_idx, session.y_len + idx] = samples
                            session.input_pos.add_(1)

                        with torch_profiler.record("EOS"):
                            argmax_token = torch.argmax(logits, dim=-1)
                            sample_token = samples.squeeze(1)
                            EOS_mask = (argmax_token == decoder.EOS) | (sample_token == decoder.EOS)

                            newly_done_mask = EOS_mask & (~session.completed)
                            newly_done_indices = newly_done_mask.nonzero()

                            if newly_done_indices.numel() > 0:
                                for i in newly_done_indices:
                                    session.y_results[i] = session.y[i, session.y_len : session.y_len + idx]
                                    session.completed[newly_done_indices] = True

                            if torch.all(session.completed).item():
                                if session.y[:, session.y_len :].sum() == 0:
                                    session.y_results = [torch.tensor(0) for _ in range(session.bsz)]
                                    logger.error("Bad Zero Prediction")
                                else:
                                    logger.info(
                                        f"T2S Decoding EOS {session.prefill_len.tolist().__str__().strip('[]')} -> {[i.size(-1) for i in session.y_results].__str__().strip('[]')}"
                                    )
                                    logger.info(f"Infer Speed: {(idx - 1) / (time.perf_counter() - t1):.2f} token/s")
                                    infer_time = time.perf_counter() - t1
                                    infer_speed = (idx - 1) / infer_time
                                break

                            if (request.early_stop_num != -1 and idx >= request.early_stop_num) or idx == max_token - 1:
                                for i in range(session.bsz):
                                    if not session.completed[i].item():
                                        session.y_results[i] = session.y[i, session.y_len : session.y_len + 1499]
                                        session.completed[i] = True
                                    logger.error("Bad Full Prediction")
                                break

                        with torch_profiler.record("NextPos"):
                            y_emb = decoder.ar_audio_embedding(samples)
                            session.xy_pos = decoder.ar_audio_position(session.input_pos - session.x_lens, y_emb)

                        if idx == 1:
                            torch_profiler.start()
                            t1 = time.perf_counter()

                        if idx == 51:
                            torch_profiler.end()

                        if idx % 100 == 0:
                            match session.device.type:
                                case "cuda":
                                    torch.cuda.empty_cache()
                                case "mps":
                                    torch.mps.empty_cache()
                                case "xpu":
                                    torch.xpu.empty_cache()
                                case "mtia":
                                    torch.mtia.empty_cache()

            match session.device.type:
                case "cuda":
                    if session.stream is not None:
                        torch.cuda.current_stream().wait_stream(session.stream)
                    torch.cuda.empty_cache()
                case "mps":
                    torch.mps.empty_cache()
                case "xpu":
                    torch.xpu.empty_cache()
                case "mtia":
                    torch.mtia.empty_cache()
                case "cpu":
                    gc.collect()

            torch_profiler.end()
            if request.use_cuda_graph and self.graphcache.is_applicable:
                self.graphcache.release_graph(session)

            return session.y_results[: request.valid_length], infer_speed, infer_time

    def generate(self, request: T2SRequest):
        try:
            result, infer_speed, infer_time = self._handle_request(request)
            t2s_result = T2SResult(result=result, infer_speed=(infer_speed, infer_time), status="Success")
        except Exception as e:
            t2s_result = T2SResult(status="Error", exception=e, traceback=traceback.format_exc())
        if self.decoder_model.compiled:
            self.decoder_model.save_compile_cache()
            self.compiled = None
        return t2s_result

    @staticmethod
    def load_decoder(weights_path: os.PathLike, max_batch_size: int = 1, backend: str = "Flash-Attn-Varlen-CUDAGraph"):
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

        return decoder.eval()

    def init_cache(self):
        assert self.decoder_model

        module_name = self.decoder_model.__class__.__module__
        module = sys.modules.get(module_name)
        assert module

        target_class: type[CUDAGraphCacheABC] = getattr(module, "CUDAGraphCache")

        return target_class(self.decoder_model)
