from __future__ import annotations

import os
import threading
import time
import traceback
from collections import deque
from importlib import import_module
from queue import Empty, SimpleQueue
from typing import Literal, Optional, Union

import mlx.core as mx
import torch
from rich.progress import BarColumn, Progress, TextColumn

from ...logger import SpeedColumnToken, Timer, console, logger

from .runner_mlx import ModelRunner
from .structs_mlx import (
    T2SEngineProtocol,
    T2SRequest,
    T2SRequestHandle,
    T2SResult,
    T2SSessionMLX as T2SSession,
    T2SStreamResponse,
)
from .t2s_model_abc_mlx import T2SDecoderABC


timer = Timer("T2S Engine MLX")

cpu = mx.Device(mx.cpu)

empty_result = T2SResult(
    result=None,
    infer_speed=(0.0, 0.0),
    total_tokens=0,
    status="Error",
    exception=RuntimeError("Empty Result"),
)


def synchronize_device(device: torch.device) -> None:
    device_type = device.type
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
    elif device_type == "xpu":
        torch.xpu.synchronize()
    elif device_type == "mtia":
        torch.mtia.synchronize()
    elif device_type == "cpu":
        pass


class T2SEngine(T2SEngineProtocol):
    def __init__(
        self,
        decoder_model: T2SDecoderABC,
        device: Union[mx.Device, torch.device] = cpu,
        dtype: Union[torch.dtype, mx.Dtype] = torch.float32,
        start_background: bool = True,
        *args,
        **kwds,
    ) -> None:
        if isinstance(device, torch.device):
            device_type = device.type
            if device_type == "cpu":
                self.device = cpu
            elif device_type == "cuda" or device_type == "mps":
                self.device = mx.Device(mx.gpu)
            else:
                raise RuntimeError(f"Device {device} not supported")
        elif isinstance(device, mx.Device):
            self.device = device
        else:
            raise RuntimeError(f"Device {device} not supported")

        if dtype == torch.float32:
            self.dtype = mx.float16 if self.device.type == mx.gpu else mx.float32
        elif dtype == torch.float16:
            self.dtype = mx.float16
        elif dtype == torch.bfloat16:
            self.dtype = mx.bfloat16
        elif dtype == mx.float16 or dtype == mx.bfloat16 or dtype == mx.float32:
            self.dtype = dtype
        else:
            raise RuntimeError(f"Dtype {dtype} Not Supported")

        mx.set_default_device(self.device)
        decoder_model.set_dtype(self.dtype)

        self.decoder_model: T2SDecoderABC = decoder_model
        self.waiting_bind: deque[T2SSession] = deque()

        self.model_runner = ModelRunner(self.decoder_model, self.device, self.dtype, waiting_bind=self.waiting_bind)

        self.pending_requests: SimpleQueue[tuple[T2SRequest, T2SRequestHandle, bool]] = SimpleQueue()
        self.handles: dict[Union[str, int], T2SRequestHandle] = {}
        self._stop_event = threading.Event()
        self._request_counter = 0
        self._loop_thread: Optional[threading.Thread] = None

        if start_background:
            self._loop_thread = threading.Thread(target=self._background_loop, daemon=True)
            self._loop_thread.start()

    def add_request(
        self,
        request: T2SRequest,
        stream_interval: Optional[int] = None,
        show_progress: bool = False,
    ) -> T2SRequestHandle:
        if stream_interval is not None:
            request.stream_interval = stream_interval
        if request.request_id is None:
            self._request_counter += 1
            request.request_id = f"req-{self._request_counter}"

        handle = T2SRequestHandle(request_id=request.request_id)
        self.handles[request.request_id] = handle
        self.pending_requests.put((request, handle, show_progress))

        if self._loop_thread is None or not self._loop_thread.is_alive():
            self._loop_thread = threading.Thread(target=self._background_loop, daemon=True)
            self._loop_thread.start()

        return handle

    def stream_generate(self, request: T2SRequest, show_progress: bool = False):
        """Generator yielding streaming responses until completion."""
        handle = self.add_request(request, stream_interval=request.stream_interval, show_progress=show_progress)
        while True:
            resp = handle.queue.get()
            yield resp
            if resp.finished:
                break

    def generate(self, request: T2SRequest, show_progress: bool = False):
        """Blocking generate that consumes the streaming queue until finished."""
        final_result = empty_result
        for response in self.stream_generate(request, show_progress=show_progress):
            if response.finished:
                final_result = response.result or T2SResult(
                    status="Error",
                    exception=response.exception,
                    request_id=request.request_id,
                )
        return final_result

    def shutdown(self):
        self._stop_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=1.0)
            self._loop_thread = None

    def _emit_stream(
        self,
        session: T2SSession,
        tokens: Optional[list[torch.Tensor]],
        finished: bool = False,
        result: Optional[T2SResult] = None,
        exception: Optional[Exception] = None,
    ):
        handle = self.handles.get(session.request_id)
        if handle is None:
            return

        step = -1 if finished else session.step_count
        payload = T2SStreamResponse(
            request_id=session.request_id,
            step=step,
            tokens=tokens,
            finished=finished,
            result=result,
            exception=exception,
        )
        handle.queue.put(payload)
        if finished:
            handle.done = True
            self.handles.pop(session.request_id, None)

    def _maybe_stream_partial(self, session: T2SSession):
        if session.stream_interval <= 0:
            return
        if session.step_count == 0 or session.step_count - session.last_stream_step < session.stream_interval:
            return

        start = session.prompt_len + session.last_stream_step
        end = session.prompt_len + session.step_count
        tokens = [torch.tensor(self.model_runner.y_buf[session.slot_indices[i], start:end]) for i in range(session.bsz)]
        self._emit_stream(session, tokens=tokens, finished=False)
        session.last_stream_step = session.step_count

    def _finalize_session(
        self,
        session: T2SSession,
        exception: Optional[Exception] = None,
    ):
        request = session.request
        infer_time = max(time.perf_counter() - session.start_time, 1e-6)
        infer_speed = (session.total_tokens / infer_time if session.total_tokens else 0.0, infer_time)
        status: Literal["Success", "Error"] = "Error" if exception else "Success"

        if exception:
            logger.error(f"T2S request {session.request_id} failed: {exception}")
            traceback.print_exc()

        runner = self.model_runner
        tokens_out = [
            torch.tensor(
                runner.y_buf[session.slot_indices[i], session.prompt_len : session.prompt_len + session.step_count]
            )
            for i in range(min(request.valid_length, session.bsz))
        ]

        result = T2SResult(
            result=tokens_out if not exception else None,
            infer_speed=infer_speed,
            total_tokens=session.total_tokens,
            status=status,
            exception=exception,
            traceback=traceback.format_exc() if exception else None,
            request_id=session.request_id,
        )

        self._emit_stream(
            session,
            tokens=tokens_out if result.result is not None else None,
            finished=True,
            result=result,
            exception=exception,
        )

        if session.progress_task is not None and session.progress is not None:
            try:
                session.progress.update(
                    session.progress_task,
                    completed=session.progress.tasks[session.progress_task].total,
                )
                session.progress.remove_task(session.progress_task)
                session.progress.stop()
            except Exception:
                pass
            session.progress = None

    def _drain_new_requests(self) -> bool:
        new_added = False
        while True:
            try:
                request, handle, show_progress = self.pending_requests.get_nowait()
            except Empty:
                break

            try:
                session = self.model_runner.prefill(request)
            except Exception as e:
                logger.error(f"Prefill failed for request {request.request_id}: {e}")
                error_result = T2SResult(
                    status="Error",
                    exception=e,
                    traceback=traceback.format_exc(),
                    request_id=request.request_id,
                )
                handle.queue.put(
                    T2SStreamResponse(
                        request_id=request.request_id or "unknown",
                        step=-1,
                        tokens=None,
                        finished=True,
                        result=error_result,
                        exception=e,
                    )
                )
                handle.done = True
                self.handles.pop(request.request_id, None)
                continue

            total_tokens = max(1, session.max_decode_steps * session.bsz)
            if show_progress:
                session.progress = Progress(
                    TextColumn("[cyan]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total} tokens"),
                    SpeedColumnToken(show_speed=True),
                    console=console,
                    transient=True,
                )
                session.progress.start()
                session.progress_task = session.progress.add_task(f"T2S[{session.request_id}]", total=total_tokens)
            self.waiting_bind.append(session)
            new_added = True
        return new_added

    def _background_loop(self):
        torch.set_grad_enabled(False)
        while not self._stop_event.is_set():
            self._drain_new_requests()

            if not self.model_runner.bound_sessions:
                if not self.waiting_bind:
                    time.sleep(0.001)
                    continue

            try:
                finished_sessions = self.model_runner.decode_step()
            except Exception as e:
                for session in list(self.model_runner.bound_sessions):
                    self._finalize_session(session, exception=e)
                    self.model_runner.unbind_session(session)
                continue

            for session in list(self.model_runner.bound_sessions):
                self._maybe_stream_partial(session)
                if session.progress_task is not None and session.progress is not None:
                    session.progress.update(session.progress_task, advance=session.bsz)

            if finished_sessions:
                for session in finished_sessions:
                    self._finalize_session(session, exception=None)

    def __del__(self):
        self.shutdown()

    @staticmethod
    def load_decoder(
        weights_path: os.PathLike,
        max_batch_size: int = 1,
        backend: str = "MLX-Varlen",
        quantize_mode: Optional[Literal["Affine", "MXFP8"]] = None,
        compile: bool = False,
    ) -> T2SDecoderABC:
        logger.info(f"Loading Text2Semantic Weights from {weights_path} with {backend} Backend")
        dict_s1 = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
        config = dict_s1["config"]

        module_path = f".backends.{backend.lower().replace('-', '_')}"
        decoder_cls_name = "T2SDecoder"
        decoder_mod = import_module(module_path, package=__package__)
        decoder_cls: type[T2SDecoderABC] = getattr(decoder_mod, decoder_cls_name)
        decoder: T2SDecoderABC = decoder_cls(config, max_batch_size=max_batch_size)
        state_dict = dict_s1["weight"]
        state_dict_mlx = T2SEngine.replace_key(state_dict)

        decoder.load_weights(state_dict_mlx)

        if quantize_mode is not None:
            decoder.quantize(quantize_mode)
            logger.info(
                f"Quantized to {decoder.bits}-Bit with Group Size {decoder.group_size} by {quantize_mode} Quantization"
            )

        mx.eval(decoder)
        return decoder

    @staticmethod
    def replace_key(state_dict: dict[str, torch.Tensor]):
        state_dict_mlx: list[tuple[str, mx.array]] = []
        for key, value in state_dict.items():
            key = (
                key.replace("model.", "")
                .replace("in_proj_", "in_proj.")
                .replace("self_attn", "attention")
                .replace("linear", "feed_forward.linear")
                .replace("norm1", "attention_norm")
                .replace("norm2", "ffn_norm")
            )
            value_mlx = mx.array(value.to(torch.float32).cpu().numpy())
            state_dict_mlx.append((key, value_mlx))
        return state_dict_mlx
