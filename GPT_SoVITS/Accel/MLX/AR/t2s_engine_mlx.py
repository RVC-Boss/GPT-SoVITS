import os
import time
import traceback
from typing import Literal, cast

import mlx.core as mx
import torch
from rich.progress import BarColumn, Progress, TextColumn

from ...logger import SpeedColumnToken, Timer, console, logger
from ...PyTorch.AR.structs import T2SEngineProtocol, T2SRequest, T2SResult
from .backends import mlx_static, mlx_varlen
from .structs_mlx import T2SSessionMLX
from .t2s_model_abc import T2SDecoderABC

Array = mx.array
Tensor = torch.Tensor

timer = Timer()


class T2SEngine(T2SEngineProtocol):
    def __init__(
        self,
        decoder_model: T2SDecoderABC,
        device: mx.Device | torch.device = mx.Device(mx.cpu),
        dtype: torch.dtype | mx.Dtype = torch.float32,
        *args,
        **kwds,
    ) -> None:
        match device:
            case _ if isinstance(device, torch.device):
                match device.type:
                    case "cpu":
                        self.device = mx.Device(mx.gpu)
                    case "cuda" | "mps":
                        self.device = mx.Device(mx.gpu)
                    case _:
                        raise RuntimeError(f"Device {device} not supported")
            case _ if isinstance(device, mx.Device):
                self.device = device
            case _:
                raise RuntimeError(f"Device {device} not supported")

        match dtype:
            case torch.float32:
                self.dtype = mx.float16 if self.device.type == mx.gpu else mx.float32
            case torch.float16:
                self.dtype = mx.float16
            case torch.bfloat16:
                self.dtype = mx.bfloat16
            case mx.float16 | mx.bfloat16 | mx.float32:
                self.dtype = dtype
            case _:
                raise RuntimeError(f"Dtype {dtype} Not Supported")

        mx.set_default_device(self.device)
        decoder_model.set_dtype(self.dtype)

        self.decoder_model: T2SDecoderABC = decoder_model

    def _handle_request(self, request: T2SRequest):
        mx.clear_cache()
        decoder = self.decoder_model
        session = T2SSessionMLX(decoder, request, device=self.device, dtype=self.dtype)
        batch_idx = mx.arange(session.bsz)
        debug = request.debug

        t1 = 0.0
        infer_speed = 0.0
        infer_time = 0.0
        idx = 0

        with (
            Progress(
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                SpeedColumnToken(show_speed=True),
                console=console,
                transient=True,
            ) as progress,
        ):
            max_token = min(1500 - int(session.input_pos.max()), 1000) * session.bsz

            task = progress.add_task("T2S Decoding", total=max_token)
            for idx in range(max_token):
                progress.update(task, advance=session.bsz)
                if idx == 0:
                    session.kv_cache = decoder.init_cache(session.bsz)
                    t1 = time.perf_counter()
                    with timer("MLX.Prefill", debug=debug):
                        xy_dec = decoder.h.prefill(
                            session.xy_pos,
                            session.attn_mask,
                            session.kv_cache,
                        )  # bs, seq_len, embed_dim
                        xy_dec = xy_dec[batch_idx, None, session.input_pos - 1]
                        if debug:
                            mx.eval(xy_dec)
                else:
                    args, kwds = decoder.pre_forward(session)

                    if debug:
                        mx.eval(session.input_pos, session.xy_pos, session.kv_cache, args, kwds, batch_idx)

                    if debug and idx == 50 and os.environ.get("MTL_CAPTURE_ENABLED") == "1":
                        os.makedirs("./profiler/mlx", exist_ok=True)
                        mx.metal.start_capture(f"./profiler/mlx/{time.time()}.gputrace")

                    with timer("MLX.Decode", debug=debug):
                        xy_dec = decoder.h(
                            session.xy_pos,
                            session.input_pos,
                            int(session.input_pos.max()),
                            session.kv_cache,
                            batch_idx,
                            *args,
                            **kwds,
                        )
                        if debug:
                            mx.eval(xy_dec)

                    if debug and idx == 50 and os.environ.get("MTL_CAPTURE_ENABLED") == "1":
                        mx.metal.stop_capture()

                decoder.post_forward(idx, session)
                logits = decoder.ar_predict_layer(xy_dec.squeeze(1))
                session.input_pos += 1

                if idx == 0:
                    logits[:, -1] = -mx.inf

                with timer("MLX.Sampling", debug=debug):
                    samples = session.sample(
                        logits=logits,
                        previous_tokens=session.y[:, : session.y_len + idx],
                        top_k=request.top_k,
                        top_p=request.top_p,
                        repetition_penalty=request.repetition_penalty,
                        temperature=request.temperature,
                    )

                    session.y[batch_idx.reshape(-1, 1), session.y_len + idx] = samples

                    if debug:
                        mx.eval(samples)

                with timer("MLX.EOS", debug=debug):
                    mx.set_default_device(mx.Device(mx.cpu))
                    argmax_token = mx.argmax(logits, axis=-1)
                    sample_token = samples.squeeze(1)
                    EOS_mask = (cast(Array, argmax_token == decoder.EOS)) | (sample_token == decoder.EOS)

                    newly_done_mask = EOS_mask & (~session.completed)
                    newly_done_indices = mx.where(newly_done_mask, batch_idx, -1)
                    pos = mx.where(newly_done_indices != -1, batch_idx, session.bsz)
                    pos_sorted = mx.sort(pos, axis=0)
                    valid_count = session.bsz - mx.sum(cast(Array, pos_sorted == session.bsz))
                    pos_final = pos_sorted[: int(valid_count)]
                    newly_done_indices = newly_done_indices[pos_final]
                    mx.set_default_device(self.device)

                    if debug:
                        mx.eval(newly_done_indices)

                if newly_done_indices.size > 0:
                    for i in newly_done_indices:
                        session.y_results[int(i)] = session.y[i, session.y_len : session.y_len + idx]
                        session.completed[newly_done_indices] = True

                if mx.all(session.completed).item():
                    logger.info(
                        f"T2S Decoding EOS {session.prefill_len.tolist().__str__().strip('[]')} -> {[i.shape[-1] for i in session.y_results].__str__().strip('[]')}"
                    )
                    logger.info(f"Infer Speed: {(idx + 1) * session.bsz / (time.perf_counter() - t1):.2f} token/s")
                    infer_time = time.perf_counter() - t1
                    infer_speed = (idx + 1) * session.bsz / infer_time
                    break

                if (request.early_stop_num != -1 and idx >= request.early_stop_num) or idx == max_token - 1:
                    for j in range(session.bsz):
                        if not session.completed[j].item():
                            session.y_results[j] = session.y[j, session.y_len : session.y_len + idx]
                            session.completed[j] = True
                    logger.error("Bad Full Prediction")
                    logger.info(f"Infer Speed: {(idx + 1) * session.bsz / (time.perf_counter() - t1):.2f} token/s")
                    infer_time = time.perf_counter() - t1
                    infer_speed = (idx + 1) * session.bsz / infer_time
                    break

                with timer("MLX.NextPos", debug=debug):
                    y_emb = decoder.ar_audio_embedding(samples)
                    session.xy_pos = decoder.ar_audio_position(session.input_pos - session.x_lens, y_emb)
                    mx.eval(session.xy_pos, session.y)

                if idx % 128 == 0:
                    mx.clear_cache()

        result_mlx = session.y_results[: request.valid_length]
        mx.eval(result_mlx)
        result = [torch.tensor(k) for k in result_mlx]

        if debug:
            timer.summary()
            timer.clear()

        return result, infer_speed, infer_time, (idx + 1) * session.bsz

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
        return t2s_result

    @staticmethod
    def replace_key(state_dict: dict[str, Tensor]):
        state_dict_mlx: list[tuple[str, Array]] = []
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

    @staticmethod
    def load_decoder(
        weights_path: os.PathLike,
        max_batch_size: int = 1,
        backend: str = "MLX-Varlen",
        quantize_mode: Literal["Affine", "MXFP4"] | None = None,
    ) -> T2SDecoderABC:
        logger.info(f"Loading Text2Semantic Weights from {weights_path} with {backend} Backend")
        dict_s1 = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
        config = dict_s1["config"]
        match backend:
            case "MLX-Varlen":
                decoder_cls: type[T2SDecoderABC] = mlx_varlen.T2SDecoder
            case "MLX-Static":
                decoder_cls = mlx_static.T2SDecoder
            case _:
                raise RuntimeError(f"Backend {backend} Not Found")

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
