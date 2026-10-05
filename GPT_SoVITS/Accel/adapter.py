from __future__ import annotations

import threading
from itertools import count

import torch

from tools.acceleration import resolve_acceleration
from .PyTorch.AR.structs import T2SRequest
from .PyTorch.AR.t2s_engine import T2SEngine


class AccelInference:
    def __init__(
        self,
        weights_path,
        device=None,
        dtype=torch.float16,
        max_batch_size=32,
        use_cuda_graph=True,
        use_flash_attention=True,
    ):
        self.device = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.weights_path = weights_path
        self.backend, _ = resolve_acceleration(
            self.device, self.dtype, use_cuda_graph, use_flash_attention
        )
        self.decoder = None
        if self.backend is not None:
            self.decoder = T2SEngine.load_decoder(
                weights_path,
                max_batch_size=max_batch_size,
                backend=self.backend,
            )
        self.engine = None
        self.graph_enabled = False
        self.graph_failed = False
        self._request_ids = count()
        self._mode_lock = threading.RLock()

    def prepare(self, use_cuda_graph=True, use_flash_attention=True):
        with self._mode_lock:
            backend, graph = resolve_acceleration(
                self.device,
                self.dtype,
                use_cuda_graph and not self.graph_failed,
                use_flash_attention,
            )
            if backend is None:
                self.close()
                return False
            if self.backend != backend or self.decoder is None:
                self.close()
                self.decoder = T2SEngine.load_decoder(
                    self.weights_path,
                    max_batch_size=self.max_batch_size,
                    backend=backend,
                )
                self.backend = backend
            if self.engine is None or self.graph_enabled != graph:
                self.close()
                self.engine = T2SEngine(
                    self.decoder,
                    device=self.device,
                    dtype=self.dtype,
                    use_cuda_graph=graph,
                )
                runner = self.engine.model_runner
                self.graph_enabled = runner.graph_applicable
                if runner.graph_error is not None:
                    self.graph_failed = True
            if backend == "torch_static_cuda_graph" and not self.graph_enabled:
                self.close()
                return False
            return True

    def infer_batch(
        self,
        x,
        x_lens,
        prompts,
        bert_feature,
        parallel_infer=True,
        use_cuda_graph=True,
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        early_stop_num=-1,
        repetition_penalty=1.35,
        use_flash_attention=True,
        **kwargs,
    ):
        with self._mode_lock:
            if not self.prepare(use_cuda_graph, use_flash_attention):
                raise RuntimeError("AR acceleration is unavailable; use the ordinary AR model")
            lengths = x_lens.detach().cpu().tolist()
            requests = []
            for index, length in enumerate(lengths):
                phones = x[index][:length]
                bert = bert_feature[index][:, :length]
                prompt = (
                    torch.empty((1, 0), dtype=torch.long, device=phones.device)
                    if prompts is None
                    else prompts[index : index + 1] if len(prompts) > 1 else prompts
                )
                requests.append(
                    T2SRequest(
                        x=[phones],
                        x_lens=torch.tensor([length], dtype=torch.long),
                        prompts=prompt,
                        bert_feature=[bert],
                        valid_length=1,
                        top_k=int(top_k),
                        top_p=float(top_p),
                        temperature=float(temperature),
                        early_stop_num=int(early_stop_num),
                        repetition_penalty=float(repetition_penalty),
                        use_cuda_graph=self.graph_enabled,
                        request_id=f"tts-{next(self._request_ids)}",
                        return_partial=False,
                    )
                )
            engine = self.engine
            if parallel_infer:
                handles = [engine.add_request(request) for request in requests]
                results = []
                for handle in handles:
                    while True:
                        response = handle.queue.get()
                        if response.finished:
                            results.append(response.result)
                            break
            else:
                results = [engine.generate(request) for request in requests]
            tokens = []
            for result in results:
                if result.exception is not None:
                    raise result.exception
                tokens.append(result.result[0])
        return tokens, [token.numel() for token in tokens]

    def close(self):
        with self._mode_lock:
            if self.engine is not None:
                self.engine.shutdown()
                self.engine = None
            self.graph_enabled = False
