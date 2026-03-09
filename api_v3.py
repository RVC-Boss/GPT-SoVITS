"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
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
    "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
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
    "streaming_mode": False,      # bool or int. return audio chunk by chunk.T he available options are: 0,1,2,3 or True/False (0/False: Disabled | 1/True: Best Quality, Slowest response speed (old version streaming_mode) | 2: Medium Quality, Slow response speed | 3: Lower Quality, Faster response speed )
    "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
    "min_chunk_length": 16,       # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400

"""

import asyncio
import os
import sys
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Union

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

from runtime_preload import preload_text_runtime_deps

preload_text_runtime_deps()

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.prepare_coordinator import PrepareCoordinator
from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import (
    SchedulerRequestSpec,
    T2SActiveBatch,
    T2SFinishedItem,
    T2SRequestState,
    merge_active_batches,
    decode_one_step,
    run_prefill_active_batch,
    run_scheduler_continuous,
)
from GPT_SoVITS.TTS_infer_pack.unified_engine import RuntimeControlCallbacks, UnifiedTTSEngine
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
import threading

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)
tts_engine = UnifiedTTSEngine(
    tts_pipeline,
    cut_method_names=cut_method_names,
    control_callbacks=RuntimeControlCallbacks(
        restart=lambda: os.execl(sys.executable, sys.executable, *argv),
        exit=lambda: os.kill(os.getpid(), signal.SIGTERM),
    ),
)

APP = FastAPI()


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16


class Scheduler_Debug_Request_Item(BaseModel):
    request_id: str | None = None
    text: str
    text_lang: str
    ref_audio_path: str
    prompt_lang: str
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    repetition_penalty: float = 1.35
    early_stop_num: int = -1
    ready_step: int = 0


class Scheduler_Debug_Request(BaseModel):
    requests: List[Scheduler_Debug_Request_Item]
    max_steps: int = 1500
    seed: int = -1


class Scheduler_Submit_Request(BaseModel):
    request_id: str | None = None
    text: str
    text_lang: str
    ref_audio_path: str
    prompt_lang: str
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    repetition_penalty: float = 1.35
    early_stop_num: int = -1
    speed_factor: float = 1.0
    sample_steps: int = 32
    media_type: str = "wav"
    timeout_sec: float = 30.0


@dataclass
class SchedulerPendingJob:
    request_id: str
    state: T2SRequestState
    done_event: threading.Event
    done_loop: asyncio.AbstractEventLoop | None
    done_future: asyncio.Future | None
    enqueue_time: float
    speed_factor: float
    sample_steps: int
    media_type: str
    prepare_wall_ms: float = 0.0
    prepare_profile_total_ms: float = 0.0
    first_schedule_time: float | None = None
    prefill_ms: float = 0.0
    merge_ms: float = 0.0
    decode_ms: float = 0.0
    finalize_wait_ms: float = 0.0
    synth_ms: float = 0.0
    pack_ms: float = 0.0
    decode_steps: int = 0
    result_ready_time: float | None = None
    result: dict | None = None
    sample_rate: int | None = None
    audio_data: np.ndarray | None = None
    error: str | None = None


@dataclass
class SchedulerFinalizeTask:
    request_id: str
    item: T2SFinishedItem
    enqueued_time: float


class SchedulerDebugWorker:
    def __init__(self, tts: TTS, max_steps: int = 1500, micro_batch_wait_ms: int = 5):
        self.tts = tts
        self.max_steps = max_steps
        self.micro_batch_wait_s = micro_batch_wait_ms / 1000.0
        self.prepare_coordinator = PrepareCoordinator(tts)
        self.condition = threading.Condition()
        self.prepare_inflight = 0
        self.prepare_peak_inflight = 0
        self.finalize_condition = threading.Condition()
        self.finalize_pending_tasks: deque[SchedulerFinalizeTask] = deque()
        self.finalize_pending_peak = 0
        self.finalize_inflight = 0
        self.finalize_inflight_peak = 0
        self.finalize_workers = max(1, int(os.environ.get("GPTSOVITS_FINALIZE_WORKERS", 1)))
        self.finalize_mode = os.environ.get("GPTSOVITS_FINALIZE_MODE", "async").strip().lower()
        self.finalize_batch_max_items = max(1, int(os.environ.get("GPTSOVITS_FINALIZE_BATCH_MAX_ITEMS", 16)))
        self.finalize_batch_wait_s = max(0.0, float(os.environ.get("GPTSOVITS_FINALIZE_BATCH_WAIT_MS", "2")) / 1000.0)
        self.pending_jobs: List[SchedulerPendingJob] = []
        self.active_batch: T2SActiveBatch | None = None
        self.job_map: dict[str, SchedulerPendingJob] = {}
        self.total_finished = 0
        self.total_submitted = 0
        self.worker_thread = threading.Thread(target=self._run_loop, name="t2s-scheduler-debug-worker", daemon=True)
        self.worker_thread.start()
        self.finalize_threads = [
            threading.Thread(
                target=self._run_finalize_loop,
                name=f"t2s-scheduler-finalize-{worker_index}",
                daemon=True,
            )
            for worker_index in range(self.finalize_workers)
        ]
        for finalize_thread in self.finalize_threads:
            finalize_thread.start()

    def _sync_device(self) -> None:
        try:
            device_str = str(self.tts.configs.device)
            if device_str.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(self.tts.configs.device)
            elif device_str == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass

    def prepare_state(self, spec: SchedulerRequestSpec) -> T2SRequestState:
        raise RuntimeError("prepare_state sync path has been replaced by PrepareCoordinator")

    def submit(
        self,
        state: T2SRequestState,
        speed_factor: float,
        sample_steps: int,
        media_type: str,
        prepare_wall_ms: float,
        prepare_profile_total_ms: float,
        done_loop: asyncio.AbstractEventLoop | None = None,
        done_future: asyncio.Future | None = None,
    ) -> SchedulerPendingJob:
        job = SchedulerPendingJob(
            request_id=state.request_id,
            state=state,
            done_event=threading.Event(),
            done_loop=done_loop,
            done_future=done_future,
            enqueue_time=time.perf_counter(),
            speed_factor=float(speed_factor),
            sample_steps=int(sample_steps),
            media_type=media_type,
            prepare_wall_ms=float(prepare_wall_ms),
            prepare_profile_total_ms=float(prepare_profile_total_ms),
        )
        with self.condition:
            self.pending_jobs.append(job)
            self.job_map[job.request_id] = job
            self.total_submitted += 1
            self.condition.notify_all()
        with self.finalize_condition:
            self.finalize_condition.notify_all()
        return job

    async def prepare_state_async(self, spec: SchedulerRequestSpec) -> T2SRequestState:
        state, _, _ = await self.prepare_coordinator.prepare_state_profiled_async(spec, time.perf_counter())
        return state

    async def prepare_states_batch_async(self, specs: List[SchedulerRequestSpec]) -> List[T2SRequestState]:
        return await asyncio.gather(*[self.prepare_state_async(spec) for spec in specs])

    async def prepare_state_profiled_async(
        self,
        spec: SchedulerRequestSpec,
        prepare_submit_at: float,
    ) -> tuple[T2SRequestState, float, float]:
        return await self.prepare_coordinator.prepare_state_profiled_async(spec, prepare_submit_at)

    def _mark_prefill_started(self, jobs: List[SchedulerPendingJob], started_at: float) -> None:
        with self.condition:
            for job in jobs:
                tracked_job = self.job_map.get(job.request_id)
                if tracked_job is not None and tracked_job.first_schedule_time is None:
                    tracked_job.first_schedule_time = started_at

    def _add_prefill_time(self, jobs: List[SchedulerPendingJob], elapsed_s: float) -> None:
        elapsed_ms = elapsed_s * 1000.0
        with self.condition:
            for job in jobs:
                tracked_job = self.job_map.get(job.request_id)
                if tracked_job is not None:
                    tracked_job.prefill_ms += elapsed_ms

    def _add_merge_time(self, request_ids: List[str], elapsed_s: float) -> None:
        elapsed_ms = elapsed_s * 1000.0
        with self.condition:
            for request_id in request_ids:
                job = self.job_map.get(request_id)
                if job is not None:
                    job.merge_ms += elapsed_ms

    def _add_decode_time(self, request_ids: List[str], elapsed_s: float) -> None:
        elapsed_ms = elapsed_s * 1000.0
        with self.condition:
            for request_id in request_ids:
                job = self.job_map.get(request_id)
                if job is not None:
                    job.decode_ms += elapsed_ms
                    job.decode_steps += 1

    def _add_finalize_wait_ms(self, request_ids: List[str], elapsed_ms: float) -> None:
        with self.condition:
            for request_id in request_ids:
                job = self.job_map.get(request_id)
                if job is not None:
                    job.finalize_wait_ms += elapsed_ms

    def _synthesize_finished_audio(self, job: SchedulerPendingJob, item: T2SFinishedItem) -> tuple[int, np.ndarray]:
        semantic_tokens = item.semantic_tokens.detach().clone().unsqueeze(0).unsqueeze(0).to(self.tts.configs.device)
        phones = job.state.phones.detach().clone().unsqueeze(0).to(self.tts.configs.device)
        prompt_semantic = job.state.prompt_semantic.detach().clone()
        prompt_phones = job.state.prompt_phones.detach().clone()
        refer_spec = (
            job.state.refer_spec[0].detach().clone(),
            None if job.state.refer_spec[1] is None else job.state.refer_spec[1].detach().clone(),
        )
        raw_audio = job.state.raw_audio.detach().clone()
        audio_fragment = self.tts.synthesize_audio_request_local(
            semantic_tokens=semantic_tokens,
            phones=phones,
            prompt_semantic=prompt_semantic,
            prompt_phones=prompt_phones,
            refer_spec=refer_spec,
            raw_audio=raw_audio,
            raw_sr=job.state.raw_sr,
            speed=float(job.speed_factor),
            sample_steps=int(job.sample_steps),
        )
        output_sr = self.tts.configs.sampling_rate if not self.tts.configs.use_vocoder else self.tts.vocoder_configs["sr"]
        return self.tts.audio_postprocess(
            audio=[[audio_fragment]],
            sr=int(output_sr),
            batch_index_list=None,
            speed_factor=float(job.speed_factor),
            split_bucket=False,
            fragment_interval=0.0,
            super_sampling=False,
        )

    def get_state(self) -> dict:
        with self.finalize_condition:
            finalize_pending = len(self.finalize_pending_tasks)
            finalize_pending_peak = self.finalize_pending_peak
            finalize_inflight = self.finalize_inflight
            finalize_inflight_peak = self.finalize_inflight_peak
        with self.condition:
            bert_stage = self.tts.prepare_bert_stage_limiter.snapshot()
            ref_audio_stage = self.tts.prepare_ref_audio_stage_limiter.snapshot()
            bert_batch_worker = (
                None
                if self.tts.prepare_bert_batch_worker is None
                else self.tts.prepare_bert_batch_worker.snapshot()
            )
            ref_semantic_batch_worker = (
                None
                if self.tts.prepare_ref_semantic_batch_worker is None
                else self.tts.prepare_ref_semantic_batch_worker.snapshot()
            )
            prepare_coordinator_state = self.prepare_coordinator.snapshot()
            return {
                "pending_jobs": len(self.pending_jobs),
                "running_requests": 0 if self.active_batch is None else len(self.active_batch.request_ids),
                "prepare_inflight": prepare_coordinator_state["inflight"],
                "prepare_peak_inflight": prepare_coordinator_state["peak_inflight"],
                "finalize_pending": finalize_pending,
                "finalize_pending_peak": finalize_pending_peak,
                "finalize_inflight": finalize_inflight,
                "finalize_inflight_peak": finalize_inflight_peak,
                "finalize_workers": self.finalize_workers,
                "finalize_mode": self.finalize_mode,
                "finalize_batch_max_items": self.finalize_batch_max_items,
                "finalize_batch_wait_ms": self.finalize_batch_wait_s * 1000.0,
                "prepare_request_executor_workers": 0,
                "prepare_text_cpu_workers": int(getattr(self.tts, "prepare_text_cpu_workers", 0)),
                "prepare_text_feature_workers": int(prepare_coordinator_state["text_feature_workers"]),
                "prepare_ref_audio_workers": int(prepare_coordinator_state["ref_audio_workers"]),
                "prepare_bert_stage": bert_stage,
                "prepare_bert_batch_worker": bert_batch_worker,
                "prepare_ref_audio_stage": ref_audio_stage,
                "prepare_ref_semantic_batch_worker": ref_semantic_batch_worker,
                "tracked_jobs": len(self.job_map),
                "total_submitted": self.total_submitted,
                "total_finished": self.total_finished,
                "max_steps": self.max_steps,
                "micro_batch_wait_ms": int(self.micro_batch_wait_s * 1000),
            }

    def _enqueue_finalize_finished(self, items: List[T2SFinishedItem]) -> None:
        if not items:
            return
        tasks: List[SchedulerFinalizeTask] = []
        enqueued_time = time.perf_counter()
        with self.condition:
            for item in items:
                job = self.job_map.get(item.request_id)
                if job is not None:
                    tasks.append(
                        SchedulerFinalizeTask(
                            request_id=item.request_id,
                            item=item,
                            enqueued_time=enqueued_time,
                        )
                    )
        if not tasks:
            return
        with self.finalize_condition:
            self.finalize_pending_tasks.extend(tasks)
            if len(self.finalize_pending_tasks) > self.finalize_pending_peak:
                self.finalize_pending_peak = len(self.finalize_pending_tasks)
            self.finalize_condition.notify_all()

    @staticmethod
    def _finalize_batch_key(job: SchedulerPendingJob) -> tuple[float, int]:
        return (round(float(job.speed_factor), 6), int(job.sample_steps))

    def _take_finalize_task_batch(self) -> List[SchedulerFinalizeTask]:
        with self.finalize_condition:
            while not self.finalize_pending_tasks:
                self.finalize_condition.wait()
            if self.finalize_mode == "after_t2s_drain":
                while not self._is_t2s_drained():
                    self.finalize_condition.wait(timeout=self.micro_batch_wait_s)
            task = self.finalize_pending_tasks.popleft()
            selected_tasks = [task]
            batch_key = None
            with self.condition:
                first_job = self.job_map.get(task.request_id)
                if first_job is not None:
                    batch_key = self._finalize_batch_key(first_job)
            batch_deadline = time.perf_counter() + self.finalize_batch_wait_s
            while len(selected_tasks) < self.finalize_batch_max_items:
                if batch_key is None:
                    break
                matched_index = None
                for pending_index, pending_task in enumerate(self.finalize_pending_tasks):
                    with self.condition:
                        pending_job = self.job_map.get(pending_task.request_id)
                    if pending_job is None:
                        matched_index = pending_index
                        break
                    if self._finalize_batch_key(pending_job) == batch_key:
                        matched_index = pending_index
                        break
                if matched_index is not None:
                    selected_tasks.append(self.finalize_pending_tasks[matched_index])
                    del self.finalize_pending_tasks[matched_index]
                    continue
                remaining = batch_deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self.finalize_condition.wait(timeout=remaining)
            self.finalize_inflight += len(selected_tasks)
            if self.finalize_inflight > self.finalize_inflight_peak:
                self.finalize_inflight_peak = self.finalize_inflight
            return selected_tasks

    def _finalize_task_done(self, count: int) -> None:
        with self.finalize_condition:
            self.finalize_inflight = max(0, self.finalize_inflight - count)

    def _is_t2s_drained(self) -> bool:
        with self.condition:
            return (
                self.active_batch is None
                and not self.pending_jobs
                and self.prepare_inflight <= 0
            )

    def _complete_finalize_task(self, job: SchedulerPendingJob, item: T2SFinishedItem, sample_rate: int, audio_data: np.ndarray) -> None:
        finished_at = time.perf_counter()
        with self.condition:
            if self.job_map.get(item.request_id) is not job:
                return
            queue_wait_ms = 0.0
            if job.first_schedule_time is not None:
                queue_wait_ms = max(0.0, (job.first_schedule_time - job.enqueue_time) * 1000.0)
            worker_total_ms = max(0.0, (finished_at - job.enqueue_time) * 1000.0)
            worker_residual_ms = max(
                0.0,
                worker_total_ms
                - queue_wait_ms
                - job.prefill_ms
                - job.merge_ms
                - job.decode_ms
                - job.finalize_wait_ms
                - job.synth_ms,
            )
            worker_other_ms = max(0.0, job.merge_ms + job.finalize_wait_ms + worker_residual_ms)
            job.sample_rate = int(sample_rate)
            job.audio_data = audio_data
            job.result_ready_time = finished_at
            prepare_profile = dict(job.state.prepare_profile)
            job.result = {
                "request_id": item.request_id,
                "semantic_len": int(item.semantic_tokens.shape[0]),
                "finish_idx": int(item.finish_idx),
                "finish_reason": item.finish_reason,
                "prepare_ms": job.prepare_wall_ms,
                "prepare_wall_ms": job.prepare_wall_ms,
                "prepare_profile_total_ms": job.prepare_profile_total_ms,
                "prepare_profile": prepare_profile,
                "queue_wait_ms": queue_wait_ms,
                "prefill_ms": job.prefill_ms,
                "merge_ms": job.merge_ms,
                "decode_ms": job.decode_ms,
                "finalize_wait_ms": job.finalize_wait_ms,
                "synth_ms": job.synth_ms,
                "worker_residual_ms": worker_residual_ms,
                "worker_other_ms": worker_other_ms,
                "worker_total_ms": worker_total_ms,
                "decode_steps": int(job.decode_steps),
                "sample_rate": int(sample_rate),
                "media_type": job.media_type,
            }
            job.done_event.set()
            self._notify_done_future(job)
            self.job_map.pop(item.request_id, None)
            self.total_finished += 1

    def _synthesize_finished_audio_batch(
        self,
        jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]],
    ) -> List[tuple[int, np.ndarray]]:
        semantic_tokens_list = [item.semantic_tokens.detach().clone() for _, item in jobs_and_items]
        phones_list = [job.state.phones.detach().clone() for job, _ in jobs_and_items]
        refer_specs = []
        speeds = []
        sample_steps_list = []
        for job, _ in jobs_and_items:
            refer_specs.append(
                (
                    job.state.refer_spec[0].detach().clone(),
                    None if job.state.refer_spec[1] is None else job.state.refer_spec[1].detach().clone(),
                )
            )
            speeds.append(float(job.speed_factor))
            sample_steps_list.append(int(job.sample_steps))
        audio_fragments = self.tts.synthesize_audio_requests_local_batched(
            semantic_tokens_list=semantic_tokens_list,
            phones_list=phones_list,
            refer_specs=refer_specs,
            speeds=speeds,
            sample_steps_list=sample_steps_list,
        )
        output_sr = self.tts.configs.sampling_rate if not self.tts.configs.use_vocoder else self.tts.vocoder_configs["sr"]
        results: List[tuple[int, np.ndarray]] = []
        for (job, _), audio_fragment in zip(jobs_and_items, audio_fragments):
            results.append(
                self.tts.audio_postprocess(
                    audio=[[audio_fragment]],
                    sr=int(output_sr),
                    batch_index_list=None,
                    speed_factor=float(job.speed_factor),
                    split_bucket=False,
                    fragment_interval=0.0,
                    super_sampling=False,
                )
            )
        return results

    def _run_finalize_loop(self) -> None:
        while True:
            tasks = self._take_finalize_task_batch()
            try:
                jobs_and_items: List[tuple[SchedulerPendingJob, T2SFinishedItem]] = []
                finalize_wait_request_ids: List[str] = []
                with self.condition:
                    for task in tasks:
                        job = self.job_map.get(task.request_id)
                        if job is None:
                            continue
                        jobs_and_items.append((job, task.item))
                        finalize_wait_request_ids.append(task.request_id)
                if not jobs_and_items:
                    continue
                now = time.perf_counter()
                for task in tasks:
                    self._add_finalize_wait_ms([task.request_id], max(0.0, (now - task.enqueued_time) * 1000.0))
                self._sync_device()
                synth_start = time.perf_counter()
                if len(jobs_and_items) == 1 or self.tts.configs.use_vocoder:
                    job, item = jobs_and_items[0]
                    batch_results = [self._synthesize_finished_audio(job, item)]
                else:
                    batch_results = self._synthesize_finished_audio_batch(jobs_and_items)
                self._sync_device()
                synth_ms = (time.perf_counter() - synth_start) * 1000.0
                with self.condition:
                    for job, _ in jobs_and_items:
                        tracked_job = self.job_map.get(job.request_id)
                        if tracked_job is not None:
                            tracked_job.synth_ms += synth_ms
                for (job, item), (sample_rate, audio_data) in zip(jobs_and_items, batch_results):
                    self._complete_finalize_task(job, item, sample_rate=sample_rate, audio_data=audio_data)
            except Exception as exc:
                self._finalize_error([task.request_id for task in tasks], str(exc))
            finally:
                self._finalize_task_done(len(tasks))

    def _finalize_error(self, request_ids: List[str], error: str) -> None:
        if not request_ids:
            return
        with self.condition:
            for request_id in request_ids:
                job = self.job_map.get(request_id)
                if job is None:
                    continue
                job.error = error
                job.done_event.set()
                self._notify_done_future(job)
                self.job_map.pop(request_id, None)
                self.total_finished += 1

    @staticmethod
    def _resolve_done_future(job: SchedulerPendingJob) -> None:
        future = job.done_future
        if future is None or future.done():
            return
        future.set_result(True)

    def _notify_done_future(self, job: SchedulerPendingJob) -> None:
        if job.done_loop is None or job.done_future is None:
            return
        try:
            job.done_loop.call_soon_threadsafe(self._resolve_done_future, job)
        except RuntimeError:
            pass

    def _take_pending_snapshot(self, wait_for_batch: bool) -> List[SchedulerPendingJob]:
        with self.condition:
            if not self.pending_jobs and self.active_batch is None:
                self.condition.wait(timeout=self.micro_batch_wait_s)
            elif wait_for_batch and self.pending_jobs:
                self.condition.wait(timeout=self.micro_batch_wait_s)
            if not self.pending_jobs:
                return []
            pending = list(self.pending_jobs)
            self.pending_jobs.clear()
            with self.finalize_condition:
                self.finalize_condition.notify_all()
            return pending

    def _run_loop(self) -> None:
        while True:
            wait_for_batch = self.active_batch is None
            pending_jobs = self._take_pending_snapshot(wait_for_batch=wait_for_batch)

            if pending_jobs:
                try:
                    self._sync_device()
                    prefill_start = time.perf_counter()
                    self._mark_prefill_started(pending_jobs, prefill_start)
                    admitted_active_batch, admitted_finished = run_prefill_active_batch(
                        self.tts.t2s_model.model,
                        [job.state for job in pending_jobs],
                        max_steps=self.max_steps,
                    )
                    self._sync_device()
                    self._add_prefill_time(pending_jobs, time.perf_counter() - prefill_start)
                    self._enqueue_finalize_finished(admitted_finished)
                    merge_start = time.perf_counter()
                    self.active_batch = merge_active_batches(
                        self.tts.t2s_model.model,
                        self.active_batch,
                        admitted_active_batch,
                    )
                    self._add_merge_time(
                        [] if self.active_batch is None else list(self.active_batch.request_ids),
                        time.perf_counter() - merge_start,
                    )
                    with self.finalize_condition:
                        self.finalize_condition.notify_all()
                except Exception as exc:
                    self._finalize_error([job.request_id for job in pending_jobs], str(exc))

            if self.active_batch is not None:
                try:
                    active_request_ids = [state.request_id for state in self.active_batch.states]
                    self._sync_device()
                    decode_start = time.perf_counter()
                    self.active_batch, step_finished = decode_one_step(
                        self.tts.t2s_model.model,
                        self.active_batch,
                        max_steps=self.max_steps,
                    )
                    self._sync_device()
                    self._add_decode_time(active_request_ids, time.perf_counter() - decode_start)
                    self._enqueue_finalize_finished(step_finished)
                    with self.finalize_condition:
                        self.finalize_condition.notify_all()
                except Exception as exc:
                    self._finalize_error(active_request_ids, str(exc))
                    self.active_batch = None
                    with self.finalize_condition:
                        self.finalize_condition.notify_all()
                continue

            if not pending_jobs:
                with self.finalize_condition:
                    self.finalize_condition.notify_all()
                time.sleep(self.micro_batch_wait_s)


scheduler_debug_worker = tts_engine.scheduler_worker


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    # Author: AkagawaTsurunaki
    # Issue:
    #   Stack overflow probabilistically occurs
    #   when the function `sf_writef_short` of `libsndfile_64bit.dll` is called
    #   using the Python library `soundfile`
    # Note:
    #   This is an issue related to `libsndfile`, not this project itself.
    #   It happens when you generate a large audio tensor (about 499804 frames in my PC)
    #   and try to convert it to an ogg file.
    # Related:
    #   https://github.com/RVC-Boss/GPT-SoVITS/issues/1199
    #   https://github.com/libsndfile/libsndfile/issues/1023
    #   https://github.com/bastibe/python-soundfile/issues/396
    # Suggestion:
    #   Or split the whole audio data into smaller audio segment to avoid stack overflow?

    def handle_pack_ogg():
        with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)



    # See: https://docs.python.org/3/library/threading.html
    # The stack size of this thread is at least 32768
    # If stack overflow error still occurs, just modify the `stack_size`.
    # stack_size = n * 4096, where n should be a positive integer.
    # Here we chose n = 4096.
    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as e:
        # If changing the thread stack size is unsupported, a RuntimeError is raised.
        print("RuntimeError: {}".format(e))
        print("Changing the thread stack size is unsupported.")
    except ValueError as e:
        # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
        print("ValueError: {}".format(e))
        print("The specified stack size is invalid.")

    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    # elif media_type == "ogg" and not streaming_mode:
    #     return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


def set_scheduler_seed(seed: int):
    if seed in ["", None]:
        return
    seed = int(seed)
    if seed < 0:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_scheduler_request_specs(request_items: List[Scheduler_Debug_Request_Item]) -> List[SchedulerRequestSpec]:
    specs: List[SchedulerRequestSpec] = []
    for index, item in enumerate(request_items):
        payload = item.dict()
        req = {
            "text": payload["text"],
            "text_lang": payload["text_lang"].lower(),
            "ref_audio_path": payload["ref_audio_path"],
            "aux_ref_audio_paths": None,
            "prompt_text": payload["prompt_text"],
            "prompt_lang": payload["prompt_lang"].lower(),
            "top_k": payload["top_k"],
            "top_p": payload["top_p"],
            "temperature": payload["temperature"],
            "text_split_method": "cut5",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "speed_factor": 1.0,
            "split_bucket": False,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": False,
            "repetition_penalty": payload["repetition_penalty"],
            "sample_steps": 32,
            "super_sampling": False,
            "overlap_length": 2,
            "min_chunk_length": 16,
        }
        check_res = check_params(req)
        if check_res is not None:
            detail = check_res.body.decode("utf-8") if hasattr(check_res, "body") else str(check_res)
            raise ValueError(f"request[{index}] 参数非法: {detail}")
        specs.append(
            SchedulerRequestSpec(
                request_id=payload["request_id"] or f"req_{index:03d}",
                ref_audio_path=Path(payload["ref_audio_path"]),
                prompt_text=payload["prompt_text"],
                prompt_lang=payload["prompt_lang"].lower(),
                text=payload["text"],
                text_lang=payload["text_lang"].lower(),
                top_k=int(payload["top_k"]),
                top_p=float(payload["top_p"]),
                temperature=float(payload["temperature"]),
                repetition_penalty=float(payload["repetition_penalty"]),
                early_stop_num=int(payload["early_stop_num"]),
                ready_step=int(payload["ready_step"]),
            )
        )
    return specs


def summarize_scheduler_states(states: List[T2SRequestState]) -> List[dict]:
    return [
        {
            "request_id": state.request_id,
            "ready_step": int(state.ready_step),
            "ref_audio_path": str(state.ref_audio_path),
            "prompt_semantic_len": int(state.prompt_semantic.shape[0]),
            "all_phone_len": int(state.all_phones.shape[0]),
            "bert_len": int(state.all_bert_features.shape[-1]),
            "norm_text": state.norm_text,
        }
        for state in states
    ]


def summarize_scheduler_finished(items: List[T2SFinishedItem]) -> List[dict]:
    return [
        {
            "request_id": item.request_id,
            "semantic_len": int(item.semantic_tokens.shape[0]),
            "finish_idx": int(item.finish_idx),
            "finish_reason": item.finish_reason,
        }
        for item in items
    ]


def build_scheduler_submit_spec(request: Scheduler_Submit_Request) -> SchedulerRequestSpec:
    payload = request.dict()
    request_id = payload["request_id"] or f"job_{uuid.uuid4().hex[:12]}"
    req = {
        "text": payload["text"],
        "text_lang": payload["text_lang"].lower(),
        "ref_audio_path": payload["ref_audio_path"],
        "aux_ref_audio_paths": None,
        "prompt_text": payload["prompt_text"],
        "prompt_lang": payload["prompt_lang"].lower(),
        "top_k": payload["top_k"],
        "top_p": payload["top_p"],
        "temperature": payload["temperature"],
        "text_split_method": "cut5",
        "batch_size": 1,
        "batch_threshold": 0.75,
        "speed_factor": float(payload["speed_factor"]),
        "split_bucket": False,
        "fragment_interval": 0.3,
        "seed": -1,
        "media_type": payload["media_type"],
        "streaming_mode": False,
        "parallel_infer": False,
        "repetition_penalty": payload["repetition_penalty"],
        "sample_steps": int(payload["sample_steps"]),
        "super_sampling": False,
        "overlap_length": 2,
        "min_chunk_length": 16,
    }
    check_res = check_params(req)
    if check_res is not None:
        detail = check_res.body.decode("utf-8") if hasattr(check_res, "body") else str(check_res)
        raise ValueError(f"request 参数非法: {detail}")
    return SchedulerRequestSpec(
        request_id=request_id,
        ref_audio_path=Path(payload["ref_audio_path"]),
        prompt_text=payload["prompt_text"],
        prompt_lang=payload["prompt_lang"].lower(),
        text=payload["text"],
        text_lang=payload["text_lang"].lower(),
        top_k=int(payload["top_k"]),
        top_p=float(payload["top_p"]),
        temperature=float(payload["temperature"]),
        repetition_penalty=float(payload["repetition_penalty"]),
        early_stop_num=int(payload["early_stop_num"]),
        ready_step=0,
    )


async def tts_scheduler_debug_handle(request: Scheduler_Debug_Request):
    try:
        result = await tts_engine.run_scheduler_debug(
            request_items=[item.dict() for item in request.requests],
            max_steps=int(request.max_steps),
            seed=int(request.seed),
        )
        return JSONResponse(status_code=200, content=result.payload)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": "scheduler debug failed", "Exception": str(e)},
        )


async def tts_scheduler_submit_handle(request: Scheduler_Submit_Request):
    try:
        result = await tts_engine.run_scheduler_submit(request.dict())
        return Response(result.audio_bytes, media_type=result.media_type, headers=result.headers)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": "scheduler submit failed", "Exception": str(e)},
        )


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
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
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
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
                "streaming_mode": False,      # bool or int. return audio chunk by chunk.T he available options are: 0,1,2,3 or True/False (0/False: Disabled | 1/True: Best Quality, Slowest response speed (old version streaming_mode) | 2: Medium Quality, Slow response speed | 3: Lower Quality, Faster response speed )
                "overlap_length": 2,          # int. overlap length of semantic tokens for streaming mode.
                "min_chunk_length": 16,       # int. The minimum chunk length of semantic tokens for streaming mode. (affects audio chunk size)
            }
    returns:
        StreamingResponse: audio stream response.
    """

    try:
        result = tts_engine.run_direct_tts(req)
        if result.streaming:
            return StreamingResponse(result.audio_generator, media_type=f"audio/{result.media_type}")
        return Response(result.audio_bytes, media_type=f"audio/{result.media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    try:
        tts_engine.handle_control(command)
        return JSONResponse(status_code=200, content={"message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "control failed", "Exception": str(e)})


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 15,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
    streaming_mode: Union[bool, int] = False,
    overlap_length: int = 2,
    min_chunk_length: int = 16,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
        "overlap_length": int(overlap_length),
        "min_chunk_length": int(min_chunk_length),
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.post("/tts_scheduler_debug")
async def tts_scheduler_debug_endpoint(request: Scheduler_Debug_Request):
    return await tts_scheduler_debug_handle(request)


@APP.post("/tts_scheduler_submit")
async def tts_scheduler_submit_endpoint(request: Scheduler_Submit_Request):
    return await tts_scheduler_submit_handle(request)


@APP.get("/tts_scheduler_state")
async def tts_scheduler_state_endpoint():
    return JSONResponse(status_code=200, content=tts_engine.get_runtime_state())


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        payload = tts_engine.set_refer_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content=payload)


# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})

#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())

#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        payload = tts_engine.set_gpt_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content=payload)


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        payload = tts_engine.set_sovits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content=payload)


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
