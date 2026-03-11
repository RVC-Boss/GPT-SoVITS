from __future__ import annotations

import asyncio
from typing import Callable

from GPT_SoVITS.TTS_infer_pack.t2s_scheduler import T2SRequestState
from GPT_SoVITS.TTS_infer_pack.unified_engine_components import EngineDispatchTask, EngineGpuPrepareTask


class EngineStageFutureMixin:
    def bind_notify_arbiter(self, notify_arbiter: Callable[[], None]) -> None:
        self._notify_arbiter = notify_arbiter

    def notify_arbiter(self) -> None:
        if self._notify_arbiter is not None:
            self._notify_arbiter()

    @staticmethod
    def _resolve_dispatch_error_future(future: asyncio.Future, error: Exception) -> None:
        if future.done():
            return
        future.set_exception(error)

    @staticmethod
    def _resolve_prepare_future(
        future: asyncio.Future,
        payload: tuple[T2SRequestState, float, float],
    ) -> None:
        if future.done():
            return
        future.set_result(payload)

    def _notify_dispatch_error(self, task: EngineDispatchTask, error: Exception) -> None:
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_dispatch_error_future, task.done_future, error)
        except RuntimeError:
            pass

    def _notify_prepare_error(self, task: EngineGpuPrepareTask, error: Exception) -> None:
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_dispatch_error_future, task.done_future, error)
        except RuntimeError:
            pass

    def _notify_prepare_result(
        self,
        task: EngineGpuPrepareTask,
        payload: tuple[T2SRequestState, float, float],
    ) -> None:
        if task.done_loop is None or task.done_future is None:
            return
        try:
            task.done_loop.call_soon_threadsafe(self._resolve_prepare_future, task.done_future, payload)
        except RuntimeError:
            pass
