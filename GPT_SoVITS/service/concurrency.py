"""Cancellation-safe threadpool and generator iteration primitives."""

import asyncio

import anyio
from starlette.concurrency import run_in_threadpool


STREAM_END = object()


async def await_shielded(awaitable):
    """Finish an in-flight worker operation before propagating cancellation."""

    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # AnyIO cancellation is level-triggered. Shield the worker until it
        # finishes, then let the original cancellation unwind the stream.
        with anyio.CancelScope(shield=True):
            try:
                await task
            except BaseException:
                pass
        raise


async def run_in_threadpool_shielded(function, *args):
    return await await_shielded(run_in_threadpool(function, *args))


def next_or_end(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return STREAM_END


async def next_tts_chunk(iterator, skip_empty=False):
    """Advance a synchronous TTS iterator without blocking the event loop."""

    while True:
        item = await run_in_threadpool_shielded(next_or_end, iterator)
        if item is STREAM_END or not skip_empty or item[1].size > 0:
            return item


async def close_generator(generator):
    """Close a synchronous generator with cancellation-safe worker affinity."""

    if generator is None:
        return
    close = getattr(generator, "close", None)
    if callable(close):
        await run_in_threadpool_shielded(close)
