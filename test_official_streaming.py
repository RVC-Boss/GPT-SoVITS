"""Regressions for the opt-in official streaming path, without loading weights."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from itertools import count
import sys
import threading
import numpy as np
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent / "GPT_SoVITS")]
from GPT_SoVITS.Accel.adapter import AccelInference
from GPT_SoVITS.Accel.PyTorch.AR.structs import T2SRequestHandle, T2SResult, T2SStreamResponse
from GPT_SoVITS.Accel.PyTorch.AR.t2s_engine import T2SEngine
from module.models import CFM, SynthesizerTrn

# Load the official helper file directly so importing its package does not
# eagerly import the full TTS stack (and its optional librosa/numba cache).
_streaming_path = Path(__file__).resolve().parent / "GPT_SoVITS" / "TTS_infer_pack" / "streaming.py"
_streaming_spec = importlib.util.spec_from_file_location("GPT_SoVITS.TTS_infer_pack.streaming", _streaming_path)
_streaming_module = importlib.util.module_from_spec(_streaming_spec)
_streaming_spec.loader.exec_module(_streaming_module)
v2_token_chunks = _streaming_module.v2_token_chunks
validate_streaming_options = _streaming_module.validate_streaming_options
PipelineStop = _streaming_module.PipelineStop
stream_v5_windows = _streaming_module.stream_v5_windows
StreamingWindowSchedule = _streaming_module.StreamingWindowSchedule


class WindowScheduleTests(unittest.TestCase):
    def test_progressive_windows_preserve_playback_lead_on_a_slower_device(self):
        clock = [0.0]
        with patch.object(_streaming_module.time, "perf_counter", side_effect=lambda: clock[0]):
            schedule = StreamingWindowSchedule(200, 541)
            self.assertEqual(schedule.choose(2000), 200)
            clock[0] += 1.2
            schedule.observe(200, 94080, 48000)
            sizes = []
            for _ in range(8):
                size = schedule.choose(2000)
                sizes.append(size)
                elapsed = 0.2 + 0.5 * size / 100
                available = schedule.audio_seconds - (clock[0]-schedule.first_pcm_at)
                self.assertLess(elapsed, available)
                clock[0] += elapsed
                schedule.observe(size, size*480, 48000)
            self.assertEqual(sizes[0], 200)
            self.assertIn(300, sizes)
            self.assertIn(541, sizes)
            # Immediately switching to the largest window would stall despite RTF < 1.
            self.assertGreater(0.2+0.5*5.41, 1.96)
            self.assertLess(0.2+0.5*5.41, 5.41)

    def test_backpressure_and_slower_observations_reduce_window_growth(self):
        clock = [0.0]
        with patch.object(_streaming_module.time, "perf_counter", side_effect=lambda: clock[0]):
            schedule = StreamingWindowSchedule(16, 64)
            clock[0] = 0.05
            schedule.observe(16, 17920, 32000)
            self.assertEqual(schedule.choose(1000), 24)
            clock[0] += 0.5
            self.assertEqual(schedule.choose(1000), 16)
            schedule.observe(16, 20480, 32000)
            self.assertEqual(schedule.choose(1000), 16)

    def test_short_available_tail_is_flushed_without_padding(self):
        schedule = StreamingWindowSchedule(200, 541)
        self.assertEqual(schedule.choose(17), 17)


class ReferenceTests(unittest.TestCase):
    def test_reference_features_align_before_tail_crop(self):
        features = torch.arange(8).reshape(1, 1, -1).float()
        mel = torch.arange(5).reshape(1, 1, -1).float()
        cache = dict(refer_spec=[(torch.zeros(1, 2, 4), None)],
                     prompt_semantic=torch.ones(4, dtype=torch.long), phones=[1, 2],
                     bert_features=torch.zeros(1024, 2), raw_audio=torch.zeros(1, 10), raw_sr=32000)
        pipeline = SimpleNamespace(prompt_cache=cache, precision=torch.float32,
            configs=SimpleNamespace(device=torch.device("cpu")),
            vits_model=SimpleNamespace(decode_encp=lambda *args: (features, None)))
        stub = SimpleNamespace(resample=None, mel_fn_v4=lambda _: mel, norm_spec=lambda x: x)
        with patch.dict(sys.modules, {"GPT_SoVITS.TTS_infer_pack.TTS": stub}):
            reference = _streaming_module.v5_reference(pipeline)
        torch.testing.assert_close(reference[4], features[..., :5])
        torch.testing.assert_close(reference[6], mel)


class AdapterTests(unittest.TestCase):
    def adapter(self, partials=(), final=None, error=None):
        handle = T2SRequestHandle("test")
        for partial in partials:
            handle.queue.put(T2SStreamResponse("test", 0, [torch.tensor(partial)]))
        if final is not None or error is not None:
            handle.queue.put(T2SStreamResponse("test", -1, None, finished=True,
                result=T2SResult(result=[torch.tensor(final)] if final is not None else None,
                                 exception=error), exception=error))
            handle.done = True
        a = AccelInference.__new__(AccelInference)
        a.device = torch.device("cpu")
        a._mode_lock = threading.RLock()
        a._request_ids = count()
        a.graph_enabled = False
        a.decoder = SimpleNamespace(max_seq_length=1024)
        requests = []
        def add(req):
            requests.append(req)
            return handle
        def cancel(h):
            h.cancelled.set()
            h.done = True
        a.engine = SimpleNamespace(add_request=add, cancel_request=cancel, _stop_event=threading.Event())
        a.prepare = lambda *args: True
        return a, handle, requests

    def stream(self, a, **kw):
        return a.infer_stream(torch.ones(4, dtype=torch.long), torch.ones(1, 2, dtype=torch.long),
                             torch.zeros(1024, 4), max_decode_steps=20, **kw)

    def test_decoder_graph_preparation_keeps_inference_context_consistent(self):
        a, _, _ = self.adapter(final=[1, 2])
        a.engine = None
        a.backend = "fake"
        a.dtype = torch.float32
        a.graph_failed = False
        seen = []
        def engine(*args, **kwargs):
            seen.append(torch.is_inference_mode_enabled())
            return SimpleNamespace(model_runner=SimpleNamespace(graph_applicable=True, graph_error=None))
        with patch("GPT_SoVITS.Accel.adapter.resolve_acceleration", return_value=("fake", True)), \
                patch("GPT_SoVITS.Accel.adapter.T2SEngine", engine):
            self.assertTrue(AccelInference.prepare(a))
        self.assertEqual(seen, [True])
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_partial_deltas_and_complete_final_are_not_duplicated(self):
        a, _, requests = self.adapter([[1, 2], [3, 4]], [1, 2, 3, 4, 5])
        updates = list(self.stream(a))
        self.assertEqual([t.tolist() for t, _ in updates], [[1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5]])
        self.assertEqual([done for _, done in updates], [False, False, True])
        self.assertEqual(requests[0].max_decode_steps, 20)

    def test_final_prefix_mismatch_is_rejected(self):
        a, _, _ = self.adapter([[1, 2]], [3, 4, 5])
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            list(self.stream(a))

    def test_limit_is_an_error_instead_of_successful_truncation(self):
        a, _, _ = self.adapter(final=list(range(20)))
        with self.assertRaisesRegex(ValueError, "token limit"):
            list(self.stream(a))

    def test_error_is_propagated(self):
        a, _, _ = self.adapter(error=RuntimeError("decoder failed"))
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            list(self.stream(a))

    def test_close_on_another_thread_cancels_without_rlock_owner_error(self):
        a, handle, _ = self.adapter([[1, 2]])
        iterator = self.stream(a)
        def lock_is_available():
            acquired = a._mode_lock.acquire(False)
            if acquired:
                a._mode_lock.release()
            return acquired
        with ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(max_workers=1) as second:
            first.submit(next, iterator).result()
            second.submit(iterator.close).result()
            self.assertTrue(second.submit(lock_is_available).result())
        self.assertTrue(handle.cancelled.is_set())

    def test_stop_interrupts_an_empty_queue(self):
        a, handle, _ = self.adapter()
        stop = threading.Event()
        iterator = self.stream(a, stop=stop)
        timer = threading.Timer(0.03, stop.set)
        timer.start()
        try:
            self.assertEqual(list(iterator), [])
        finally:
            timer.join()
        self.assertTrue(handle.cancelled.is_set())

    def test_hybrid_heartbeat_rechecks_waiting_prefix_and_still_cancels(self):
        a,handle,_ = self.adapter([[1,2]])
        stop = threading.Event()
        iterator = self.stream(a,stop=stop,stream_state={"poll_partial":True})
        first,done = next(iterator)
        heartbeat,done = next(iterator)
        self.assertFalse(done)
        self.assertTrue(torch.equal(first,heartbeat))
        stop.set()
        self.assertEqual(list(iterator),[])
        self.assertTrue(handle.cancelled.is_set())

    def test_stop_during_heartbeat_wait_does_not_yield_a_stale_prefix(self):
        a,handle,_ = self.adapter([[1,2]])
        stop = threading.Event()
        iterator = self.stream(a,stop=stop,stream_state={"poll_partial":True})
        next(iterator)
        timer = threading.Timer(0.01,stop.set);timer.start()
        try:self.assertEqual(list(iterator),[])
        finally:timer.join()
        self.assertTrue(handle.cancelled.is_set())


class AcousticTests(unittest.TestCase):
    def fixture(self, frames):
        calls, decoded, progress = [], [], []
        class Diffusion:
            def inference(self, mu, lengths, prompt, steps, inference_cfg_rate, generator):
                calls.append(mu.shape[1])
                torch.randn(1, generator=generator)
                return mu.transpose(1, 2)
        def enc(tokens, *args):
            n = tokens.numel() * 4
            decoded.append(n)
            return torch.arange(n).float().view(1, 1, -1) / 2000, None
        w = SimpleNamespace(device="cpu", vq_model=SimpleNamespace(cfm=Diffusion(), decode_encp=enc),
            denorm_spec=lambda mel: mel, hifigan_model=lambda mel: mel.repeat_interleave(480, -1))
        ref = torch.zeros(1, 1, 50)
        reference = (None, None, None, None, ref, None, ref)
        req = dict(streaming_chunk_seconds=2, streaming_lookahead_seconds=.12,
            gpt_streaming_lookahead_seconds=.5, streaming_overlap_seconds=.04,
            speed_factor=1, sample_steps=16, cfg_rate=0)
        def updates():
            for n in range(16, frames//4, 16):
                progress.append((n, False))
                yield torch.arange(n), False
            progress.append((frames//4, True))
            yield torch.arange(frames//4), True
        return w, reference, req, updates(), calls, decoded, progress

    def stream(self, w, updates, phones, reference, req, stop):
        return stream_v5_windows(w.vq_model, w.hifigan_model, w.denorm_spec,
                                 w.device, updates, phones, reference, req, stop)

    def test_prefixes_preserve_every_frame_in_order_and_flush_tail(self):
        for frames in (4, 196, 200, 204, 400, 404, 856):
            with self.subTest(frames=frames):
                w, ref, req, updates, calls, decoded, progress = self.fixture(frames)
                output = list(self.stream(w, updates, [], ref, req, threading.Event()))
                pcm = np.concatenate(output)
                expected = (torch.arange(frames).repeat_interleave(480).float()/2000*32767).to(torch.int16).numpy()
                self.assertEqual(pcm.size, frames*480)
                np.testing.assert_allclose(pcm.astype(float), expected.astype(float), atol=1)
                self.assertTrue(all(n <= 1000 for n in calls))

    def test_first_window_precedes_eos_and_skips_unproductive_prefixes(self):
        w, ref, req, updates, calls, decoded, progress = self.fixture(856)
        iterator = self.stream(w, updates, [], ref, req, threading.Event())
        first = next(iterator)
        self.assertEqual(first.size, 196*480)
        self.assertFalse(progress[-1][1])
        self.assertEqual(decoded, [320])
        iterator.close()

    def test_cancel_does_not_generate_another_window(self):
        w, ref, req, updates, calls, _, _ = self.fixture(856)
        stop = threading.Event()
        iterator = self.stream(w, updates, [], ref, req, stop)
        next(iterator)
        stop.set()
        self.assertEqual(list(iterator), [])
        self.assertEqual(len(calls), 1)

    def test_missing_eos_is_rejected(self):
        w, ref, req, _, _, _, _ = self.fixture(856)
        updates = iter([(torch.arange(80), False)])
        with self.assertRaisesRegex(RuntimeError, "without a final"):
            list(self.stream(w, updates, [], ref, req, threading.Event()))


    def test_hybrid_preserves_short_and_exact_boundary_frames(self):
        for frames in (4,196,200,204,400,404,856,1600):
            w,ref,req,updates,calls,_,_ = self.fixture(frames)
            req["streaming_window_mode"] = "hybrid"
            pcm = np.concatenate(list(self.stream(w,updates,[],ref,req,threading.Event())))
            expected = (torch.arange(frames).repeat_interleave(480).float()/2000*32767).to(torch.int16).numpy()
            np.testing.assert_allclose(pcm.astype(float),expected.astype(float),atol=1)
            self.assertTrue(all(n<=1000 for n in calls))

    def test_hybrid_gradually_grows_without_receiver_underrun(self):
        w,ref,req,_,calls,_,_ = self.fixture(2200)
        req["streaming_window_mode"] = "hybrid"
        clock = [0.0]
        original = w.vq_model.cfm.inference
        spans = []
        def inference(mu,lengths,prompt,*args,**kwargs):
            span = mu.shape[1]-prompt.shape[-1]
            spans.append(span)
            clock[0] += 0.2+0.5*span/100
            return original(mu,lengths,prompt,*args,**kwargs)
        w.vq_model.cfm.inference = inference
        updates = iter([(torch.arange(80),False),(torch.arange(400),False),(torch.arange(550),True)])
        with patch.object(_streaming_module.time,"perf_counter",side_effect=lambda:clock[0]):
            chunks = [];first = None;playable = 0.0
            for pcm in self.stream(w,updates,[],ref,req,threading.Event()):
                if first is None:first=clock[0]
                self.assertLessEqual(clock[0]-first,playable+1e-6)
                playable += pcm.size/48000
                chunks.append(pcm)
        self.assertEqual(sum(x.size for x in chunks),2200*480)
        self.assertIn(300,spans)
        self.assertTrue(any(n>400 for n in spans))
        self.assertTrue(all(n<=1000 for n in calls))

    def test_waiting_for_a_larger_window_falls_back_on_heartbeat(self):
        w,ref,req,_,_,_,_ = self.fixture(1000)
        req["streaming_window_mode"] = "hybrid"
        clock = [0.0];spans=[]
        original = w.vq_model.cfm.inference
        def inference(mu,lengths,prompt,*args,**kwargs):
            span=mu.shape[1]-prompt.shape[-1];spans.append(span)
            clock[0] += 0.2+0.5*span/100
            return original(mu,lengths,prompt,*args,**kwargs)
        w.vq_model.cfm.inference=inference
        def updates():
            yield torch.arange(80),False
            yield torch.arange(128),False
            yield torch.arange(184),False
            clock[0] += 0.5
            yield torch.arange(184),False
            yield torch.arange(250),True
        with patch.object(_streaming_module.time,"perf_counter",side_effect=lambda:clock[0]):
            first=None;playable=0.
            for pcm in self.stream(w,updates(),[],ref,req,threading.Event()):
                if first is None:first=clock[0]
                self.assertLessEqual(clock[0]-first,playable+1e-6)
                playable+=pcm.size/48000
        self.assertEqual(spans[:3],[212,200,200])


class RNGTests(unittest.TestCase):
    def test_cfm_generator_preserves_global_rng_and_is_reproducible(self):
        class Estimator(torch.nn.Module):
            def forward(self, x, *args, **kwargs):
                return torch.zeros_like(x).transpose(1, 2), None, None
        model = CFM(1, Estimator())
        mu, prompt, lengths = torch.zeros(1, 40, 1), torch.zeros(1, 1, 10), torch.tensor([40])
        torch.manual_seed(123)
        original = torch.random.get_rng_state().clone()
        def generate():
            return model.inference(mu, lengths, prompt, 2,
                generator=torch.Generator().manual_seed(456))
        first, second = generate(), generate()
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(original, torch.random.get_rng_state()))
        model.inference(mu, lengths, prompt, 2)
        self.assertFalse(torch.equal(original, torch.random.get_rng_state()))


class CancelTests(unittest.TestCase):
    def test_owner_thread_releases_waiting_and_bound_sessions(self):
        for waiting in (True, False):
            e = T2SEngine.__new__(T2SEngine)
            e._stop_event = threading.Event()
            e._loop_thread = None
            handle = T2SRequestHandle("test")
            handle.cancelled.set()
            session = SimpleNamespace(request_id="test")
            e.handles = {"test":handle}
            e.waiting_bind = [session] if waiting else []
            bound = [] if waiting else [session]
            with torch.inference_mode():
                inference_buffer = torch.ones(1)
            def unbind(s):
                inference_buffer[0] = 0
                bound.remove(s)
            e.model_runner = SimpleNamespace(bound_sessions=bound, unbind_session=unbind)
            def step():
                e._stop_event.set()
                return []
            e.model_runner.decode_step = step
            e._drain_new_requests = lambda: None
            finalized = []
            def finalize(s, exc, cancelled=False):
                finalized.append(cancelled)
                e._stop_event.set()
            e._finalize_session = finalize
            e._background_loop()
            self.assertEqual(finalized, [True])
            self.assertEqual(e.waiting_bind, [])
            self.assertEqual(bound, [])


class TokenChunksTests(unittest.TestCase):
    def updates(self, sizes):
        for i, n in enumerate(sizes):
            yield torch.arange(n), i == len(sizes)-1

    def flattened(self, outputs):
        tensors = [tokens.reshape(-1) for tokens, _ in outputs if tokens is not None]
        return torch.cat(tensors).tolist() if tensors else []

    def test_fixed_delta_tail_and_exact_boundary_eos(self):
        for sizes in [(16, 32, 37), (16, 32, 32), (5,), (16, 64, 68)]:
            with self.subTest(sizes=sizes):
                out = list(v2_token_chunks(self.updates(sizes), 16))
                self.assertEqual(self.flattened(out), list(range(sizes[-1])))
                self.assertTrue(out[-1][1])
                self.assertEqual(sum(final for _, final in out), 1)

    def test_silence_pair_emits_only_through_the_cut(self):
        scores = torch.full((80,), -1.)
        scores[15:17] = 1.
        scores[35:37] = 1.
        out = list(v2_token_chunks(self.updates([24, 48, 60]), 16, scores))
        self.assertEqual(self.flattened(out), list(range(60)))
        self.assertEqual(out[0][0].numel(), 16)
        self.assertTrue(out[-1][1])

    def test_no_silence_waits_for_eos(self):
        out = list(v2_token_chunks(self.updates([16, 32, 40]), 16, torch.full((80,), -1.)))
        self.assertEqual(len(out), 1)
        self.assertEqual(self.flattened(out), list(range(40)))

    def test_single_last_token_is_not_a_silence_pair(self):
        scores = torch.full((80,), -1.)
        scores[23] = 1.
        out = list(v2_token_chunks(self.updates([24, 26]), 16, scores))
        self.assertEqual(len(out), 1)
        self.assertEqual(self.flattened(out), list(range(26)))

    def test_closing_consumer_closes_upstream(self):
        closed = []
        def updates():
            try:
                yield torch.arange(16), False
                yield torch.arange(32), True
            finally:
                closed.append(True)
        chunks = v2_token_chunks(updates(), 16)
        next(chunks)
        chunks.close()
        self.assertEqual(closed, [True])

    def test_decreasing_prefix_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "backwards"):
            list(v2_token_chunks(self.updates([32, 16]), 16))


    def test_hybrid_eos_flush_is_bounded_and_silence_wait_does_not_stall(self):
        clock = [0.0]
        with patch.object(_streaming_module.time,"perf_counter",side_effect=lambda:clock[0]):
            schedule = StreamingWindowSchedule(16,64)
            output = [];playable = 0.0;first = None
            iterator = v2_token_chunks(self.updates([32,200]),16,torch.full((220,),-1.),schedule=schedule)
            for tokens,finished in iterator:
                if tokens is None:continue
                n = tokens.numel();clock[0] += 0.01+n*0.005
                if first is None:first=clock[0]
                self.assertLessEqual(clock[0]-first,playable+1e-6)
                samples = int((n-(2 if not output else 0))*1280)
                schedule.observe(n,samples,32000)
                playable += samples/32000
                output.append((tokens,finished))
            self.assertEqual(self.flattened(output),list(range(200)))
            self.assertTrue(output[-1][1])
            self.assertEqual(sum(final for _,final in output),1)
            self.assertEqual(output[0][0].numel(),16)
            self.assertTrue(any(tokens.numel()>16 for tokens,_ in output))
            self.assertTrue(all(tokens.numel()<=64 for tokens,_ in output))


class ValidationTests(unittest.TestCase):
    def req(self, **changes):
        req = dict(gpt_streaming=True, streaming_mode=2, text_split_method="cut0", batch_size=1, prompt_text="hello")
        req.update(changes)
        return req

    def test_old_requests_are_not_revalidated(self):
        validate_streaming_options(dict(gpt_streaming=False, streaming_mode=0), "v1", api_mode=True)

    def test_modes_versions_and_finite_window_limits(self):
        for version in ["v5turbo", "v5dev", "v2ProPlus"]:
            validate_streaming_options(self.req(), version, api_mode=True)
        for patch in [dict(streaming_mode=True), dict(streaming_mode=0), dict(streaming_mode=2.0), dict(batch_size=2),
                      dict(text_split_method="cut5"), dict(media_type="aac"),
                      dict(streaming_chunk_seconds=float("nan")), dict(speed_factor=0),
                      dict(cfg_rate=float("inf")), dict(overlap_length=0)]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_streaming_options(self.req(**patch), "v5turbo", api_mode=True)
        with self.assertRaises(ValueError):
            validate_streaming_options(self.req(aux_ref_audio_paths=["ref.wav"]), "v5turbo", api_mode=True)
        validate_streaming_options(self.req(aux_ref_audio_paths=["ref.wav"]), "v2ProPlus", api_mode=True)
        with self.assertRaises(ValueError):
            validate_streaming_options(self.req(), "v4", api_mode=True)

    def test_pipeline_stop_and_disconnect_share_a_stop_signal(self):
        pipeline = SimpleNamespace(stop_flag=False)
        stop = PipelineStop(pipeline)
        self.assertFalse(stop.is_set())
        pipeline.stop_flag = True
        self.assertTrue(stop.is_set())
        pipeline.stop_flag = False
        stop.set()
        self.assertTrue(stop.is_set())


class AcousticRNGTests(unittest.TestCase):
    def fixture(self):
        shape = (1, 2, 32)
        model = SimpleNamespace(
            version="v2", is_v2pro=False, semantic_frame_rate="25hz",
            quantizer=SimpleNamespace(decode=lambda codes: torch.zeros(1, 2, codes.shape[-1])),
            enc_p=lambda *args, **kwargs: (None, torch.zeros(shape), torch.zeros(shape),
                torch.ones(shape), None, None),
            flow=lambda z, *args, **kwargs: z,
            dec=lambda z, **kwargs: z)
        return model

    def decode(self, model, **kw):
        return SynthesizerTrn.decode_streaming(model, torch.zeros(1, 1, 16, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long), None, **kw)[0]

    def test_private_v2_noise_does_not_advance_ar_rng(self):
        model = self.fixture()
        torch.manual_seed(29)
        state = torch.random.get_rng_state().clone()
        first = self.decode(model, generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(first, self.decode(model, generator=torch.Generator().manual_seed(7))))
        self.assertGreater(first.abs().sum().item(), 0)

    def test_default_v2_noise_retains_global_rng_behavior(self):
        torch.manual_seed(29)
        state = torch.random.get_rng_state().clone()
        self.decode(self.fixture())
        self.assertFalse(torch.equal(state, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
