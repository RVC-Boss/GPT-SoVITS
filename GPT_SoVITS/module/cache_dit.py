"""cache-dit / DBCache (three-segment block caching) for the DiT/CFM sampler.

Follows vipshop/cache-dit (`CachedBlocks_Pattern_Base.forward`), adapted to the way this repo
samples: the block stack is split into three segments, and the middle one is skipped when the
front residual barely moved.

    front : trunk(x_in) -> blocks[0:Fn] -> x_F        (every step)
            R    = x_F - trunk(x_in)                  Fn residual, the decision signal
            fd   = sum|R - R_prev| / (sum|R_prev| + eps)          (relative -> scale free)
    hit   : x_mid = x_F + MnResidual_prev, then head only (norm_out -> proj_out)
    miss  : x_F -> blocks[Fn:DEPTH] -> x_M;  MnResidual = x_M - x_F;  R -> R_prev
    head  : x_mid -> norm_out -> proj_out -> vel              (every step, hit included)

A hit additionally requires `can_cache`: not in warmup, `fd < threshold`, and the three gates
`max_cached` (cached steps per sampling run), `max_cont` (consecutive hits) and `max_accum`
(accumulated residual drift).

CFG. This repo runs the conditional and the unconditional branch as **two separate forward
passes** (`CFM.inference`), so each branch owns its own `R_prev`, `MnResidual`, hit/miss
counters and gate counters -- the equivalent of cache-dit's `enable_separate_cfg` plus
`cfg_diff_compute_separate`.  Nothing is shared between the branches.

Everything here is inert unless a `DBCacheConfig` with `fn > 0` is handed to `DiT.forward`;
`cache_dit="off"` keeps the sampler on the identical code path it had before this change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch

__all__ = [
    "DBCacheConfig",
    "DBCacheState",
    "resolve_fn",
    "config_from_params",
    "MODES",
]

MODES = ("off", "auto", "f4", "f8", "f16")

# Defaults of the cache-dit reference recipe, and the `auto` policy used by the HTTP API
# (F8 from 16 sampling steps on, off below that).
DEFAULT_FN = 8
DEFAULT_THRESHOLD = 0.02
DEFAULT_WARMUP = 8
DEFAULT_DOWNSAMPLE = 4
AUTO_FN = 8
AUTO_MIN_STEPS = 16

_EPS = 1e-6


def resolve_fn(mode, steps=32, depth=None):
    """`off|auto|f4|f8|f16` (or a plain integer) -> number of front blocks."""
    if mode is None:
        return 0
    if isinstance(mode, bool):
        mode = int(mode)
    if isinstance(mode, (int, float)):
        fn = int(mode)
    else:
        text = str(mode).strip().lower()
        if text in ("", "off", "none", "0", "false", "no"):
            fn = 0
        elif text in ("on", "true", "yes"):
            fn = DEFAULT_FN
        elif text == "auto":
            fn = AUTO_FN if int(steps) >= AUTO_MIN_STEPS else 0
        elif text.startswith("f") and text[1:].isdigit():
            fn = int(text[1:])
        elif text.isdigit():
            fn = int(text)
        else:
            raise ValueError("unknown cache_dit mode %r (expected off|auto|f4|f8|f16)" % (mode,))
    if depth is not None:
        fn = max(0, min(fn, int(depth) - 1))
    return fn


@dataclass
class DBCacheConfig:
    """cache-dit knobs. `fn == 0` means off (the default)."""

    mode: str = "off"
    fn: int = 0
    threshold: float = DEFAULT_THRESHOLD
    warmup: int = DEFAULT_WARMUP
    downsample: int = DEFAULT_DOWNSAMPLE
    warmup_interval: int = 1
    max_cached: int = -1
    max_cont: int = -1
    max_accum: float = 0.0
    # fd is reduced over the whole sequence by default: the prompt/reference frames carry a
    # nearly constant residual, so they dilute the ratio and threshold 0.02 stays usable. Set
    # to True to reduce over the target frames only -- a ~3x stricter signal on this model, so
    # thresholds are not transferable between the two conventions.
    fd_ignore_prompt: bool = False
    trace: bool = False
    # expert override: allow cache-dit below 16 sampling steps (see docs in
    # `DBCacheMixin.dbcache_config`: too few steps gain nothing and the quality risk is real)
    force: bool = False

    @property
    def enabled(self) -> bool:
        return self.fn > 0

    def clamped(self, depth: int | None = None) -> "DBCacheConfig":
        if depth is None:
            return self
        return replace(self, fn=max(0, min(int(self.fn), int(depth) - 1)))

    def __str__(self) -> str:  # for logs
        if not self.enabled:
            return "off"
        return ("%s(fn=%d thr=%.3g warmup=%d ds=%d mc=%d mcont=%d macc=%.3g)"
                % (self.mode, self.fn, self.threshold, self.warmup, self.downsample,
                   self.max_cached, self.max_cont, self.max_accum))


def _get(params, key, env, default, cast):
    """request dict first (API field names), then env, then default."""
    if params is not None:
        for k in (key,):
            if k in params and params[k] is not None and params[k] != "":
                try:
                    return cast(params[k])
                except (TypeError, ValueError):
                    pass
    val = os.environ.get(env)
    if val is not None and val != "":
        try:
            return cast(val)
        except (TypeError, ValueError):
            pass
    return default


def config_from_params(params=None, steps=32, depth=None) -> DBCacheConfig:
    """Build the config from a request dict / env vars.

    Field names match the HTTP API of our C++ server:
        cache_dit (alias dbcache) = off|auto|f4|f8|f16
        dbcache_fn, dbcache_threshold, dbcache_warmup, dbcache_downsample
        dbcache_warmup_interval, dbcache_max_cached, dbcache_max_cont, dbcache_max_accum
    Env fallbacks: GSV_DBCACHE / GSV_DBCACHE_FN / _THRESHOLD / _WARMUP / _DOWNSAMPLE / ...
    """
    p = dict(params or {})
    mode = p.get("cache_dit", p.get("dbcache", os.environ.get("GSV_DBCACHE", "off")))
    mode = "off" if mode is None else str(mode).strip().lower()

    cfg = DBCacheConfig(mode=mode)
    cfg.threshold = _get(p, "dbcache_threshold", "GSV_DBCACHE_THRESHOLD", cfg.threshold, float)
    cfg.warmup = _get(p, "dbcache_warmup", "GSV_DBCACHE_WARMUP", cfg.warmup, int)
    cfg.downsample = max(1, _get(p, "dbcache_downsample", "GSV_DBCACHE_DOWNSAMPLE", cfg.downsample, int))
    cfg.warmup_interval = max(1, _get(p, "dbcache_warmup_interval", "GSV_DBCACHE_WARMUP_INTERVAL",
                                      cfg.warmup_interval, int))
    cfg.max_cached = _get(p, "dbcache_max_cached", "GSV_DBCACHE_MAX_CACHED", cfg.max_cached, int)
    cfg.max_cont = _get(p, "dbcache_max_cont", "GSV_DBCACHE_MAX_CONT", cfg.max_cont, int)
    cfg.max_accum = _get(p, "dbcache_max_accum", "GSV_DBCACHE_MAX_ACCUM", cfg.max_accum, float)
    cfg.fd_ignore_prompt = bool(_get(p, "dbcache_fd_ignore_prompt", "GSV_DBCACHE_FD_IGNORE_PROMPT",
                                     int(cfg.fd_ignore_prompt), int))
    cfg.trace = bool(_get(p, "dbcache_trace", "GSV_DBCACHE_TRACE", 0, int))
    cfg.force = bool(_get(p, "dbcache_force", "GSV_DBCACHE_FORCE", 0, int))

    # an explicit fn wins over the mode
    if p.get("dbcache_fn") not in (None, ""):
        cfg.fn = int(p["dbcache_fn"])
        cfg.mode = "custom"
    elif os.environ.get("GSV_DBCACHE_FN"):
        cfg.fn = int(os.environ["GSV_DBCACHE_FN"])
        cfg.mode = "custom"
    else:
        cfg.fn = resolve_fn(mode, steps)
    if depth is not None:
        cfg.fn = max(0, min(cfg.fn, int(depth) - 1))
    return cfg


class DBCacheMixin:
    """The CFM-side plumbing (`set_dbcache` / per-run state / hit counters).

    Mixed into `module.models.CFM` and `module.v5_inference.CFMV5`; both expose the DiT as
    `self.estimator`.  Defaults are class attributes, so `cache off` needs no `__init__` change
    and the object layout of an existing checkpoint-compatible model is untouched.
    """

    dbcache_spec = None            # DBCacheConfig or request dict of dbcache_* fields
    last_dbcache_stats = None      # stats of the most recent inference() call
    _dbcache_cum = None            # accumulated over all calls (reset by dbcache_stats)

    # ------------------------------------------------------------------ config
    @property
    def cache_depth(self):
        return int(getattr(self.estimator, "depth", 0) or 0)

    def set_dbcache(self, spec=None, **kw):
        """Enable/disable cache-dit for the next sampling runs.

        `spec` is either a `module.cache_dit.DBCacheConfig` or a dict using the C++ server's
        field names (`cache_dit`/`dbcache` + `dbcache_fn|threshold|warmup|downsample|...`).
        Nothing is applied retroactively: `inference()` resolves it per call.
        """
        if isinstance(spec, DBCacheConfig):
            self.dbcache_spec = spec
        else:
            merged = dict(spec or {})
            merged.update(kw)
            cfg = merged.pop("config", None)
            self.dbcache_spec = cfg if isinstance(cfg, DBCacheConfig) else merged
        return self.dbcache_spec

    def dbcache_config(self, steps):
        """Resolve the spec against the step count of the run that is about to start.

        With no explicit spec the env switch (`GSV_DBCACHE=off|auto|f4|f8|f16`) is honoured,
        so every entry point (WebUI, api.py, api_v2.py, CLI, Colab) can enable cache-dit
        without extra plumbing.  Below 16 sampling steps (the 4-step v5turbo) it stays off:
        the measured gain is negative and the quality is unstable there.
        """
        depth = self.cache_depth or None
        spec = self.dbcache_spec
        if isinstance(spec, DBCacheConfig):
            fn = spec.fn if spec.mode in ("custom", None) else resolve_fn(spec.mode, steps, depth)
            cfg = DBCacheConfig(**{**spec.__dict__,
                                   "fn": fn if depth is None else min(fn, depth - 1)})
        else:
            cfg = config_from_params(spec, steps, depth)
        if cfg.enabled and int(steps) < AUTO_MIN_STEPS and not cfg.force:
            cfg.fn = 0
        return cfg

    # ------------------------------------------------------------------ per run
    def _dbcache_begin(self, steps, seq_len, prompt_len, x_lens, device):
        """Fresh state per sampling run (new request / new chunk)."""
        cfg = self.dbcache_config(steps)
        if not cfg.enabled:
            self.last_dbcache_stats = None
            return None
        return DBCacheState(cfg, self.cache_depth).begin(seq_len, prompt_len, x_lens, device)

    def _dbcache_end(self, state):
        if state is None:
            return
        st = state.stats()
        self.last_dbcache_stats = st
        cum = self._dbcache_cum
        if cum is None:
            cum = self._dbcache_cum = {"calls": 0, "steps": 0, "hits_pos": 0, "hits_neg": 0,
                                       "misses_pos": 0, "misses_neg": 0}
        cum["calls"] += 1
        for k in ("steps", "hits_pos", "hits_neg", "misses_pos", "misses_neg"):
            cum[k] = cum.get(k, 0) + st[k]
        cum["hits"] = cum["hits_pos"] + cum["hits_neg"]
        cum["misses"] = cum["misses_pos"] + cum["misses_neg"]
        cum["total"] = cum["hits"] + cum["misses"]
        cum["hit_rate"] = cum["hits"] / cum["total"] if cum["total"] else 0.0
        cum["fn"] = st.get("fn")
        cum["threshold"] = st.get("threshold")
        cum["last"] = st

    def dbcache_stats(self, reset=True):
        """Hit/miss counters accumulated since the last reset (None when cache-dit is off)."""
        st = self._dbcache_cum
        if reset:
            self._dbcache_cum = None
        return st


class _Branch:
    """Per-CFG-branch bookkeeping."""

    __slots__ = ("valid", "r_prev", "r_cur", "m_res", "hits", "misses",
                 "cached_steps", "cont", "acc", "last_fd")

    def __init__(self):
        self.valid = False
        self.r_prev = None
        self.r_cur = None
        self.m_res = None
        self.hits = 0
        self.misses = 0
        self.cached_steps = 0
        self.cont = 0
        self.acc = 0.0
        self.last_fd = 0.0

    def reset(self):
        self.__init__()


class DBCacheState:
    """State of one sampling run (`CFM.inference` = one run = one new request / chunk)."""

    def __init__(self, config: DBCacheConfig, depth: int, trace: bool | None = None):
        self.cfg = config
        self.depth = int(depth)
        self.trace = config.trace if trace is None else trace
        self.step = 0
        self.n_steps = 0
        self.branches = [_Branch(), _Branch()]
        self.ds = 1
        self.note = ""  # free-form tag for logs

    # ------------------------------------------------------------------ lifecycle
    def begin(self, seq_len: int, prompt_len: int, x_lens=None, device=None):
        """Reset for a new sampling run. Called once per `CFM.inference`."""
        for b in self.branches:
            b.reset()
        self.step = 0
        self.n_steps = 0
        ds = int(self.cfg.downsample) if self.cfg.downsample else 1
        self.ds = ds if ds >= 1 and seq_len % ds == 0 else 1
        self._build_frame_mask(seq_len, prompt_len, x_lens, device)
        return self

    def _build_frame_mask(self, seq_len, prompt_len, x_lens, device):
        """Frames used for fd: skip the prompt/reference rows and the padding."""
        self.mask = None
        if not self.cfg.fd_ignore_prompt:
            return
        idx = torch.arange(seq_len, device=device)
        mask = idx.unsqueeze(0) >= int(max(0, prompt_len))
        if x_lens is not None:
            lens = torch.as_tensor(x_lens, device=device).reshape(-1, 1)
            if lens.shape[0] == 1 and seq_len >= int(lens.max()):
                mask = mask & (idx.unsqueeze(0) < lens)
        mask = mask.to(torch.float32)
        if self.ds > 1:
            mask = mask[:, :: self.ds]
        mask = mask.unsqueeze(-1)          # (b, n/ds, 1)
        if float(mask.sum()) <= 0.0:       # degenerate (all-prompt) -> fall back to everything
            mask = None
        self.mask = mask

    def step_begin(self, step: int):
        """`db_set_step(j)`: current diffusion step, drives warmup and the trace."""
        self.step = int(step)
        self.n_steps += 1

    def in_warmup(self) -> bool:
        if self.step >= self.cfg.warmup:
            return False
        return (self.step % max(1, self.cfg.warmup_interval)) == 0

    # ------------------------------------------------------------------ per-step ops
    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    @property
    def fn(self) -> int:
        return self.cfg.fn

    def branch(self, index: int) -> _Branch:
        return self.branches[0 if index == 0 else 1]

    def front_update(self, index: int, x_f, trunk) -> float:
        """Store x_F / R, update R_prev-based similarity, return `fd` (one host sync)."""
        br = self.branch(index)
        r = x_f.float() - trunk.float()                  # Fn residual; fp32 keeps fd stable
        if self.ds > 1:
            r = r[:, :: self.ds, :]
        r_prev = br.r_prev
        if r_prev is None or r_prev.shape != r.shape:
            r_prev = torch.zeros_like(r)
        num = (r - r_prev).abs()
        den = r_prev.abs()
        if self.mask is not None and self.mask.shape[0] == num.shape[0] \
                and self.mask.shape[1] == num.shape[1]:
            num = num * self.mask
            den = den * self.mask
        num = num.sum(dim=(1, 2))
        den = den.sum(dim=(1, 2)) + _EPS
        fd = float((num / den).max().item())            # the only device->host scalar per step
        br.r_cur = r
        br.last_fd = fd
        return fd

    def can_cache(self, index: int, fd: float) -> bool:
        br = self.branch(index)
        cfg = self.cfg
        hit = False
        if br.valid and cfg.threshold > 0.0 and not self.in_warmup():
            ok = fd < cfg.threshold
            if ok and cfg.max_cached >= 0 and br.cached_steps >= cfg.max_cached:
                ok = False
            if ok and cfg.max_cont >= 0 and br.cont >= cfg.max_cont:
                ok = False
                br.cont = 0
            if ok and cfg.max_accum > 0.0 and br.acc >= cfg.max_accum:
                ok = False
            hit = ok
        if fd > 0.0:
            br.acc += fd
        return hit

    def mid_residual(self, index: int):
        return self.branch(index).m_res

    def on_hit(self, index: int):
        br = self.branch(index)
        br.hits += 1
        br.cached_steps += 1
        br.cont += 1

    def on_miss(self, index: int, x_m, x_f):
        br = self.branch(index)
        br.misses += 1
        br.valid = True
        br.cont = 0
        br.m_res = (x_m - x_f)
        if br.r_cur is not None:
            if br.r_prev is None or br.r_prev.shape != br.r_cur.shape:
                br.r_prev = br.r_cur.clone()
            else:
                br.r_prev.copy_(br.r_cur)

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict:
        out = {"steps": self.n_steps, "last_step": self.step, "fn": self.cfg.fn,
               "threshold": self.cfg.threshold, "warmup": self.cfg.warmup,
               "downsample": self.ds, "mode": self.cfg.mode}
        for i, name in enumerate(("pos", "neg")):
            br = self.branches[i]
            out["hits_" + name] = br.hits
            out["misses_" + name] = br.misses
            out["fd_" + name] = br.last_fd
            out["acc_" + name] = br.acc
        out["hits"] = out["hits_pos"] + out["hits_neg"]
        out["misses"] = out["misses_pos"] + out["misses_neg"]
        total = out["hits"] + out["misses"]
        out["total"] = total
        out["hit_rate"] = (out["hits"] / total) if total else 0.0
        return out
