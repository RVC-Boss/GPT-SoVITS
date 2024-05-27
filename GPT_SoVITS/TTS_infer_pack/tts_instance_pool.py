import threading
from time import perf_counter
import traceback
from typing import Dict, Union

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config


class TTSWrapper(TTS):
    heat: float = 0
    usage_count: int = 0
    usage_counter: int = 0
    usage_time: float = 0.0
    first_used_time: float = 0.0

    def __init__(self, configs: Union[dict, str, TTS_Config]):
        super(TTSWrapper, self).__init__(configs)
        self.first_used_time = perf_counter()

    def __hash__(self) -> int:
        return hash(self.first_used_time)

    def run(self, *args, **kwargs):
        self.usage_counter += 1
        t0 = perf_counter()
        for result in super(TTSWrapper, self).run(*args, **kwargs):
            yield result
        t1 = perf_counter()
        self.usage_time += t1 - t0
        idle_time = self.usage_time - self.first_used_time
        self.heat = self.usage_counter / idle_time

    def reset_heat(self):
        self.heat: int = 0
        self.usage_count: int = 0
        self.usage_time: float = 0.0
        self.first_used_time: float = perf_counter()


class TTSInstancePool:
    def __init__(self, max_size):
        self.max_size: int = max_size
        self.semaphore: threading.Semaphore = threading.Semaphore(max_size)
        self.pool_lock: threading.Lock = threading.Lock()
        self.pool: Dict[int, TTSWrapper] = dict()
        self.current_index: int = 0
        self.size: int = 0

    def acquire(self, configs: TTS_Config):

        self.semaphore.acquire()
        try:
            with self.pool_lock:
                # 查询最匹配的实例
                indexed_key = None
                rank = []
                for key, tts_instance in self.pool.items():
                    if tts_instance.configs.vits_weights_path == configs.vits_weights_path \
                            and tts_instance.configs.t2s_weights_path == configs.t2s_weights_path:
                        indexed_key = key
                    rank.append((tts_instance.heat, key))
                rank.sort(key=lambda x: x[0])
                matched_key = None if len(rank) == 0 else rank[0][1]

                # 如果已有实例匹配，则直接复用
                if indexed_key is not None:
                    tts_instance = self._reuse_instance(indexed_key, configs)
                    print(f"如果已有实例匹配，则直接复用: {configs.vits_weights_path} {configs.t2s_weights_path}")
                    return tts_instance

                # 如果pool未满，则创建一个新实例
                if self.size < self.max_size:
                    tts_instance = TTSWrapper(configs)
                    self.size += 1
                    print(f"如果pool未满，则创建一个新实例: {configs.vits_weights_path} {configs.t2s_weights_path}")
                    return tts_instance
                else:
                    # 否则用最合适的实例进行复用
                    tts_instance = self._reuse_instance(matched_key, configs)
                    print(f"否则用最合适的实例进行复用: {configs.vits_weights_path} {configs.t2s_weights_path}")
                    return tts_instance
        except Exception as e:
            self.semaphore.release()
            traceback.print_exc()
            raise e

    def release(self, tts_instance: TTSWrapper):
        assert tts_instance is not None
        with self.pool_lock:
            key = hash(tts_instance)
            if key in self.pool.keys():
                return
            self.pool[key] = tts_instance
        self.semaphore.release()

    def clear_pool(self):
        for i in range(self.max_size):
            self.semaphore.acquire()
        with self.pool_lock:
            self.pool.clear()
        # for i in range(self.max_size):
        self.semaphore.release(self.max_size)

    def _reuse_instance(self, instance_key: int, configs: TTS_Config) -> TTSWrapper:
        """
        复用已有实例
        args:
            instance_key: int, 已有实例的Key
            config: TTS_Config
        return:
            TTS_Wrapper: 返回复用的TTS实例
        """

        # 复用已有实例
        tts_instance = self.pool.pop(instance_key, None)
        if tts_instance is None:
            raise ValueError("Instance not found")

        tts_instance.configs.device = configs.device
        if tts_instance.configs.vits_weights_path != configs.vits_weights_path \
                or tts_instance.configs.t2s_weights_path != configs.t2s_weights_path:
            tts_instance.reset_heat()

        if tts_instance.configs.vits_weights_path != configs.vits_weights_path:
            tts_instance.init_vits_weights(configs.vits_weights_path, False)
            tts_instance.configs.vits_weights_path = configs.vits_weights_path

        if tts_instance.configs.t2s_weights_path != configs.t2s_weights_path:
            tts_instance.init_t2s_weights(configs.t2s_weights_path, False)
            tts_instance.configs.t2s_weights_path = configs.t2s_weights_path

        tts_instance.set_device(configs.device, False)
        return tts_instance
