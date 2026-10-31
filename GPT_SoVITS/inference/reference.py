"""Reference audio, semantic, spectrogram, and speaker cache ownership."""

import os

import numpy as np
import torch
import torchaudio

from GPT_SoVITS.inference.audio import load_audio_gpu, resample
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.runtime.i18n import I18nAuto

language = os.environ.get("language", "Auto")
i18n = I18nAuto(language=language)


class ReferenceMixin:
    def set_ref_audio(self, ref_audio_path: str):
        """
        To set the reference audio for the TTS model,
            including the prompt_semantic and refer_spepc.
        Args:
            ref_audio_path: str, the path of the reference audio.
        """
        self._set_prompt_semantic(ref_audio_path)
        self._set_ref_spec(ref_audio_path)
        self._set_ref_audio_path(ref_audio_path)

    def _set_ref_audio_path(self, ref_audio_path):
        self.prompt_cache["ref_audio_path"] = ref_audio_path

    def _set_ref_spec(self, ref_audio_path):
        spec_audio = self._get_ref_spec(ref_audio_path)
        if self.prompt_cache["refer_spec"] in [[], None]:
            self.prompt_cache["refer_spec"] = [spec_audio]
        else:
            self.prompt_cache["refer_spec"][0] = spec_audio

    def _get_ref_spec(self, ref_audio_path):
        raw_audio, raw_sr = torchaudio.load(ref_audio_path)
        raw_audio = raw_audio.to(self.configs.device).float()
        self.prompt_cache["raw_audio"] = raw_audio
        self.prompt_cache["raw_sr"] = raw_sr

        if raw_sr != self.configs.sampling_rate:
            audio = raw_audio.to(self.configs.device)
            if audio.shape[0] == 2:
                audio = audio.mean(0).unsqueeze(0)
            audio = resample(audio, raw_sr, self.configs.sampling_rate, self.configs.device)
        else:
            audio = raw_audio.to(self.configs.device)
            if audio.shape[0] == 2:
                audio = audio.mean(0).unsqueeze(0)

        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        spec = spectrogram_torch(
            audio,
            self.configs.filter_length,
            self.configs.sampling_rate,
            self.configs.hop_length,
            self.configs.win_length,
            center=False,
        )
        if self.configs.is_half:
            spec = spec.half()
        if self.is_v2pro == True:
            audio = resample(audio, self.configs.sampling_rate, 16000, self.configs.device)
            if self.configs.is_half:
                audio = audio.half()
        else:
            audio = None
        return spec, audio

    def _set_prompt_semantic(self, ref_wav_path: str):
        zero_wav = np.zeros(
            int(self.configs.sampling_rate * 0.3),
            dtype=np.float16 if self.configs.is_half else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = load_audio_gpu(
                ref_wav_path, 16000, self.configs.device, mono=True,
                dtype=torch.float32, match_librosa=True,
            )
            if wav16k.shape[-1] > 160000 or wav16k.shape[-1] < 48000:
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.squeeze(0)
            zero_wav_torch = zero_wav_torch.to(self.configs.device)
            if self.configs.is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()

            wav16k = torch.cat([wav16k, zero_wav_torch])
            hubert_feature = self.cnhuhbert_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(
                1, 2
            )  # .float()
            codes = self.vits_model.extract_latent(hubert_feature)

            prompt_semantic = codes[0, 0].to(self.configs.device)
            self.prompt_cache["prompt_semantic"] = prompt_semantic
