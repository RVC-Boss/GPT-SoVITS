"""
TTS Only - GPT-SoVITS에서 TTS 기능만 분리한 단순 함수

Usage:
    from tts_simple import TTSEngine

    # Initialize TTS engine
    tts = TTSEngine(model="v4", device="cuda")

    # Generate speech
    audio_data = tts.generate_speech(
        text="안녕하세요, 테스트입니다.",
        text_lang="ko",
        prompt_text="안녕",
        prompt_lang="ko",
        ref_audio_path="reference.wav"
    )

    # Save to file
    tts.save_audio(audio_data, "output.wav")
"""

import os
import sys
import warnings
import numpy as np
import soundfile as sf
from typing import Tuple

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)

from TTS_infer_pack.TTS import TTS, TTS_Config

# Suppress warnings
warnings.filterwarnings("ignore")

# Model Configurations - V4 and V2Pro only
MODEL_CONFIGS = {
    "v4": {
        "description": "V4 model - 48kHz output, fixed metallic noise issues from V3",
        "version": "v4",
        "t2s_weights_path": "pretrained_models/gsv-v4-pretrained/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "vits_weights_path": "pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
        "vocoder_path": "pretrained_models/gsv-v4-pretrained/vocoder.pth",
        "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
        "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
    },
    "v2pro": {
        "description": "V2Pro model - Better performance than V4 with V2-level hardware cost",
        "version": "v2Pro",
        "t2s_weights_path": "pretrained_models/v2Pro/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "vits_weights_path": "pretrained_models/v2Pro/s2Gv2Pro.pth",
        "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
        "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
        "sv_model_path": "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    },
    "v2proplus": {
        "description": "V2ProPlus model - Enhanced version of V2Pro",
        "version": "v2ProPlus",
        "t2s_weights_path": "pretrained_models/v2Pro/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "vits_weights_path": "pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
        "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
        "sv_model_path": "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    }
}


class TTSEngine:
    """Simple TTS Engine for GPT-SoVITS V4/V2Pro models"""

    def __init__(self, model: str = "v4", device: str = "cuda", is_half: bool = True):
        """
        Initialize TTS Engine

        Args:
            model: Model version ("v4", "v2pro", "v2proplus")
            device: Device to use ("cuda", "cpu")
            is_half: Use half precision (FP16)
        """
        if model not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model}. Available: {list(MODEL_CONFIGS.keys())}")

        self.model = model
        self.device = device
        self.is_half = is_half
        self.config = MODEL_CONFIGS[model].copy()
        self.config.update({
            "device": device,
            "is_half": is_half
        })

        # Initialize TTS
        self.tts_config = TTS_Config(self.config)
        self.tts = TTS(self.tts_config)

        print(f"Initialized TTS Engine with {model.upper()} model on {device}")

    def generate_speech(
        self,
        text: str,
        text_lang: str,
        prompt_text: str,
        prompt_lang: str,
        ref_audio_path: str,
        text_split_method: str = "cut5",
        batch_size: int = 1,
        speed_factor: float = 1.0,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        sample_steps: int = 50,
        super_sampling: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text

        Args:
            text: Text to synthesize
            text_lang: Language of the text ("ko", "en")
            prompt_text: Reference text
            prompt_lang: Language of the reference text
            ref_audio_path: Path to reference audio file
            text_split_method: Text splitting method
            batch_size: Batch size for inference
            speed_factor: Speed factor (1.0 = normal speed)
            top_k: Top-k sampling
            top_p: Top-p sampling
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty
            sample_steps: Number of sampling steps
            super_sampling: Enable super sampling

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            generator = self.tts.run(
                inputs={
                    "text": text,
                    "text_lang": text_lang.upper(),
                    "prompt_text": prompt_text,
                    "prompt_lang": prompt_lang.upper(),
                    "ref_audio_path": ref_audio_path,
                    "aux_ref_audio_paths": [],
                    "text_split_method": text_split_method,
                    "batch_size": batch_size,
                    "speed_factor": speed_factor,
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    "sample_steps": sample_steps,
                    "super_sampling": super_sampling
                }
            )

            # Collect all audio chunks
            audio_chunks = []
            sample_rate = None

            for chunk in generator:
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    audio_data, sr = chunk
                    if sample_rate is None:
                        sample_rate = sr
                    audio_chunks.append(audio_data)
                elif isinstance(chunk, dict) and "audio" in chunk:
                    audio_data = chunk["audio"]
                    sr = chunk.get("sample_rate", 48000)
                    if sample_rate is None:
                        sample_rate = sr
                    audio_chunks.append(audio_data)

            if not audio_chunks:
                raise RuntimeError("No audio generated")

            # Concatenate all chunks
            final_audio = np.concatenate(audio_chunks, axis=0)

            return final_audio, sample_rate

        except Exception as e:
            raise RuntimeError(f"TTS generation failed: {str(e)}") from e

    def save_audio(self, audio_data: np.ndarray, output_path: str, sample_rate: int = 48000):
        """
        Save audio data to file

        Args:
            audio_data: Audio data array
            output_path: Output file path
            sample_rate: Sample rate
        """
        try:
            sf.write(output_path, audio_data, sample_rate)
            print(f"Audio saved to: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save audio: {str(e)}") from e

    def generate_and_save(
        self,
        text: str,
        text_lang: str,
        prompt_text: str,
        prompt_lang: str,
        ref_audio_path: str,
        output_path: str,
        **kwargs
    ):
        """
        Generate speech and save to file in one step

        Args:
            text: Text to synthesize
            text_lang: Language of the text
            prompt_text: Reference text
            prompt_lang: Language of the reference text
            ref_audio_path: Path to reference audio file
            output_path: Output file path
            **kwargs: Additional arguments for generate_speech
        """
        audio_data, sample_rate = self.generate_speech(
            text=text,
            text_lang=text_lang,
            prompt_text=prompt_text,
            prompt_lang=prompt_lang,
            ref_audio_path=ref_audio_path,
            **kwargs
        )

        self.save_audio(audio_data, output_path, sample_rate)


def example_usage():
    """Example usage of TTSEngine"""
    try:
        # Initialize TTS engine
        tts = TTSEngine(model="v4", device="cuda")

        # Example generation
        audio_data, sample_rate = tts.generate_speech(
            text="안녕하세요, GPT-SoVITS TTS 엔진 테스트입니다.",
            text_lang="ko",
            prompt_text="안녕",
            prompt_lang="ko",
            ref_audio_path="reference.wav"  # You need to provide this
        )

        # Save audio
        tts.save_audio(audio_data, "output.wav", sample_rate)

        print("TTS generation completed successfully!")

    except (RuntimeError, ValueError, OSError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_usage()
