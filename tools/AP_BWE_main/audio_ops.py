"""Compatibility shim for the vendored AP-BWE audio helpers."""

from GPT_SoVITS.vendor.ap_bwe.audio_ops import amp_pha_istft, amp_pha_stft

__all__ = ["amp_pha_istft", "amp_pha_stft"]
