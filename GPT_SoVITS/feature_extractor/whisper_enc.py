import torch
import whisper
from whisper import log_mel_spectrogram, pad_or_trim


def get_model():
    model = whisper.load_model("small", device="cpu")

    return model.encoder


def get_content(model: whisper.Whisper, wav_16k_tensor: torch.Tensor):
    assert model
    dev = next(model.parameters()).device
    mel = log_mel_spectrogram(wav_16k_tensor).to(dev)[:, :3000]
    # if torch.cuda.is_available():
    #     mel = mel.to(torch.float16)
    feature_len = mel.shape[-1] // 2
    assert mel.shape[-1] < 3000, "输入音频过长，只允许输入30以内音频"
    with torch.no_grad():
        feature = model(pad_or_trim(mel, 3000).unsqueeze(0))[:1, :feature_len, :].transpose(1, 2)  # type: ignore
    return feature
