import onnxruntime as ort
import numpy as np
import onnx
from tqdm import tqdm
import torchaudio
import torch
from TTS_infer_pack.TextPreprocessor_onnx import TextPreprocessorOnnx


MODEL_PATH = "playground/v2pro_export/v2pro"

def audio_postprocess(
    audios,
    fragment_interval: float = 0.3,
):
    zero_wav = np.zeros((int(32000 * fragment_interval),)).astype(np.float32)
    for i, audio in enumerate(audios):
        max_audio = np.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio /= max_audio
        audio = np.concatenate([audio, zero_wav], axis=0)
        audios[i] = audio

    audio = np.concatenate(audios, axis=0)

    # audio = (audio * 32768).astype(np.int16)

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    torchaudio.save('playground/output.wav', audio_tensor, 32000)

    return audio

def load_and_preprocess_audio(audio_path, max_length=160000):
    """Load and preprocess audio file to 16k"""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Take first channel
    if waveform.shape[0] > 1:
        waveform = waveform[0:1]
    
    # Limit length for testing (10 seconds at 16kHz)
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]

    # make a zero tensor that has length 3200*0.3
    zero_tensor = torch.zeros((1, 4800), dtype=torch.float32)

    # concate zero_tensor with waveform
    waveform = torch.cat([waveform, zero_tensor], dim=1)

    return waveform

def get_audio_hubert(audio_path):
    """Get HuBERT features for the audio file"""
    waveform = load_and_preprocess_audio(audio_path)
    ort_session = ort.InferenceSession("playground/hubert/chinese-hubert-base.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: waveform.numpy()}
    hubert_feature = ort_session.run(None, ort_inputs)[0].astype(np.float32)
    # transpose axis 1 and 2 with numpy
    hubert_feature = hubert_feature.transpose(0, 2, 1)
    return hubert_feature

def preprocess_text(text:str):
    preprocessor = TextPreprocessorOnnx("playground/bert")
    [phones, bert_features, norm_text] = preprocessor.segment_and_extract_feature_for_text(text, 'all_zh', 'v2')
    phones = np.expand_dims(np.array(phones, dtype=np.int64), axis=0)
    return phones, bert_features.T.astype(np.float32)


# input_phones_saved = np.load("playground/ref/input_phones.npy")
# input_bert_saved = np.load("playground/ref/input_bert.npy").T.astype(np.float32)
[input_phones, input_bert] = preprocess_text("震撼视角,感受开幕式,闭幕式烟花")


# ref_phones = np.load("playground/ref/ref_phones.npy")
# ref_bert = np.load("playground/ref/ref_bert.npy").T.astype(np.float32)
[ref_phones, ref_bert] = preprocess_text("今日江苏苏州荷花市集开张热闹与浪漫交织")


audio_prompt_hubert = get_audio_hubert("playground/ref/audio.wav")


encoder = ort.InferenceSession(MODEL_PATH+"_export_t2s_encoder.onnx")

[y, k, v, y_emb, x_example] = encoder.run(None, {
    "text_seq": input_phones,
    "text_bert": input_bert,
    "ref_seq": ref_phones,
    "ref_bert": ref_bert,
    "ssl_content": audio_prompt_hubert
})

# fsdec = ort.InferenceSession(MODEL_PATH+"_export_t2s_fsdec.onnx")
sdec = ort.InferenceSession(MODEL_PATH+"_export_t2s_sdec.onnx")

# for i in tqdm(range(10000)):
# [y, k, v, y_emb, x_example] = fsdec.run(None, {
#     "x": x,
#     "prompts": prompts
# })


stop = False
for idx in tqdm(range(1, 1500)):
    # [1, N] [N_layer, N, 1, 512] [N_layer, N, 1, 512] [1, N, 512] [1] [1, N, 512] [1, N]
    [y, k, v, y_emb, logits, samples] = sdec.run(None, {
        "iy": y,
        "ik": k,
        "iv": v,
        "iy_emb": y_emb,
        "ix_example": x_example
    })
    if np.argmax(logits, axis=-1)[0] == 1024 or samples[0, 0] == 1024: # 1024 is the EOS token
        break
y[0, -1] = 0


pred_semantic = np.expand_dims(y[:, -idx:], axis=0)
# Read and resample reference audio
waveform, sample_rate = torchaudio.load("playground/ref/audio.wav")
if sample_rate != 32000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=32000)
    waveform = resampler(waveform)
print(f"Waveform shape: {waveform.shape}")
ref_audio = waveform.numpy().astype(np.float32)

vtis = ort.InferenceSession(MODEL_PATH+"_export_vits.onnx")

[audio] = vtis.run(None, {
    "text_seq": input_phones,
    "pred_semantic": pred_semantic,
    "ref_audio": ref_audio
})

audio_postprocess([audio])
