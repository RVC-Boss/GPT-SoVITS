import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import math
from torchaudio.transforms import Resample
import VoiceSave
import uuid

def get_train_set(voice_file_path):
    if type(voice_file_path) == str:
        voice_file_path = [voice_file_path]
    ret = []
    for i in voice_file_path:
        tensors_ = VoiceSave.load_tensor(i,
                              f"get_{uuid.uuid4()}",
                              find_func=VoiceSave.__find_func__,
                              MySet=set())
        ret.append(tensors_)
    return ret

class MelSpectrogram(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.filter_length = hps.data.filter_length
        self.hop_length = hps.data.hop_length
        self.win_length = hps.data.win_length
        self.sampling_rate = hps.data.sampling_rate
        self.n_mel_channels = hps.data.n_mel_channels  
        self.mel_fmin = hps.data.mel_fmin if hasattr(hps.data, 'mel_fmin') else 0
        self.mel_fmax = hps.data.mel_fmax if hasattr(hps.data, 'mel_fmax') else None

        # 构建梅尔频谱变换
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_mels=192,  # self.n_mel_channels,
            window_fn=torch.hann_window,
            center=False,
            power=1.0,
        )

    def forward(self, audio):
        """
        输入：audio [B, 1, T] 或 [1, T]（单声道音频）
        输出：mel_spec [B, n_mel_channels, T']
        """
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(0)  # [1, T] → [1, 1, T]
        
        # 提取梅尔频谱
        mel_spec = self.mel_transform(audio.squeeze(1))  # [B, n_mel, T']
        
        # 对数缩放（TTS标准操作）
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        return mel_spec

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_seq_length, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        self.pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        self.register_buffer('pe', self.pe.unsqueeze(0))  # 注册为缓冲区
        
    def forward(self, x):
        # 将位置编码添加到输入中
        return x + self.pe[:, :x.size(1)]
    
class Spliter(nn.Module):
    '''output: z_p shape: torch.Size([1, 192, x]), y_mask shape: torch.Size([1, 1, x]), ge shape: torch.Size([1, 1024, 1])'''
    def __init__(self,
                 hps,
                 ge,
                 device):
        super().__init__()
        self.hps = hps

        self.ge = ge
        self.device = device
        #TODO: 将mel_spec与ge输入Transformer模型
        self.mel_dim = 192
        self.ge_dim = 1024                
        self.transformer_dim = 512
        self.ge_proj = nn.Linear(self.ge_dim, self.transformer_dim).to(self.device)
        self.mel_proj = nn.Linear(self.mel_dim, self.transformer_dim).to(self.device)
        self.pos_encoder = PositionalEncoding(self.transformer_dim).to(self.device)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.transformer_dim,
                nhead=hps.model.nhead,
                dim_feedforward=hps.model.ffn_dim,
                batch_first=False,
                dropout=0.1
            ),
            num_layers=hps.model.num_layers
        ).to(self.device)
    
        self.out_proj = nn.Linear(self.transformer_dim, self.mel_dim).to(self.device)

    @torch.no_grad()
    def mel_(self,audio_path, hps, device, dtype):
        sr_target = int(hps.data.sampling_rate)
        audio, sr_origin = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if sr_origin != sr_target:
            resampler = Resample(sr_origin, sr_target).to(device)
            audio = resampler(audio.to(device))
        else:
            audio = audio.to(device)
        max_audio = audio.abs().max()
        if max_audio > 1.0:
            audio = audio / max_audio
        mel_extractor = MelSpectrogram(hps).to(device)
        mel_spec = mel_extractor(audio).to(dtype)
        return mel_spec
    
    def forward(self, audio_path, ge,device,dtype):
        # 输入：audio_path, ge
        # 输出：z_p, y_mask, ge
        ge_ = ge
        mel = self.mel_(audio_path, self.hps, device, dtype)

        mel = mel.permute(2, 0, 1)  
        # 梅尔谱投影到Transformer维度：[T, 1, 512]
        mel_feat = self.mel_proj(mel)  
        
        # 全局情感特征GE处理：[1,1024,1] → [1,1024] → [1,1,512]
        ge = ge.to(device, dtype=dtype)
        ge_squeeze = ge.squeeze(-1)  # [1, 1024]
        ge_feat = self.ge_proj(ge_squeeze).unsqueeze(0)  # [1, 1, 512]
        
        # ===================== 3. 特征融合与Transformer输入 =====================
        # 将GE特征拼接在梅尔谱序列开头：[T+1, 1, 512]
        self.transformer_input = torch.cat([ge_feat, mel_feat], dim=0)
        # 添加位置编码
        self.transformer_input = self.pos_encoder(self.transformer_input)
        
        # ===================== 4. Transformer编码 =====================
        transformer_out = self.transformer(self.transformer_input)  # [T+1, 1, 512]
        
        # ===================== 5. 输出特征重构 =====================
        # 去除GE开头，提取梅尔谱对应的输出：[T, 1, 512]
        mel_out = transformer_out[1:, :, :]
        # 投影回原始梅尔维度：[T, 1, 192]
        mel_out = self.out_proj(mel_out)
        # 转换为目标格式：[1, 192, T] → z_p
        z_p = mel_out.permute(1, 2, 0)
        
        # ===================== 6. 生成掩码 =====================
        T = z_p.shape[-1]  # 梅尔谱时间步
        y_mask = torch.ones(1, 1, T, device=device, dtype=dtype)  # [1,1,T] 全1掩码
        
        # ===================== 7. 输出（严格匹配注释格式） =====================
        return z_p, y_mask, ge_

class SpliterDataset(torch.utils.data.Dataset):
    def __init__(self, voice_file_paths):
        self.voice_file_paths = voice_file_paths
        self.datas = get_train_set(voice_file_paths)

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        return self.datas[idx]