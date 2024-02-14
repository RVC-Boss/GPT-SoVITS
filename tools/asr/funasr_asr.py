# -*- coding:utf-8 -*-

import argparse
import os
import traceback

import torch
from funasr import AutoModel

from tools.asr.config import BaseASR
from tools.my_utils import ASR_Logger

funasr_component = {
    'asr': {
        'name': 'Paraformer-Large',
        'size': 'speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    },
    'vad': {
        'name': 'FSMN-Monophone VAD',
        'size': 'speech_fsmn_vad_zh-cn-16k-common-pytorch',
    },
    'punc': {
        'name': 'Controllable Time-delay Transformer',
        'size': 'punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    }
}

class FunASR(BaseASR):
    def __init__(self, model_size='large', device="cuda", precision="float16"):
        self.check_local_models()
        self.model = self.load_model()
        assert self.model is not None, ASR_Logger.error('模型不存在')

    @classmethod
    def check_local_models(self):
        '''
        启动时检查本地是否有 FunASR 相关模型.
        '''
        self.model_path_dict = funasr_component
        for code, dic in self.model_path_dict.items():
            model_name = dic['size']
            model_path, flag = super().check_local_model(
                self,
                model_name = model_name,
                model_file = 'model.pt',
                cache_path = os.path.normpath(os.path.expanduser(f"~/.cache/modelscope/hub/"))
            )
            if model_path:
                self.model_path_dict[code]['path'] = model_path
            else:
                # 没有本地路径时, 路径设置为网络链接
                self.model_path_dict[code]['path'] = 'iic/' + model_name
        return self.model_path_dict

    def load_model(self):
        try:
            for code, dic in self.model_path_dict.items():
                if os.path.exists(dic['path']):
                    ASR_Logger.info(f"加载模型: 从 {dic['path']} 加载 {dic['name']} 模型.")
                    if 'modelscope' in dic['path']:
                        ASR_Logger.warning(f"可将 {dic['path']} 移动到 tools/asr/models/ 文件夹下.")
                else:
                    ASR_Logger.warning(f"下载模型: 从 {dic['path']} 下载 {dic['name']} 模型.")
            model = AutoModel(
                model               = self.model_path_dict['asr']['path'],
                model_revision      = "v2.0.4",
                vad_model           = self.model_path_dict['vad']['path'],
                vad_model_revision  = "v2.0.4",
                punc_model          = self.model_path_dict['punc']['path'],
                punc_model_revision = "v2.0.4",
            )
            ASR_Logger.propagate = False # 避免 FunASR 库导致打印重复日志

            if model.kwargs['device'] != 'cpu':
                device_name = torch.cuda.get_device_name(model.kwargs['device'])
            else:
                device_name = 'CPU'
            ASR_Logger.info(f"运行设备: {device_name}, 设定精度: --.")
            ASR_Logger.info(f"创建模型: FunASR 完成.\n")
            return model
        except:
            ASR_Logger.error(traceback.format_exc())
            raise ValueError(ASR_Logger.error(f"模型加载失败 or 下载失败, 可访问 https://modelscope.cn/organization/iic 自行下载, 并放置于 tools/asr/models/ 文件夹下"))
        
    def inference(self, file_path, language='zh'):
        try:
            text = self.model.generate(input=file_path)[0]["text"]
            return text, language
        except:
            ASR_Logger.error(f"当前文件 {file_path} 转写失败, 可能不是有效的音频文件.")
            ASR_Logger.error(traceback.format_exc())
            return '', ''
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file_or_folder", type=str, required=True,
                        help="Input audio file path or folder contain audio files.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, 
                        help="Output folder to store transcriptions.")
    parser.add_argument("-s", "--model_size", type=str, default='large',
                        help="Model Size of FunASR is Large")
    parser.add_argument("-l", "--language", type=str, default='zh', choices=['zh'],
                        help="Language of the audio files.")
    parser.add_argument("-p", "--precision", type=str, default='float16', 
                        choices=['float16','float32'], help="fp16 or fp32")#还没接入

    cmd = parser.parse_args()
    ASR = FunASR(
        model_size = cmd.model_size,
        precision  = cmd.precision,
    )
    ASR.inference_file_or_folder(
        input_file_or_folder = cmd.input_file_or_folder,
        output_folder        = cmd.output_folder,
        language             = cmd.language,
    )
