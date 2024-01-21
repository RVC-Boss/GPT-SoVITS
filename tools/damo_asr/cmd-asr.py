# -*- coding:utf-8 -*-

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models import Model
import multiprocessing 
import sys,os,traceback
from threading import Lock
lock = Lock()

# 进程数
processes = 2

dir=sys.argv[1]
# opt_name=dir.split("\\")[-1].split("/")[-1]
opt_name=os.path.basename(dir)
# FunAsr三语转写model
lang2model = {
            'zh': 'tools/damo_asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            'ja': "tools/damo_asr/models/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline",
            "en": "tools/damo_asr/models/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-offline",
}

model = Model.from_pretrained(lang2model["zh"])

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=model,
    vad_model='tools/damo_asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    punc_model='tools/damo_asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
)

def process_audio_file(dir,name,opt_name):

    try:
        text = inference_pipeline(audio_in="%s/%s" % (dir, name))["text"]

        with lock:
            with open(filename,"a",encoding="utf-8")as f:f.write("%s/%s|%s|ZH|%s\n" % (dir, name, opt_name, text))

    except:
        print(traceback.format_exc())

def run__process():  # 主进程

    opt_dir="output/asr_opt"
    os.makedirs(opt_dir,exist_ok=True)
    filename = "%s/%s.list"%(opt_dir,opt_name)
    os.remove(filename,exist_ok=True)

    with multiprocessing.Pool(processes=processes) as pool:
        pool.starmap(process_audio_file, [(dir, name ,opt_name) for name in os.listdir(dir)])
    
if __name__ == '__main__':
    
    run__process()
