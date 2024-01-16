# -*- coding:utf-8 -*-

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import sys,os,traceback
dir=sys.argv[1]
# opt_name=dir.split("\\")[-1].split("/")[-1]
opt_name=os.path.basename(dir)
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='tools/damo_asr/models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    vad_model='tools/damo_asr/models/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    punc_model='tools/damo_asr/models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
)

opt=[]
for name in os.listdir(dir):
    try:
        text = inference_pipeline(audio_in="%s/%s"%(dir,name))["text"]
        opt.append("%s/%s|%s|ZH|%s"%(dir,name,opt_name,text))
    except:
        print(traceback.format_exc())

opt_dir="output/asr_opt"
os.makedirs(opt_dir,exist_ok=True)
with open("%s/%s.list"%(opt_dir,opt_name),"w",encoding="utf-8")as f:f.write("\n".join(opt))

