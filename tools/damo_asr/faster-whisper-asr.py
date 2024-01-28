import sys,os,traceback
dir=sys.argv[1]
model_name=sys.argv[2]
# opt_name=dir.split("\\")[-1].split("/")[-1]
opt_name=os.path.basename(dir)

from faster_whisper import WhisperModel

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# model_name="small"

if device == "cuda":
    model = WhisperModel(model_name, device="cuda", compute_type="float16",download_root="./model_from_whisper",local_files_only=False)
else:
    model = WhisperModel(model_name, device="cpu", compute_type="int8",download_root="./model_from_whisper",local_files_only=False)

def make_faster(file_path):

    segments, info = model.transcribe(file_path, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    text_str = ""
    for segment in segments:
        text_str += f"{segment.text.lstrip()},"
    
    print(text_str)

    return info.language,text_str.rstrip(",")

opt=[]
for name in os.listdir(dir):
    try:
        res = make_faster("%s/%s"%(dir,name))
        text = res[1]
        opt.append("%s/%s|%s|%s|%s"%(dir,name,opt_name,res[0],text))
    except:
        print(traceback.format_exc())

opt_dir="output/asr_opt"
os.makedirs(opt_dir,exist_ok=True)
with open("%s/%s.list"%(opt_dir,"slicer_opt"),"w",encoding="utf-8")as f:f.write("\n".join(opt))