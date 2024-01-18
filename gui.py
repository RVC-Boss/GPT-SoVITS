import json,yaml,warnings,torch
import platform
import psutil
import os
import signal
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget
from PyQt5.QtWidgets import QSlider, QComboBox, QLabel, QCheckBox, QTextEdit, QLineEdit, QPushButton

warnings.filterwarnings("ignore")
torch.manual_seed(233333)
import os,pdb,sys
now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
import site
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
for site_packages_root in site_packages_roots:
    with open("%s/users.pth" % (site_packages_root), "w") as f:
        f.write(
            "%s\n%s/tools\n%s/tools/damo_asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
            % (now_dir, now_dir, now_dir, now_dir, now_dir)
        )
import traceback
sys.path.append(now_dir)
import shutil
import pdb
import gradio as gr
from subprocess import Popen
import signal
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()
from scipy.io import wavfile
from tools.my_utils import load_audio
from multiprocessing import cpu_count
n_cpu=cpu_count()
           
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ["10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L"]):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])

pretrained_sovits_name="GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):SoVITS_names.append(name)
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append(name)
    return SoVITS_names,GPT_names
SoVITS_weight_root="SoVITS_weights"
GPT_weight_root="GPT_weights"
os.makedirs(SoVITS_weight_root,exist_ok=True)
os.makedirs(GPT_weight_root,exist_ok=True)
SoVITS_names,GPT_names = get_weights_names()

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names), "__type__": "update"}, {"choices": sorted(GPT_names), "__type__": "update"}

p_label=None
p_uvr5=None
p_asr=None
p_tts_inference=None

def kill_proc_tree(pid, including_parent=True):  
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass

system=platform.system()
def kill_process(pid):
    if(system=="Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)
    

def change_label(if_label,path_list):
    global p_label
    if(if_label==True and p_label==None):
        cmd = '"%s" tools/subfix_webui.py --load_list "%s" --webui_port %s'%(python_exec,path_list,webui_port_subfix)
        yield "打标工具WebUI已开启"
        print(cmd)
        p_label = Popen(cmd, shell=True)
    elif(if_label==False and p_label!=None):
        kill_process(p_label.pid)
        p_label=None
        yield "打标工具WebUI已关闭"

def change_uvr5(if_uvr5):
    global p_uvr5
    if(if_uvr5==True and p_uvr5==None):
        cmd = '"%s" tools/uvr5/webui.py "%s" %s %s'%(python_exec,infer_device,is_half,webui_port_uvr5)
        yield "UVR5已开启"
        print(cmd)
        p_uvr5 = Popen(cmd, shell=True)
    elif(if_uvr5==False and p_uvr5!=None):
        kill_process(p_uvr5.pid)
        p_uvr5=None
        yield "UVR5已关闭"

def change_tts_inference(if_tts,bert_path,cnhubert_base_path,gpu_number,gpt_path,sovits_path):
    global p_tts_inference
    if(if_tts==True and p_tts_inference==None):
        os.environ["gpt_path"]=gpt_path if "/" in gpt_path else "%s/%s"%(GPT_weight_root,gpt_path)
        os.environ["sovits_path"]=sovits_path if "/"in sovits_path else "%s/%s"%(SoVITS_weight_root,sovits_path)
        os.environ["cnhubert_base_path"]=cnhubert_base_path
        os.environ["bert_path"]=bert_path
        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_number
        os.environ["is_half"]=str(is_half)
        os.environ["infer_ttswebui"]=str(webui_port_infer_tts)
        cmd = '"%s" GPT_SoVITS/inference_webui.py'%(python_exec)
        yield "TTS推理进程已开启"
        print(cmd)
        p_tts_inference = Popen(cmd, shell=True)
    elif(if_tts==False and p_tts_inference!=None):
        kill_process(p_tts_inference.pid)
        p_tts_inference=None
        yield "TTS推理进程已关闭"


def open_asr(asr_inp_dir):
    global p_asr
    if(p_asr==None):
        cmd = '"%s" tools/damo_asr/cmd-asr.py "%s"'%(python_exec,asr_inp_dir)
        yield "ASR任务开启：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr=None
        yield "ASR任务完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的ASR任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}

def close_asr():
    global p_asr
    if(p_asr!=None):
        kill_process(p_asr.pid)
        p_asr=None
    return "已终止ASR进程",{"__type__":"update","visible":True},{"__type__":"update","visible":False}

p_train_SoVITS=None
def open1Ba(batch_size,total_epoch,exp_name,text_low_lr_rate,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers1Ba,pretrained_s2G,pretrained_s2D):
    global p_train_SoVITS
    if(p_train_SoVITS==None):
        with open("GPT_SoVITS/configs/s2.json")as f:
            data=f.read()
            data=json.loads(data)
        s2_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["train"]["text_low_lr_rate"]=text_low_lr_rate
        data["train"]["pretrained_s2G"]=pretrained_s2G
        data["train"]["pretrained_s2D"]=pretrained_s2D
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["save_every_epoch"]=save_every_epoch
        data["train"]["gpu_numbers"]=gpu_numbers1Ba
        data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
        data["save_weight_dir"]=SoVITS_weight_root
        data["name"]=exp_name
        tmp_config_path="TEMP/tmp_s2.json"
        with open(tmp_config_path,"w")as f:f.write(json.dumps(data))

        cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"'%(python_exec,tmp_config_path)
        yield "SoVITS训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS=None
        yield "SoVITS训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}

def close1Ba():
    global p_train_SoVITS
    if(p_train_SoVITS!=None):
        kill_process(p_train_SoVITS.pid)
        p_train_SoVITS=None
    return "已终止SoVITS训练",{"__type__":"update","visible":True},{"__type__":"update","visible":False}

p_train_GPT=None
def open1Bb(batch_size,total_epoch,exp_name,if_save_latest,if_save_every_weights,save_every_epoch,gpu_numbers,pretrained_s1):
    global p_train_GPT
    if(p_train_GPT==None):
        with open("GPT_SoVITS/configs/s1longer.yaml")as f:
            data=f.read()
            data=yaml.load(data, Loader=yaml.FullLoader)
        s1_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["pretrained_s1"]=pretrained_s1
        data["train"]["save_every_n_epoch"]=save_every_epoch
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["half_weights_save_dir"]=GPT_weight_root
        data["train"]["exp_name"]=exp_name
        data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
        data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
        data["output_dir"]="%s/logs_s1"%s1_dir

        os.environ["_CUDA_VISIBLE_DEVICES"]=gpu_numbers.replace("-",",")
        os.environ["hz"]="25hz"
        tmp_config_path="TEMP/tmp_s1.yaml"
        with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" '%(python_exec,tmp_config_path)
        yield "GPT训练开始：%s"%cmd,{"__type__":"update","visible":False},{"__type__":"update","visible":True}
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT=None
        yield "GPT训练完成",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的GPT训练任务，需先终止才能开启下一次任务",{"__type__":"update","visible":False},{"__type__":"update","visible":True}

def close1Bb():
    global p_train_GPT
    if(p_train_GPT!=None):
        kill_process(p_train_GPT.pid)
        p_train_GPT=None
    return "已终止GPT训练",{"__type__":"update","visible":True},{"__type__":"update","visible":False}

ps_slice=[]
def open_slice(inp,opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,n_parts):
    global ps_slice
    if(os.path.exists(inp)==False):
        yield "输入路径不存在",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
        return
    if os.path.isfile(inp):n_parts=1
    elif os.path.isdir(inp):pass
    else:
        yield "输入路径存在但既不是文件也不是文件夹",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
        return
    if (ps_slice == []):
        for i_part in range(n_parts):
            cmd = '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s''' % (python_exec,inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, i_part, n_parts)
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield "切割执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps_slice:
            p.wait()
        ps_slice=[]
        yield "切割结束",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的切割任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close_slice():
    global ps_slice
    if (ps_slice != []):
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid)
            except:
                traceback.print_exc()
        ps_slice=[]
    return "已终止所有切割进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1a=[]
def open1a(inp_text,inp_wav_dir,exp_name,gpu_numbers,bert_pretrained_dir):
    global ps1a
    if (ps1a == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        config={
            "inp_text":inp_text,
            "inp_wav_dir":inp_wav_dir,
            "exp_name":exp_name,
            "opt_dir":opt_dir,
            "bert_pretrained_dir":bert_pretrained_dir,
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    "is_half": str(is_half)
                }
            )
            os.environ.update(config)#
            cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1a.append(p)
        yield "文本进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1a:
            p.wait()
        opt = []
        for i_part in range(all_parts):
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        path_text = "%s/2-name2text.txt" % opt_dir
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1a=[]
        yield "文本进程结束",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的文本任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1a():
    global ps1a
    if (ps1a != []):
        for p1a in ps1a:
            try:
                kill_process(p1a.pid)
            except:
                traceback.print_exc()
        ps1a=[]
    return "已终止所有1a进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1b=[]
def open1b(inp_text,inp_wav_dir,exp_name,gpu_numbers,ssl_pretrained_dir):
    global ps1b
    if (ps1b == []):
        config={
            "inp_text":inp_text,
            "inp_wav_dir":inp_wav_dir,
            "exp_name":exp_name,
            "opt_dir":"%s/%s"%(exp_root,exp_name),
            "cnhubert_base_dir":ssl_pretrained_dir,
            "is_half": str(is_half)
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1b.append(p)
        yield "SSL提取进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1b:
            p.wait()
        ps1b=[]
        yield "SSL提取进程结束",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的SSL提取任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1b():
    global ps1b
    if (ps1b != []):
        for p1b in ps1b:
            try:
                kill_process(p1b.pid)
            except:
                traceback.print_exc()
        ps1b=[]
    return "已终止所有1b进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

ps1c=[]
def open1c(inp_text,exp_name,gpu_numbers,pretrained_s2G_path):
    global ps1c
    if (ps1c == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        config={
            "inp_text":inp_text,
            "exp_name":exp_name,
            "opt_dir":opt_dir,
            "pretrained_s2G":pretrained_s2G_path,
            "s2config_path":"GPT_SoVITS/configs/s2.json",
            "is_half": str(is_half)
        }
        gpu_names=gpu_numbers.split("-")
        all_parts=len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1c.append(p)
        yield "语义token提取进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1c:
            p.wait()
        opt = ["item_name	semantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c=[]
        yield "语义token提取进程结束",{"__type__":"update","visible":True},{"__type__":"update","visible":False}
    else:
        yield "已有正在进行的语义token提取任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1c():
    global ps1c
    if (ps1c != []):
        for p1c in ps1c:
            try:
                kill_process(p1c.pid)
            except:
                traceback.print_exc()
        ps1c=[]
    return "已终止所有语义token进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
#####inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,cnhubert_base_dir,pretrained_s2G
ps1abc=[]
def open1abc(inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,ssl_pretrained_dir,pretrained_s2G_path):
    global ps1abc
    if (ps1abc == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        try:
            #############################1a
            path_text="%s/2-name2text.txt" % opt_dir
            if(os.path.exists(path_text)==False or (os.path.exists(path_text)==True and os.path.getsize(path_text)<10)):
                config={
                    "inp_text":inp_text,
                    "inp_wav_dir":inp_wav_dir,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "bert_pretrained_dir":bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names=gpu_numbers1a.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "进度：1a-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                for p in ps1abc:p.wait()

                opt = []
                for i_part in range(all_parts):#txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")

            yield "进度：1a-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc=[]
            #############################1b
            config={
                "inp_text":inp_text,
                "inp_wav_dir":inp_wav_dir,
                "exp_name":exp_name,
                "opt_dir":opt_dir,
                "cnhubert_base_dir":ssl_pretrained_dir,
            }
            gpu_names=gpu_numbers1Ba.split("-")
            all_parts=len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield "进度：1a-done, 1b-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            for p in ps1abc:p.wait()
            yield "进度：1a1b-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc=[]
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if(os.path.exists(path_semantic)==False or (os.path.exists(path_semantic)==True and os.path.getsize(path_semantic)<28)):
                config={
                    "inp_text":inp_text,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "pretrained_s2G":pretrained_s2G_path,
                    "s2config_path":"GPT_SoVITS/configs/s2.json",
                }
                gpu_names=gpu_numbers1c.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "进度：1a1b-done, 1cing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                for p in ps1abc:p.wait()

                opt = ["item_name	semantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                yield "进度：all-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc = []
            yield "一键三连进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
        except:
            traceback.print_exc()
            close1abc()
            yield "一键三连中途报错", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的一键三连任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}

def close1abc():
    global ps1abc
    if (ps1abc != []):
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid)
            except:
                traceback.print_exc()
        ps1abc=[]
    return "已终止所有一键三连进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
  

class GPTSoVITSGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('GPT-SoVITS GUI')
        self.setGeometry(600, 150, 1000, 600)

        main_layout = QVBoxLayout()
        tabs = QTabWidget()

        tabs.addTab(self.create_tab_data_tool(), "0-前置数据集获取工具")
        tabs.addTab(self.create_tab_tts(), "1-GPT-SoVITS-TTS")
        tabs.addTab(self.create_tab_voice_conversion(), "2-GPT-SoVITS-变声")

        main_layout.addWidget(tabs)
        self.setLayout(main_layout)
        self.show()

        self.setFixedSize(self.size())  
        self.setMinimumSize(800, 600) 

        self.setStyleSheet("""
            QWidget {
                background-color: #a3d3b1; 
            }

            QTabWidget::pane {
                background-color: #a3d3b1;  
            }

            QTabWidget::tab-bar {
                alignment: left;
            }

            QTabBar::tab {
                background: #8da4bf; 
                color: #ffffff;  
                padding: 8px;
            }

            QTabBar::tab:selected {
                background: #2a3f54; 
            }

            QLabel {
                color: #333333; 
            }

            QPushButton {
                background-color: #4CAF50; 
                color: white;  
                padding: 8px;
                border: none;
                border-radius: 4px;
            }

            QPushButton:hover {
                background-color: #45a049;  
            }

            QLabel#label_uvr5,
            QLabel#label_slice,
            QLabel#label_asr,
            QLabel#label_label,
            QLabel#label_1Aa,
            QLabel#label_1B,
            QLabel#label_1C {
                color: #000000;  
            }
        """)

    def create_tab_data_tool(self):
        tab = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("0a-UVR5人声伴奏分离&去混响去延迟工具"))

        checkbox_uvr5 = QCheckBox("是否开启UVR5-WebUI", self)
        uvr5_info = QTextEdit("UVR5进程输出信息", self)
        layout.addWidget(checkbox_uvr5)
        layout.addWidget(uvr5_info)

        layout.addWidget(QLabel("0b-语音切分工具"))

        slice_inp_path = QLineEdit("音频自动切分输入路径，可文件可文件夹", self)
        slice_opt_root = QLineEdit("切分后的子音频的输出根目录", self)
        threshold = QLineEdit("threshold:音量小于这个值视作静音的备选切割点", self)
        min_length = QLineEdit("min_length:每段最小多长", self)
        min_interval = QLineEdit("min_interval:最短切割间隔", self)
        hop_size = QLineEdit("hop_size:怎么算音量曲线", self)
        max_sil_kept = QLineEdit("max_sil_kept:切完后静音最多留多长", self)
        open_slicer_button = QPushButton("开启语音切割", self)
        close_slicer_button = QPushButton("终止语音切割", self)
        _max = QSlider(self)
        alpha = QSlider(self)
        n_process = QSlider(self)
        slicer_info = QTextEdit("语音切割进程输出信息", self)
        layout.addWidget(slice_inp_path)
        layout.addWidget(slice_opt_root)
        layout.addWidget(threshold)
        layout.addWidget(min_length)
        layout.addWidget(min_interval)
        layout.addWidget(hop_size)
        layout.addWidget(max_sil_kept)
        layout.addWidget(open_slicer_button)
        layout.addWidget(close_slicer_button)
        layout.addWidget(_max)
        layout.addWidget(alpha)
        layout.addWidget(n_process)
        layout.addWidget(slicer_info)

        layout.addWidget(QLabel("0c-中文批量离线ASR工具"))

        open_asr_button = QPushButton("开启离线批量ASR", self)
        close_asr_button = QPushButton("终止ASR进程", self)
        asr_inp_dir = QLineEdit("批量ASR(中文only)输入文件夹路径", self)
        asr_info = QTextEdit("ASR进程输出信息", self)
        layout.addWidget(open_asr_button)
        layout.addWidget(close_asr_button)
        layout.addWidget(asr_inp_dir)
        layout.addWidget(asr_info)

        layout.addWidget(QLabel("0d-语音文本校对标注工具"))

        if_label = QCheckBox("是否开启打标WebUI", self)
        path_list = QLineEdit("打标数据标注文件路径", self)
        label_info = QTextEdit("打标工具进程输出信息", self)
        layout.addWidget(if_label)
        layout.addWidget(path_list)
        layout.addWidget(label_info)

        checkbox_uvr5.stateChanged.connect(lambda: change_uvr5(checkbox_uvr5.isChecked(), [checkbox_uvr5], [uvr5_info]))
        open_asr_button.clicked.connect(lambda: open_asr(asr_inp_dir.text(), [asr_info, open_asr_button, close_asr_button]))
        close_asr_button.clicked.connect(lambda: close_asr([], [asr_info, open_asr_button, close_asr_button]))
        open_slicer_button.clicked.connect(lambda: open_slice(
            slice_inp_path.text(), slice_opt_root.text(), threshold.text(), min_length.text(),
            min_interval.text(), hop_size.text(), max_sil_kept.text(), _max.value(), alpha.value(), n_process.value(),
            [slicer_info, open_slicer_button, close_slicer_button]
        ))
        close_slicer_button.clicked.connect(lambda: close_slice([], [slicer_info, open_slicer_button, close_slicer_button]))

        tab.setLayout(layout)
        return tab

    def create_tab_tts(self):
        tab = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("1-GPT-SoVITS-TTS"))

        exp_name = QLineEdit("xxx")
        gpu_info = QLineEdit("显卡信息")
        pretrained_s2G = QLineEdit("GPT_SoVITS/pretrained_models/s2G488k.pth")
        pretrained_s2D = QLineEdit("GPT_SoVITS/pretrained_models/s2D488k.pth")
        pretrained_s1 = QLineEdit("GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

        layout.addWidget(QLabel("*实验/模型名"))
        layout.addWidget(exp_name)
        layout.addWidget(QLabel("显卡信息"))
        layout.addWidget(gpu_info)
        layout.addWidget(QLabel("预训练的SoVITS-G模型路径"))
        layout.addWidget(pretrained_s2G)
        layout.addWidget(QLabel("预训练的SoVITS-D模型路径"))
        layout.addWidget(pretrained_s2D)
        layout.addWidget(QLabel("预训练的GPT模型路径"))
        layout.addWidget(pretrained_s1)

        btn_start_tool = QPushButton("启动工具")
        btn_start_tool.clicked.connect(self.start_tts_tool)
        layout.addWidget(btn_start_tool)

        layout.addWidget(QLabel("1A-训练集格式化工具"))
        
        inp_text = QLineEdit(r"D:\RVC1006\GPT-SoVITS\raw\xxx.list")
        inp_wav_dir = QLineEdit(r"D:\RVC1006\GPT-SoVITS\raw\xxx")
        gpu_numbers1a = QLineEdit("%s-%s" % (gpus, gpus))
        bert_pretrained_dir = QLineEdit("GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
        button1a_open = QPushButton("开启文本获取")
        button1a_open.setProperty("class", "primary")  

        button1a_close = QPushButton("终止文本获取进程")
        button1a_close.setProperty("class", "primary")  

        info1a = QTextEdit("文本进程输出信息")
        
        layout.addWidget(inp_text)
        layout.addWidget(inp_wav_dir)
        layout.addWidget(QLabel("1Aa-文本内容"))
        layout.addWidget(gpu_numbers1a)
        layout.addWidget(bert_pretrained_dir)
        layout.addWidget(button1a_open)
        layout.addWidget(button1a_close)
        layout.addWidget(info1a)

        layout.addWidget(QLabel("1B-微调训练"))
        
        batch_size = QSlider()
        total_epoch = QSlider()
        text_low_lr_rate = QSlider()
        save_every_epoch = QSlider()
        if_save_latest = QCheckBox("是否仅保存最新的ckpt文件以节省硬盘空间")
        if_save_every_weights = QCheckBox("是否在每次保存时间点将最终小模型保存至weights文件夹")
        gpu_numbers1Ba = QLineEdit("%s" % (gpus))
        button1Ba_open = QPushButton("开启SoVITS训练")
        button1Ba_open.setProperty("class", "primary")  

        button1Ba_close = QPushButton("终止SoVITS训练")
        button1Ba_close.setProperty("class", "primary")  

        info1Ba = QTextEdit("SoVITS训练进程输出信息")
        
        layout.addWidget(batch_size)
        layout.addWidget(total_epoch)
        layout.addWidget(text_low_lr_rate)
        layout.addWidget(save_every_epoch)
        layout.addWidget(if_save_latest)
        layout.addWidget(if_save_every_weights)
        layout.addWidget(gpu_numbers1Ba)
        layout.addWidget(button1Ba_open)
        layout.addWidget(button1Ba_close)
        layout.addWidget(info1Ba)

        layout.addWidget(QLabel("1C-推理"))
        
        GPT_dropdown = QComboBox()
        SoVITS_dropdown = QComboBox()
        gpu_number_1C = QLineEdit(gpus)
        refresh_button = QPushButton("刷新模型路径")
        refresh_button.setProperty("class", "primary")  

        if_tts = QCheckBox("是否开启TTS推理WebUI")
        tts_info = QTextEdit("TTS推理WebUI进程输出信息")
        
        layout.addWidget(GPT_dropdown)
        layout.addWidget(SoVITS_dropdown)
        layout.addWidget(gpu_number_1C)
        layout.addWidget(refresh_button)
        layout.addWidget(if_tts)
        layout.addWidget(tts_info)

        tab.setLayout(layout)
        return tab

    def create_tab_voice_conversion(self):
        tab = QWidget()
        layout = QVBoxLayout()

        label_under_construction = QLabel("施工中，请静候佳音")
        layout.addWidget(label_under_construction)

        tab.setLayout(layout)
        return tab

    def start_data_tool(self):
        print("Starting data tool...")

    def start_tts_tool(self):
        print("Starting TTS tool...")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GPTSoVITSGUI()
    sys.exit(app.exec_())
